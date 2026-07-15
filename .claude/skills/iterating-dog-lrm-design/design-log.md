# 设计迭代日志（w/o fur 分支主线）

每节格式：**现象 → 根因 → 动作 → 验证**。时间为决策日。

## 2026-07-12 · v1 → v2 backbone 升级

- 现象：v1（384-dim/4 层 SD3 block）容量与条件化能力存疑；审查发现 v1 的
  SD3MMJointTransformerBlock 里 adaLN gate/shift/scale 全被注释——temb 条件实际是废的。
- 动作：新建 `dit_v2.py`/`model_v2.py`（不动 v1）：真 adaLN-Zero 双流调制、
  RMSNorm + per-head QK-RMSNorm、SwiGLU（8/3 hidden 等参）、图像 token 2D axial RoPE
  （点 token identity + Fourier PE）、512-dim/8 层（24M→83M）。
- 验证：冒烟 forward/backward 通过；零初始化 heads 保证初始 GS 精确贴 posed SMAL 面。

## 2026-07-12 · 训练配方（LR 崩溃教训）

- 现象：lr 5e-4（v1 的配置）训 v2，it~5600 全体 offset tanh 饱和到 ±0.15
  （日志特征 `off=0.2398` ≈ 0.15×√3−0.02 恒定），梯度死区不可恢复；坏权重还覆盖了
  唯一 checkpoint。
- 根因：83M/8 层对 v1 的 LR 过热，坏 batch 一击致命；tanh 饱和无法自愈。
- 动作：lr 2e-4 + warmup 500 + cosine 到 5%；非有限梯度跳步（grad norm 是
  allreduce 后的，各 rank 一致）；checkpoint 加 iteration 戳快照。
- 验证：run2 安全通过原崩溃点，20k iter 无事故。

## 2026-07-13 · 监督分辨率 1/8 → 4MP

- 现象：1/8 分辨率下 rgb L1 在 ~5k it 后平台期（~0.007）；且低分辨率训的模型在
  4MP 渲染下暴露强颗粒感。
- 根因：下采样把高频细节平均掉，模型的"糊"和 GT 的"糊"互相抵消；亚像素 GS 靠
  光栅化 dilation 免费覆盖，高分辨率下露馅成孤立点。原图 3496×4640，对 62k~300k
  GS 而言 4MP（scale_div 2）后监督信息已饱和，16MP 无意义。
- 动作：trainer 改逐 view backward（photometric 梯度先落在 detach 的 GS 叶子上，
  最后一次穿过模型，DDP 单次 allreduce，数学等价）——否则 48 view 的 4MP 渲染图
  攒不下；预建 cache_s2。
- 验证：4MP 下 L1 先抬升（loss 定义变难）后在 200 it 内追平 1/8 的平台值。

## 2026-07-13 · Gaussian 数量 62k → 249k（细分晶格）

- 动作：`--n_subdiv 3`；refine cross-attn 逐点独立 → 分块 + gradient checkpointing
  （`refine_chunk=16384`），显存 40GB+ → 23.8GB。heads 逐点共享 → 权重分辨率无关，
  62k 训的模型直接 249k 热启动（免费继承全部训练）。
- 验证：PSNR 峰值 38+，但出现新伪影 → 下一节。

## 2026-07-14 · 斑点/点阵诊断链（本项目最重要的一次归因）

- 现象：249k 下毛皮出现规则点阵/moiré；用户复述"GS 学不开，斑点感严重"。
- 根因（三层）：
  1. **几何作弊均衡**：classic 光栅化固定 0.3px dilation 让任意小的 GS 免费获得
     ~1px 覆盖 → 缩 scale 无 gap 惩罚；小而不重叠恰好讨好 L1/LPIPS 的锐度需求
     → 收敛到"微小、全不透明、互不重叠"点阵。`opacity_reg 0.5` 逼全不透明、
     `scale_ratio 4` 禁止贴面扁盘，把摊开的正路也堵了。
  2. **规则晶格**：细分顶点排列规则 → 点阵呈 moiré 结构。
  3. **condition 无高频信息**：外观唯一来源是 16×16 DINO token（~10cm/token），
     锚点间距 3mm——相邻数百 GS 共享同一 token，颜色只能局部均一，高频全靠几何造假。
- 动作（对应三层）：① `--rasterize_mode antialiased`（Mip-Splatting opacity 补偿）
  + `scale_ratio` 4→8；② `build_surface_sampler` 面积加权随机表面采样 300k
  （seed 固定，训练/推理一致），head/muzzle/ear 密度 ×4；③ pixel-aligned condition
  （见下节）。
- 验证：pa300k it1000 起斑点/点阵消失，短毛狗纹理接近 GT。

## 2026-07-14 · Pixel-aligned condition（plan A）

- 设计：每个 posed 锚点投影到 896px 参考图，grid_sample stride-4 浅 CNN 特征
  （64 维，~4mm/texel）+ 原图 RGB；法线朝向 + 视锥做可见性门控（不可见置零 +
  flag，退回全局通路）；`proj_in` 零初始化 → warm-start 无损。
- 备选未采用：JAFAR/FeatUp 上采样 DINO（语义高频，留作遮挡边界不足时的 ablation，
  插槽在 `_pixel_aligned`）；UV-CNN（大重构，备选）；提高 PE 频率（只加幻觉基底
  不加信息，否决）；patchify 高分辨率图（patch 粒度差 + 重引规则网格，否决）。
- 验证：见 `exps/ff_comparison/EXPERIMENTS.md` 四版对比。

## Fur 分支（并行历史，详见 memory 与 FUR_*_PLAN.md）

- 主线（re-confirmed 2026-07-07）：DiffLocks 式 feed-forward strand 生成，
  合成 (image, 3D-strand) GT（blender_fur_dataset.py），瓶颈=毛发几何初始化。
- 已验证可用的组件：coat-depth 后处理动画（animate_gs_coatdepth.py）；
  NeuralFur teacher（kotori/GaussianHaircut 优化器）+ Splatter-Image student
  蒸馏（train_fur_final.py，cov_gate 暗洞修复）。
