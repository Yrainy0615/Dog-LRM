# 训练配方与 checkpoint 台账

## 当前最优配置（Branch A w/o fur 最终形态，pa300k）

```bash
torchrun --nproc_per_node=8 train_dog_lrm_ddp.py \
    --root received_data_from_Pinstudio_20260424/unzipped/0423 \
    --arch v2 \                  # DogLRMv2: 512/8L MM-DiT (dit_v2.py)
    --surf_samples 300000 \      # 随机表面锚点(seed=0 固定), 替代细分晶格
    --head_boost 4 \             # head/muzzle/ear 采样密度 ×4
    --rasterize_mode antialiased \  # Mip-Splatting opacity 补偿, 反斑点核心
    --scale_ratio 8 \            # 放行贴面扁盘 splat (默认4会堵死"摊开")
    --proj_feat 1 --ref_res 896 \   # pixel-aligned condition (~4mm 粒度)
    --scale_div 2 \              # 4MP 监督 (1748×2320), 需 cache_s2
    --k_sup 12 --workers 0 \
    --lr 1e-4 --warmup_iters 200 --lr_final_ratio 0.05 \
    --scale_clip_warmup 0 --lpips_start 0 \   # warm-start fine-tune 必带
    --init_ckpt <上一版 model.pt> \
    --iters 6000 --save_every 1000 --out exps/<name>
```

from-scratch 时改：`--lr 2e-4 --warmup_iters 500 --iters 20000`，去掉
`--scale_clip_warmup 0 --lpips_start 0`（用默认 warmup/500）。

## 关键 flag 语义

| flag | 作用 | 注意 |
|---|---|---|
| `--surf_samples N` | N 个面积加权随机表面锚点（`build_surface_sampler`, seed=0） | 训练/推理锚点自动一致；`--proj_feat` 依赖它（要面索引算法线） |
| `--rasterize_mode antialiased` | 亚像素 GS 的 opacity 会被按比例压低 | 换模式 = 换 opacity 语义，跨模式 warm-start 需 fine-tune 适应 |
| `--proj_feat 1` | 锚点投影参考图采样浅 CNN 特征+RGB，法线可见性门控 | 新增 `ref_cnn`/`proj_in` 权重；旧 ckpt 加载时 missing 8 个键是正常的 |
| `--n_subdiv K` | 细分晶格锚点（旧方案，62k=2 / 249k=3） | 已被 surf_samples 取代；复现旧版本时才用 |
| `--k_sup` | 每 scene 每步监督视角数 | 12 视角 × b_local 4 = 单卡每步 48 次渲染 |
| `--scale_div` | 监督分辨率 = 原图/此值 | 用 2 前先确认 cache_s2 存在（`preprocess/cache_train_views.py --scale_div 2 --procs 32`） |

## Checkpoint 台账（exps/）

| run | 内容 | 状态 |
|---|---|---|
| `dog_lrm_v2_full` | v2 62k @1/8res, 20k it | 保留（对比 baseline A） |
| `dog_lrm_v2_full_run1_collapsed` | lr 5e-4 崩溃现场 | 保留作教材 |
| `dog_lrm_v2_hires300k` | 249k 晶格 @4MP | 保留（对比 C；有点阵伪影） |
| `dog_lrm_v2_pa300k` | 300k+AA+PA 最终版 | **当前最优** |
| `dog_lrm_v2_hires62k` | 62k @4MP（对比 B，复训版） | 见 ff_comparison |
| `dog_lrm_full` | v1 旧架构 | 与现代码不兼容（mmdit 重构前），勿加载 |

## 速度/显存参考（8×48G, b_local 4, k_sup 12）

| 配置 | 峰值显存 | 步时 |
|---|---|---|
| 62k @1/8 | ~44 GB | ~1.7 s |
| 249k @4MP (chunked refine) | 23.8 GB | ~9.6 s |
| 300k+PA @4MP | 30.3 GB | ~10 s |
