# 论文笔记（按主题分组）

格式：**机制一句话 + 对本项目的启示**。新增条目保持 ≤5 行。

## 1. 体模型 / 模板

### SMAL / SMBLD (BARC 版)
- 机制：四足动物参数化网格（3889 顶点，betas + betas_limbs(7) + 35 joints LBS）。
- 启示：几何表达力有限（尤其长毛狗轮廓），必须给 GS offset 自由度补偿
  （我们用 tanh 限幅 offset_max=0.15 + ACAP 正则）。skin weights 可直接用于
  身体部位分组（head/muzzle/ear = joints 15,16,32,33,34）——头部加密采样就靠它。

### BARC
- 机制：单图狗 SMAL 拟合（shape + pose + camera）。
- 启示：推理时的 SMAL fit 和相机来自 BARC——这让 pixel-aligned 投影采样在
  单图输入下可行（其他 template-free 方法没有这个便利）。

## 2. Feed-forward GS 重建（condition 设计是核心差异）

### GS-LRM / Splatter Image / LGM / pixelSplat / MVSplat（pixel-aligned 家族）
- 机制：每个 Gaussian 绑定一个图像像素/patch，per-pixel 特征直接解码 GS 参数。
- 启示：**外观 condition 粒度 = 像素级**，颜色近乎"从图里抄"，这是它们纹理锐的根本。
  我们的教训：只靠 16×16 DINO token（~10cm/token）做 condition，模型必然输出均一
  材质，然后用几何造假高频（斑点）。修复 = 投影采样高分辨率特征（plan A，已实装）。

### LHM（本仓库的祖先，人体版）
- 机制：SMPL 表面点 token × 图像 token 的 MM-DiT joint attention → per-point GS。
- 启示：①架构同构，我们 v2 就是它的现代化（真 adaLN-Zero/QK-RMSNorm/SwiGLU/RoPE）；
  ②它对头部单独开一路 token（SD3BodyHeadMMJoint block）——承认全身 token 粒度
  不够重要区域用，与我们 head_boost 采样加密思路一致。

### Animatable Gaussians / GaussianAvatar（UV-CNN 家族）
- 机制：GS 属性在 UV 空间用 2D CNN 预测，一个 texel 一个 Gaussian。
- 启示：conv 局部性天然给每个 GS 独立邻域特征，是 pixel-aligned 之外的另一条
  高频路线；如果 plan A 不够，UV-CNN 是备选大重构（512² UV ≈ 26 万 GS）。

## 3. 毛发 / strand 生成（fur 分支）

### DiffLocks
- 机制：单图 → 头皮 UV 上的 strand latent 扩散生成 3D 发丝。
- 启示：fur 分支主线参照——合成 (image, 3D-strand) GT 数据训练 feed-forward
  strand 生成；瓶颈在毛发几何初始化（blender_fur_dataset.py 造数据）。

### GaussianHaircut（kotori 管线）
- 机制：per-scene 优化的 strand-aligned 3DGS 头发重建。
- 启示：曾用作 D-SMAL 狗的 fur 优化器（NeuralFur teacher）；教训记在
  iterating-dog-lrm-design 的 dead-ends.md（custom-ssim/premult-mask 发散 bug、
  packed-absgrad densification）。

### NeuralFur teacher + Splatter-Image student（V11 路线）
- 机制：per-scene 优化出高质量 fur GS 作 teacher，蒸馏到 feed-forward student。
- 启示：cov_gate 暗洞修复 + geometric shrink 在 train_fur_final.py；teacher-student
  是 fur 分支的备选路线之一，最终线是 DiffLocks 式参数化 strand。

## 4. 抗锯齿 / 特征上采样（w/o fur 分支的质量关键）

### Mip-Splatting（gsplat 的 antialiased 模式）
- 机制：3D/2D 低通滤波 + 按 dilation 能量误差比例补偿 opacity。
- 启示：**classic 模式的固定 0.3px dilation 让亚像素 GS 免费获得覆盖**——这是
  "GS 学不开、缩成斑点"的作弊均衡根源。低分辨率训练→高分辨率渲染的颗粒感、
  以及 300k 锚点下的点阵化，都靠 antialiased 拆掉这个均衡（实测有效）。

### FeatUp / JAFAR（特征上采样）
- 机制：用原图做引导把低分辨率 ViT 特征上采样到任意分辨率（JAFAR: cross-attn 式）。
- 启示：可作为 pixel-aligned 插槽的语义增强版（替换浅 CNN 特征图）；但纹理复制
  需要的是低层外观信号，原图 RGB + 浅 CNN 已覆盖 80/20，JAFAR 留作遮挡边界
  语义一致性不足时的 ablation。输出通道 = DINO 维度（1024），上采样后需先降维。
