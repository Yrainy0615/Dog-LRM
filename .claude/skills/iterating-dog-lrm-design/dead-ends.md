# 废弃路线与死因

评审新提案前先查这里。同构方案不要重试；若确要重试，必须先说明"当年的死因如今为何不成立"。

## PatchLocks fur prior（2026-07-07 废弃，资产已删）
- 方案：平面毛发 patch 平铺到曲面身体上。
- 死因：**平面 patch 贴曲面是结构性死路**——曲率处 patch 间接缝/拉伸不可修，
  任何"贴片式"纹理方案在四足动物身体上都会遇到同样问题。

## Kotori 模板 GS（2026-07-07 废弃删除，回归 feed-forward）
- 方案：GaussianHaircut per-scene 优化的模板 GS。
- 死因：per-scene 优化与产品的单图 feed-forward 目标不符。
- 保留的教训（修 bug 时仍有效）：custom-ssim / premultiplied-mask / tool-kills-bg
  三个发散 bug；packed-absgrad densification 的正确写法。

## Fur V3 refine-only（消融后放弃）
- 方案：只做 refine 阶段的毛发细化。
- 死因：消融显示方案 #2 在捕获指标上净负收益（FUR_V3_PLAN §7.3 v3.1）。

## 自由 Gaussian + strand-flow 结构损失（train_fur_free.py，搁置）
- 方案：free Gaussians + 3D strand-flow 对齐/各向异性/一致性损失。
- 状态：Gabor 提取器质量差导致 flow 监督不可靠；等更好的 flow 来源再评估。

## 提高坐标 PE 频率来"区分相邻 GS"（2026-07-14 否决，未实施）
- 死因：PE 只提供画高频图案的基底，**不携带图像信息**——相邻 GS 该是什么颜色
  PE 说不出来，只会让 LPIPS 幻觉出更细的假纹理。信息瓶颈在 condition 的
  外观粒度，正解是 pixel-aligned 采样。

## 16MP 原生分辨率监督（2026-07-13 否决，未实施）
- 死因：300k GS 在 16MP 下每个只摊 ~260 像素，1–4MP 后监督信息饱和；
  64 倍渲染开销买不到质量。经验公式：监督像素数 ≈ GS 数 × (10~15) 即够。
