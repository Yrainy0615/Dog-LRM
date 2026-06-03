# Animal-LHM 改造计划（Pet 图像 → Gaussian + SMAL）

> 基线：LHM (arXiv 2503.10625) 推理代码。目标：单张宠物图 → 输出 3D Gaussian Avatar，并**额外回归 SMAL 参数**。
> 数据：自有宠物数据集，COLMAP 格式（多视角 image + sparse + 相机），**无 SMAL 拟合、无 mask**。

---

## 0. 关键事实 / 约束

1. **repo 无训练代码**：`LHM/runners/` 只有 `infer/`；无 loss 实现；config 里的 `video_human_flame` dataset 类不存在。→ 训练管线从零自建。
2. **预测 SMAL 但无 GT 的张力**：主干需要 SMAL 做 canonical 采样 + LBS；回归头需要 SMAL GT 监督。→ **P0 离线 SMAL 拟合**同时充当“训练期条件输入”和“回归 pseudo-GT”。
3. **编码器**：去掉 Sapiens（人体专用），单路 **frozen DINOv3**；transformer 从 `sd3_mm_bh_cond` 退回普通 `cond`（去掉 body-head 双流）。
4. **GS decoder（`GSLayer`）保持不动**；点采样先整体单分支（去掉 head/body 双采样），后续再做 fur/body。

---

## 1. 架构映射（Human → Animal）

| 组件 | LHM 现状 (文件) | 改造 |
|---|---|---|
| Body model | `rendering/smpl_x.py`, `smpl_x_voxel_dense_sampling.py` | 新建 `rendering/smal.py` 的 `SMALModel`：35 joints、betas(shape PCA)、pose 35×3 axis-angle、global rot/trans；无 expr/jaw/eye/hands/FLAME。提供 `v_template, lbs_weights, shapedirs, posedirs, J_regressor, parents, faces`，实现 `get_query_points / get_neutral_pose / transform_to_posed_verts / lbs`。canonical 用动物中性站姿替换“大 pose”。 |
| 点采样 | `dense_sample()` (`body_face_ratio=3` 分 body/head) | 单分支整体 dense sampling → 一个 `PointEmbed`+MLP。part mask (`is_body/is_upper_body/is_rhand/is_hands`) 先关闭或重定义为动物部位（躯干/四肢/尾/头）。 |
| Coarse encoder | `dinov2_fusion`（可训练） | 新建 `encoders/dinov3_wrapper.py`，**frozen**，输出 `[B, N_tok, C]`。 |
| Fine encoder | `sapiens_warpper.py`（frozen） | **删除**，单路 DINOv3。 |
| 主模型 | `modeling_human_lrm.py` | 新建 `ModelAnimalLRM`：`forward_encode_image` 单路、去 head 分支、去 `use_face_id/facesr/stylegan`；`cond_dim`=DINOv3 维度；**新增 SMAL 回归头**。 |
| Transformer | `transformer.py` `sd3_mm_bh_cond` | 改 `transformer_type: "cond"`（去 motion/head modulation）；`cond_dim` 对齐 DINOv3。 |
| GS decoder | `gs_renderer.py` `GSLayer` | **不动**，只对齐维度。 |
| 渲染 | `GS3DRenderer` | 保留；把 `smpl_type` 分支接入 `SMALModel`。 |
| 训练 | — | 全新：dataset / losses / Accelerate loop。 |

**SMAL 回归头**（满足“额外预测 SMAL”）：从 DINOv3 全局 token（或 pooled image_feats）→ MLP → `{global_rot, pose(35×3), betas, trans, (limb scale)}`。
- 训练期用 **teacher-forcing**：主干条件 + LBS 用 **P0 拟合的 SMAL**（稳定收敛）；回归头单独用 pseudo-GT + 多视角重投影监督。
- 推理期：回归头预测 SMAL → 喂主干生成 Gaussian。可选第二轮 self-improving（预测 SMAL 渲染 loss 反传）。

---

## 2. 数据管线（P0，最大前置工作）

输入：每个 pet 的 COLMAP（`images/` + `sparse/0/{cameras,images,points3D}.bin`）。
产出（训练所需 schema，每帧一条）：
```
{
  image, mask,                       # RGB + 前景 (matting)
  c2w (4x4), intrinsics (4x4),       # 由 COLMAP 转换
  smal_params: {global_rot, pose[35,3], betas[B], trans, scale},  # pseudo-GT
  keypoints_2d (可选, for reproj loss)
}
```
步骤：
1. **COLMAP → 相机**：内外参转 c2w / intrinsics（注意 COLMAP world≠SMAL canonical，需对齐尺度/朝向）。
2. **Matting / 分割**：复用 repo 里 `engine/BiRefNet` 或 `engine/SegmentAPI` 出前景 mask（mask loss 必需）。
3. **动物 2D 关键点**：用动物 pose 估计（如 ViTPose-animal / AP-10K 模型）出关键点，供 SMAL 拟合 + 训练重投影。
4. **SMAL 多视角拟合（analysis-by-synthesis）**：参考 SMALify / WLDO / BARC，用「关键点重投影 + 轮廓 IoU + 多视角一致 + 形状/pose 先验」优化 `{pose, betas, trans, scale}`。多视角 + 已知相机使其比单图稳得多。输出 = pseudo-GT。
> 已定（见 §6）：BARC 狗模型 + 其 PyTorch SMAL layer，复用作拟合与 renderer body model。

---

## 3. 分阶段实施

- **P0 资产 + 数据**：BARC 已 clone，SMBLD 狗模型文件**已在手**（`third_party/barc_release/data/smal_data/`），无需 license 下载。写 `SMALModel`（先纯 nearest-vertex LBS，不上 voxel skinning）；搭 COLMAP→相机/mask/keypoint/SMAL 拟合前处理（见 `preprocess/README.md`，Stage 1 已完成）；产出训练 schema。**验收**：能可视化 canonical SMAL mesh + 拟合 mesh 叠加到图像对齐。
- **P1 模型改造**：`dinov3_wrapper`(frozen) → `ModelAnimalLRM`（单编码器、单 dense sampling、去人脸件、加 SMAL 回归头） → 维度自检。**验收**：随机权重能 forward 出 Gaussian 并渲染一张图、回归头输出形状正确。
- **P2 训练管线 + 单 pet overfit**：自建 dataset/loss/Accelerate loop。loss = masked L1 + LPIPS + mask + (offset/scale 正则) + SMAL 回归(pseudo-GT + reproj)。先在 1 个 pet 上 overfit 跑通。
- **P3 全量训练**：调 part 分组正则（动物部位）；调权重；评估多视角 PSNR/LPIPS + SMAL 重投影误差。
- **P4（可选）**：fur/body 双分支点采样；SMAL 回归 self-improving 闭环；voxel skinning（长毛/裙摆类形变）。

---

## 4. Loss（去掉人体专用项）

保留：`masked_pixel (L1)`、`perceptual (LPIPS)`、`mask`、`offset_loss`/`ball_loss`（part 分组改动物部位）。
去掉：`face_id_weight`（ArcFace 人脸）、`facesr`、`stylegan2_prior`。
新增：`smal_param_loss`（回归头 vs pseudo-GT）+ `keypoint_reproj_loss`（预测 SMAL 投影 vs 2D kp）。

---

## 5. 主要风险

1. **无 SMAL GT** → 回归头质量上限取决于 P0 拟合质量；拟合差则全链路差。多视角约束是关键缓解。
2. **训练代码从零** → 工作量大且需复现 LHM 训练细节（LBS-anchored GS、canonical 对齐、bf16/grad-ckpt）。
3. **SMAL↔COLMAP 坐标系对齐**（尺度/朝向/原点）易出 bug，需早期可视化验证。
4. **DINOv3 frozen 特征**对动物纹理/几何是否够用，可能需要解冻末层或加适配层。
5. 物种多样性（猫/狗/体型差异）下 betas 先验与 canonical 站姿的选择。

---

## 6. 已锁定决策

| # | 决策 | 选定 | 说明 |
|---|---|---|---|
| D1 | Body model | **BARC 狗模型** | 数据以狗为主；35 joints + 尾巴/四肢建模好，PyTorch 实现成熟（`barc_release` 的 SMAL layer 复用做拟合 + renderer body model）。参数 = `{global_rot, pose[35,3], betas(shape PCA), betas_limbs(肢长缩放), trans}`。 |
| D2 | SMAL 角色 | **离线拟合 GT + SMAL decoder 回归头** | P0 先离线拟合出 pseudo-GT；主干 Gaussian 分支训练期 **teacher-forcing**（用 pseudo-GT 当条件）；挂 SMAL decoder 用 **pseudo-GT 监督**（参数 L2 + 关键点重投影）。推理期 decoder 预测 → 喂主干。**非纯自监督。** |
| D3 | 拟合 / 监督闭环 | **多视关键点 + 轮廓 IoU + 多视一致 + pose/shape 先验** | 充分利用多视角 COLMAP + 已知相机，拟合更稳。 |
| D4 | Coarse encoder | **DINOv3 ViT-L/16, frozen** | 性价比甜点 ~300M；删除 Sapiens fine 路。 |
| D5 | canonical 姿态 | **SMAL 零位四足站姿，四肢略外展** | 避免腿间自接触；工程默认，后续可微调。 |

### 待定子项（不阻塞 P0/P1，实现时定）
- 动物 2D 关键点估计器：默认 **AP-10K-trained ViTPose-animal**。
- SMAL decoder 参数向量是否含 `betas_limbs`；主干是否后期从 teacher-forcing 过渡到 decoder 预测。
