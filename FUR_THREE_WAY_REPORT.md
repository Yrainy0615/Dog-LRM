# Dog-LRM 毛发三方案对比 Report

> 2026-06-25 · feed-forward / two-stage / cascade 三条路线流程 + 对比 + 结果
> 详细架构见 `FUR_SUMMARY.md`；本报告是给人看的精简版。

---

## TL;DR

| | 一句话 | 结论 |
|---|---|---|
| **A. Feed-forward** | 一个网络一次前向，body+fur 两个头一起预测 | **质感最好、能泛化**，但毛烘进 body **不可分解** |
| **B. Two-stage** | 先训 body 冻结，再在上面加一层 fur | body 已经把毛烘进去 → fur **无处安放**，加了反而掉点 → **死路** |
| **C. Cascade** | skin↔fur 耦合优化，毛覆盖处 body 退成 undercoat | **可分解 + 可仿真**，composite≈GT；目前**逐场景优化**（上限探针），非前向 |

外加本次新增：**NeuralFur（真·GaussianHaircut）** 属于 cascade 家族的逐场景优化器，我们把它的**几何**拿来 + cascade 的**上色**做了融合。

---

## 三条流程

### A) Feed-forward joint（v6 → v9）
```
单图 ──DINOv2 backbone──┬─ body head → D-SMAL 表面高斯
                        └─ fur head  → per-root strand(长度/方向TBN/droop/curl/opacity)
                        (+ v9: pixel-aligned splatter 残差 + 对抗 → 质感最佳)
        一次前向，body+fur 同时出
```
- ✅ 泛化（单图→毛）、**质感最好**（v9 sharp、真实）
- ✗ 毛**烘进 body**，拉不出可动的毛层；脸部需 `nofur` mask

### B) Two-stage（先冻结后加层）
```
Stage-1: 训 body 高斯(含毛的照片) ──冻结──► Stage-2: 在冻结 body 上训一层 fur
```
- ✗ Stage-1 用含毛照片训 → body **已经把毛色烘进去** → Stage-2 的毛**冗余、没地方住** → 要么纯装饰要么**抬高 L1**
- **关键教训**：不让 skin 退让就解耦 → 毛冗余。直接催生了 C。

### C) Cascade（耦合 + recession，逐场景）
```
body 与 fur 联合优化：
  ├─ recession: 毛覆盖处 body 退成深色 undercoat (w_recede 先验强制)
  ├─ fur 承载毛色: 半透明 strand，partings 处露出 undercoat (物理正确→可仿真)
  └─ 几何: v6-flow strand 先验(梳理切向流+droop+curl)，脸部 w_face 排除
        多视角 photometric 优化 → composite≈GT
```
- ✅ **可分解**(undercoat↔fur)、**可仿真**(sway 时 partings 透出 undercoat)、composite≈GT
- ✗ **逐场景**优化(非前向、不能单图泛化)；质感比 v9 软；static L1 0.010–0.015

### （新）NeuralFur + cascade 上色融合
```
NeuralFur 优化器 ── strand 几何(梳理短毛) ──┐
                                            ├─ 我们 cascade 的邻域平滑 albedo-query 上色 ─► 干净彩色毛 + sway
我们的 body 高斯 ── 皮肤颜色 ───────────────┘
```
- NeuralFur 原版每根 strand 颜色独立 → speckle/偏亮；只取**几何**，颜色走我们的 query → speckle 消失

---

## 对比表

| 轴 | A) Feed-forward | B) Two-stage | C) Cascade |
|---|---|---|---|
| skin↔fur | 并行头、无耦合 | 冻结、硬解耦 | **耦合 + recession** |
| 外观住在哪 | body(烘毛) | body(烘毛) | **拆分: undercoat + fur** |
| 可分解 | ✗ | ✗ | **✓** |
| 毛的角色 | 装饰 | 冗余 | **承载毛色** |
| 可仿真 | 弱 | 弱 | **✓ (sway 成立)** |
| 质感 | **最好 (v9)** | n/a | 较软、需锐化 |
| 单图泛化 | **✓** | 部分 | ✗ (逐场景) |
| static L1 | ~body-shell(低) | 加毛抬高 | 0.010–0.015 |
| 脸部 | nofur mask | — | w_face 排除 |

> **已证伪的执念**：结构化/可仿真毛在 static L1 上**赢不了**光滑 body-shell —— `L1<0.01` 是**body-shell 指标**。毛的价值 = 可分解 + 动态 + 感知锐度，**不是** static L1。

---

## 结果

- `exps/dog_lrm_fur_v9/it02500.png` — **A) Feed-forward v9**：质感最佳（但毛烘进 body）
- `exps/fur_v11_kotori_clean/cmp.png` + `00085-kotori_decomp.png` — **C) Cascade**：可分解（undercoat↔fur 分层）+ sway
- `exps/neuralfur_final/` — **NeuralFur 几何 + cascade 上色**：colored 静帧 + 转台 + 毛 sway 动效
- B) Two-stage 无可展示结果（死路，加毛掉点）

---

## 结论 / 推荐方向

1. **当前最佳质感 = A(v9)，最佳分解 = C(cascade)** —— 二者尚未结合。
2. **目标 = v9 质感 + cascade 可分解**：把 v9 的 fur 头作为 Stage-2 毛层，root 在 Stage-1 裸皮上，加 recession 让 body 变 undercoat，脸排除。
3. **部署 = 前向化 cascade**：训一个图像条件头预测 {fur op/色/几何残差 + body recession}，用逐场景 cascade 结果 + 多视角监督做 GT → A 的泛化 + C 的可分解。
4. NeuralFur 这次证明了"几何先验 + 我们上色"这条融合可行，可作为 cascade 几何质量的一个外部来源。
