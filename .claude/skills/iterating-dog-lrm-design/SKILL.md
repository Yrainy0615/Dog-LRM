---
name: iterating-dog-lrm-design
description: Dog-LRM 方法设计的决策记录——最终双分支路线（w/o fur 单 GS head + fur
  参数化毛发）、每次架构迭代的因果链、废弃路线及其教训。当用户讨论方法设计、提出新的
  网络/表示/损失改动、问"为什么当初这么设计"、"XX 路线试过吗"，或提到 v1/v2 架构、
  offset 自由度、斑点诊断、condition 设计、PatchLocks、kotori 等历史方案时，
  务必先读本 skill。评审任何新设计提案前必须先查 dead-ends.md 防止重走废路。
  不用于：论文机制查询（用 surveying-dog-avatar-papers）、跑实验的操作规程
  （用 training-dog-lrm-experiments）。
---

# Dog-LRM 设计决策记录

## 最终路线（2026-07 确认）

Feed-forward 双分支，共享 v2 backbone（512-dim/8 层 MM-DiT + frozen DINOv2-large）：

- **Branch A（w/o fur）**：SMAL 表面锚点 + 限幅 offset 的单 GS head，不单独建模毛。
  当前最终形态 = 300k 随机表面锚点 + antialiased 光栅化 + pixel-aligned condition
  （`dog_lrm/model_v2.py`，run `exps/dog_lrm_v2_pa300k`）。
- **Branch B（fur）**：先由 image feature + fur prior 在 mesh 上生成参数化毛发
  （DiffLocks 式，合成 strand GT），再双 GS head 联合预测 body + fur。未实装。

## 设计评审规则

1. **新提案先过 dead-ends.md**——废弃路线都有明确死因，同构方案不要重试。
2. **改 condition / 表示前先读 design-log.md 的"斑点诊断链"**——那是本项目最完整的
   一次"现象→机制→修复"推理，多数渲染质量问题能在里面找到同源解释。
3. **每次架构决策当场记录**：design-log.md 追加一节，必须写"现象 / 根因 / 动作 /
   验证结果"四段，不写纯结论。
4. **保持 v1 (`dog_lrm/model.py`) 不动**——fur 分支脚本依赖它；v2 系改动全部走
   `model_v2.py` / `dit_v2.py` 新文件。

## 为什么这样做

- 决策的"因果链"比"结论"值钱：斑点问题我们三次迭代才根治（AA → 随机锚点 →
  pixel-aligned），每次都是靠把上一次的残留现象归因到新机制。只记结论的话，
  下一个同类问题还要重推一遍。
- 废路教训曾经真实救过时间：PatchLocks（平面 patch 贴曲面）死因明确后，
  后续两次"贴片式"提案直接被否，各省一周。

## 参考

- 完整设计迭代日志（v1→v2→分辨率→数量→抗斑点→condition）：design-log.md
- 废弃路线与死因：dead-ends.md
