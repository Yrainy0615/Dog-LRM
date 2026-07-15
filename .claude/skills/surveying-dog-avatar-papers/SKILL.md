---
name: surveying-dog-avatar-papers
description: 狗 3D 重建/毛发生成项目的论文知识库——每篇相关工作的核心机制与"对本项目的启示"。
  当用户讨论相关文献、问"XX 论文怎么做的"、比较方法路线（feed-forward GS 重建、
  pixel-aligned、UV-CNN avatar、毛发 strand 生成、特征上采样），或提到 GS-LRM、LHM、
  Splatter Image、Mip-Splatting、SMAL、BARC、DiffLocks、GaussianHaircut、FeatUp、JAFAR
  等论文名时，务必先读本 skill 再回答。设计新模块前必须查 papers.md 确认相关工作的做法。
  不用于：本仓库代码怎么跑（用 training-dog-lrm-experiments）、设计决策史（用 iterating-dog-lrm-design）。
---

# 狗 Avatar 论文知识库

## 使用规则

1. **回答文献问题前先查 papers.md**——里面每篇都带"对本项目的启示"，直接引用比凭
   参数化记忆可靠（论文细节容易记串，启示是结合过本项目约束的结论）。
2. **设计新模块前先对号入座**：papers.md 按主题分组（体模型 / feed-forward GS /
   毛发 / 抗锯齿与特征上采样），新想法先看对应组里是否已有被验证/被否定的同构方案。
3. **读了新论文要当场沉淀**：往 papers.md 对应主题组追加条目，格式固定为
   `机制一句话 + 对本项目的启示一句话`，不写超过 5 行。

## 为什么这样组织

- "启示"比"摘要"有用：摘要网上有，结合本项目（单图 feed-forward、SMAL 模板、
  109 个 studio scene、fur/w-o-fur 双分支）的判断没有。
- 曾经的教训：pixel-aligned 条件化在 GS-LRM/Splatter Image 里是标配，但我们直到
  出现"斑点感"才意识到自己的 condition 粒度差了 30 倍——如果设计前先对表，
  这个弯路可以省掉。

## 参考

- 全部论文笔记（按主题分组）：papers.md
