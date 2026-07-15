# Dog-LRM 项目 Skills 总览

本目录把项目的三类沉淀固化成 skill，分层结构对应 Agent Skills 的 Progressive Disclosure：

| 层级 | 内容 | 本项目对应 | Token 成本 |
|---|---|---|---|
| **L1** | frontmatter `name` + `description`（触发词 + 排除项） | 每个 skill 的入口，常驻上下文 | ~几十 token/skill |
| **L2** | SKILL.md 正文（规则骨架 + 为什么，< 500 行） | 流程、红线、判断标准，附真实翻车记录 | 命中时才加载 |
| **L3** | 引用文档（按需读取） | papers.md / design-log.md / pitfalls.md / recipe.md | 几乎无上限 |
| **L4/脚本** | scripts/ 下可执行文件 | 冒烟测试、8 卡启动、四版评测 | ≈0（执行不读取） |

## 三个 skill 的分工

| skill | 沉淀什么 | 类型 |
|---|---|---|
| [`surveying-dog-avatar-papers`](surveying-dog-avatar-papers/SKILL.md) | 论文阅读整理：每篇的核心机制 + 对本项目的启示 | 知识类 |
| [`iterating-dog-lrm-design`](iterating-dog-lrm-design/SKILL.md) | 方法设计迭代史：两分支路线、每次架构决策的因果链、废弃路线的教训 | 决策记录类 |
| [`training-dog-lrm-experiments`](training-dog-lrm-experiments/SKILL.md) | 实验规程：环境、训练配方、监控信号、评测协议 | 规程类 |

## 维护约定

- 新实验结论 / 新论文笔记 / 新踩坑：**当场**写进对应 skill 的 L3 文件（趁上下文还在，不要事后凭记忆整理）。
- SKILL.md 正文只放"会改变下一步行动"的规则；细节一律下沉 L3。
- 改动走 git PR review。
