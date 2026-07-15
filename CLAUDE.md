# CLAUDE.md

## Project skill routing (Dog-LRM)

本仓库的领域知识沉淀在 `.claude/skills/` 三个 skill 里，按任务类型触发（详见各自 L1 description）：

- 文献/方法机制问题（"XX 论文怎么做的"、比较路线、设计新模块前查相关工作）→ `surveying-dog-avatar-papers`
- 设计决策与提案评审（"为什么当初这么设计"、"XX 试过吗"、新架构/表示/损失提案）→ `iterating-dog-lrm-design`（提案必须先过 dead-ends.md）
- 跑训练/评测/排障（开训练、渲染对比、OOM/Ninja/shm 报错、斑点颗粒、loss 异常）→ `training-dog-lrm-experiments`（任何全量训练前必须冒烟）

一个任务常横跨多个（如"加个新 loss 再训一版"→ 先 ② 评审再 ③ 执行）。新结论当场沉淀回对应 skill 的 L3 文件。

---

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
