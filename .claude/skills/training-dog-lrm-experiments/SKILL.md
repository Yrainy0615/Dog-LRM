---
name: training-dog-lrm-experiments
description: 在本仓库训练/评测 Dog-LRM feed-forward 模型的操作规程。当用户要求
  "开一版训练"、"全卡训练"、fine-tune、渲染对比、算 PSNR/LPIPS，或出现训练崩溃、
  loss 不降、CUDA OOM、DataLoader 报 No space left on device、Ninja is required、
  gsplat import 报错、渲染有斑点/颗粒时，务必使用本 skill。开任何训练前必须先跑
  冒烟测试。不用于：设计层面的改动讨论（用 iterating-dog-lrm-design）、
  fur 分支旧脚本（train_fur_* 系列有各自流程）。
---

# Dog-LRM 实验规程

## 红线（违反过、都付出过代价）

1. **全量训练前必须冒烟**：`scripts/smoke_test.sh`（单卡 2 iter + 显存峰值报告，
   ~3 分钟）。不冒烟直接占 8 卡的后果：我们出过 3 连败（shm 爆、ninja 缺失、
   扩展缓存未指向）。
2. **环境三件套必须带**（launch 脚本已内置，手写命令时别漏）：
   - conda env：`/home/yyang/.conda/envs/dog-lrm`（NFS 上的 env 会 hang torch import）
   - `PATH` 前置 env 的 bin（torch JIT 找 ninja 用 PATH）
   - `TORCH_EXTENSIONS_DIR=<workspace>/.torch_ext_lhm`（预编译 gsplat_cuda）
3. **`--workers 0`**：容器 /dev/shm 只有 64M，spawn worker 传张量必爆。
4. **LR 配方不许回退**：v2 (83M) 用 lr ≤2e-4 + `--warmup_iters` + cosine
   （`--lr_final_ratio 0.05`）。5e-4 在 it~5600 崩过一次（offset tanh 全饱和，
   不可恢复）。fine-tune 再减半到 1e-4。
5. **warm-start fine-tune 必须 `--scale_clip_warmup 0`**——否则把训好的 scale
   重新钳回 0.02，等于自残。
6. **提结论前先看 loss 行不是 metric 行**：`metric` 是随机 scene 单视角，
   PSNR 15~38 波动正常；判断训练健康看 `^it` 行的 loss/off/ball
   （崩溃特征：`off=0.2398` 恒定 = tanh 饱和）。

## 标准流程

1. 冒烟：`scripts/smoke_test.sh [extra-flags]`（默认带当前最优配置全套 flag）。
2. 启动：`scripts/launch_ddp.sh <out_name> [extra-flags]`（8 卡 torchrun + 日志 +
   当前最优配置；具体 flag 语义查 recipe.md）。
3. 监控：日志在 `exps/<out_name>.log`；关注 `SKIP step`（非有限梯度跳步）、
   OOM、`off=0.23` 特征。vis tile 每 500 it 存 `exps/<out_name>/itXXXXX.png`。
4. 评测：`scripts/eval_compare.sh`——固定协议（4 scene × 3 view、ref=view20、
   4MP 渲染、白底 masked PSNR + LPIPS-alex@256），产出
   `exps/ff_comparison/EXPERIMENTS.md`。**协议不许改**，改了和历史不可比。
5. 汇报：定量表 + 同狗同视角拼图；注明"指标在训练 scene 上评测"（无 held-out 时）。

## 为什么

- 白底场景 PSNR 对斑点类伪影不敏感（背景占比大）——定性图 + LPIPS 才是有效信号，
  历史上 PSNR 三版几乎持平而视觉差异巨大。
- checkpoint 带 iteration 戳快照：崩溃权重曾覆盖过唯一 model.pt，导致只能重训。
- 逐 view backward（trainer 已内置）：4MP × 48 view 的渲染图一次性攒图必 OOM；
  这是数学等价改写，不是近似。

## 参考

- 当前最优配置与全部 flag 语义、各版本 checkpoint 位置：recipe.md
- 踩坑排查（按报错信息索引）：pitfalls.md
