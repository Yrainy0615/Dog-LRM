# 踩坑排查（按报错/现象索引）

## `RuntimeError: unable to write to file </torch_...>: No space left on device`
容器 /dev/shm 只有 64M，DataLoader spawn worker 走 POSIX shm 传张量必爆。
**修法**：`--workers 0`（cache_s2 下 IO 很轻，瓶颈在 GPU，吞吐损失可忽略）。

## `RuntimeError: Ninja is required to load C++ extensions`
用绝对路径调 torchrun 时子进程 PATH 没有 conda env 的 bin。
**修法**：`PATH=/home/yyang/.conda/envs/dog-lrm/bin:$PATH` 前置。

## `ImportError: cannot import name 'csrc' from 'gsplat'`
gsplat JIT 找不到已编译扩展。
**修法**：`TORCH_EXTENSIONS_DIR=<workspace>/.torch_ext_lhm`（内含 gsplat_cuda.so）。

## 日志 `off=0.2398` 恒定 + loss 卡在 0.2~0.3
全体 offset tanh 饱和（0.15×√3−0.02），LR 过热崩溃，**不可自愈**。
**修法**：停训，从最近健康快照降 LR 重启；预防 = lr≤2e-4 + warmup + cosine。

## 渲染颗粒感/椒盐斑点（低分辨率训的模型在高分辨率下渲染）
亚像素 GS 靠 classic 光栅化 0.3px dilation 免费覆盖，高分辨率下露馅。
**修法**：高分辨率监督 fine-tune + `--rasterize_mode antialiased`。机制详见
iterating-dog-lrm-design/design-log.md 的斑点诊断链。

## 渲染规则点阵/moiré（细分锚点 + 高分辨率）
细分晶格规则排列 + 相邻 GS 颜色 condition 趋同。
**修法**：`--surf_samples` 随机表面锚点 + `--proj_feat 1`。

## torch import 卡死
env 装在 NFS workspace 里会 hang。**修法**：用本地盘 env
`/home/yyang/.conda/envs/dog-lrm`。

## NFS 跨进程读到旧文件（stale read）
workspace NFS 对刚写入的文件跨进程可能读到旧版本。
**修法**：渲染-读取放同进程；勿依赖"写完立刻另起进程读"。

## `pkill -f train_dog_lrm` 把自己杀了
后台 wrapper shell 的命令行里含同样字符串，pkill 会命中它自己。
**修法**：先 `ps` 确认目标 PID，或 pattern 加不会出现在自身命令行里的限定词。

## 加载 ckpt 时 missing keys
- 只 missing `dino.*`：正常（DINO 冻结不存）。
- missing `ref_cnn.*`/`proj_in.*`（8 个）：旧 ckpt 上新 PA 模型，正常
  （proj_in 零初始化，行为无损）。
- missing 其他键：版本不匹配，停下来查 recipe.md 台账。
