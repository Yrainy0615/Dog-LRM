#!/usr/bin/env bash
# 固定协议四版对比评测：4 scene × 3 view, ref=view20, 4MP, PSNR+LPIPS,
# 产出 exps/ff_comparison/EXPERIMENTS.md（定量表 + 同狗同视角拼图）。
set -e
cd "$(dirname "$0")/../../../.."   # workspace root
ENV=/home/yyang/.conda/envs/dog-lrm
export PATH="$ENV/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_ext_lhm"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"

"$ENV/bin/python" eval_ff_compare.py
"$ENV/bin/python" exps/ff_comparison/make_report.py
echo "report -> exps/ff_comparison/EXPERIMENTS.md"
