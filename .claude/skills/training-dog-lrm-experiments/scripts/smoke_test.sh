#!/usr/bin/env bash
# 单卡 2-iter 冒烟：验证数据/模型/渲染/backward 全链路 + 报显存峰值 (~3 min)。
# 用法: smoke_test.sh [额外 train_dog_lrm_ddp.py flags，覆盖默认]
set -e
cd "$(dirname "$0")/../../../.."   # workspace root
ENV=/home/yyang/.conda/envs/dog-lrm
export PATH="$ENV/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_ext_lhm"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}"

"$ENV/bin/python" - "$@" <<'EOF'
import sys, torch
sys.argv = ['smoke',
    '--root', 'received_data_from_Pinstudio_20260424/unzipped/0423',
    '--arch', 'v2', '--iters', '2', '--b_local', '4', '--k_sup', '12',
    '--workers', '0', '--scale_div', '2',
    '--surf_samples', '300000', '--head_boost', '4',
    '--rasterize_mode', 'antialiased', '--scale_ratio', '8',
    '--proj_feat', '1', '--lpips_start', '0', '--scale_clip_warmup', '0',
    '--vis_every', '100', '--save_every', '0', '--out', '/tmp/dog_lrm_smoke',
] + sys.argv[1:]
import train_dog_lrm_ddp
train_dog_lrm_ddp.main()
print(f'SMOKE OK | peak GPU mem: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB')
EOF
