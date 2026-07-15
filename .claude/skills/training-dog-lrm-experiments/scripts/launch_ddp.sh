#!/usr/bin/env bash
# 8 卡 DDP 训练启动（当前最优配置）。冒烟通过后再跑！
# 用法: launch_ddp.sh <out_name> [额外 flags，如 --init_ckpt xxx --iters 6000]
set -e
[ -z "$1" ] && { echo "usage: launch_ddp.sh <out_name> [extra flags]"; exit 1; }
OUT="$1"; shift
cd "$(dirname "$0")/../../../.."   # workspace root
ENV=/home/yyang/.conda/envs/dog-lrm
export PATH="$ENV/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_ext_lhm"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup "$ENV/bin/torchrun" --nproc_per_node=8 train_dog_lrm_ddp.py \
    --root received_data_from_Pinstudio_20260424/unzipped/0423 \
    --arch v2 --surf_samples 300000 --head_boost 4 \
    --rasterize_mode antialiased --scale_ratio 8 \
    --proj_feat 1 --ref_res 896 \
    --scale_div 2 --k_sup 12 --workers 0 \
    --lr 1e-4 --warmup_iters 200 --lr_final_ratio 0.05 \
    --scale_clip_warmup 0 --lpips_start 0 \
    --save_every 1000 --out "exps/$OUT" \
    "$@" > "exps/$OUT.log" 2>&1 &
echo "launched -> exps/$OUT.log (pid $!)"
echo "monitor:  tail -f exps/$OUT.log | grep -E '^metric|SKIP|off=0.23'"
