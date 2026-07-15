# Dog-LRM

Single pet image → 3D Gaussian avatar + SMAL body params + fur.

## Status

Two branches, sharing a common DINOv2-conditioned MM-DiT backbone:

- **w/o fur (Branch A)** — feed-forward single-image → posed 3D Gaussians on the SMAL surface.
  Current best: `dog_lrm/model_v2.py`, checkpoint `exps/dog_lrm_v2_pa300k`.
- **fur (Branch B)** — DiffLocks-style strand diffusion conditioned on anchor image features
  (`train_fur_diff.py`), decoded to per-strand geometry and rendered as gaussian tubes
  (`gen_diff_fur.py`). Colour comes from projecting real photos onto the strands, not from
  the diffusion model. Not yet fused into a single feed-forward pass with Branch A.

## Setup

See [SETUP.md](SETUP.md) for the environment recipe and weight/data locations.

## Design & experiment log

The project knowledge base lives in `.claude/skills/`:

- [`surveying-dog-avatar-papers`](.claude/skills/surveying-dog-avatar-papers/) — related-work notes
- [`iterating-dog-lrm-design`](.claude/skills/iterating-dog-lrm-design/) — architecture decisions,
  causal chains, and dead ends (read before proposing a design change)
- [`training-dog-lrm-experiments`](.claude/skills/training-dog-lrm-experiments/) — training/eval
  operating procedure, current best config, checkpoint ledger, pitfalls

See also [ANIMAL_LHM_PLAN.md](ANIMAL_LHM_PLAN.md) for the original human→animal adaptation plan.

## Training (Branch A, w/o fur)

```bash
torchrun --nproc_per_node=8 train_dog_lrm_ddp.py \
    --root <per-dog COLMAP scenes> \
    --arch v2 --surf_samples 300000 --head_boost 4 \
    --rasterize_mode antialiased --scale_ratio 8 \
    --proj_feat 1 --ref_res 896 --scale_div 2 \
    --k_sup 12 --workers 0 \
    --lr 2e-4 --warmup_iters 500 --iters 20000 \
    --out exps/<name>
```

Full flag reference, current-best checkpoint, and fine-tune recipe:
`.claude/skills/training-dog-lrm-experiments/recipe.md`.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
