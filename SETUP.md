# Dog-LRM Setup (environment + weights)

Animal (dog) adaptation of LHM: pet image → 3D Gaussian avatar + SMAL params.
See `ANIMAL_LHM_PLAN.md` for the design and `preprocess/README.md` for the data pipeline.

> The base LHM install (model/inference side) is unchanged — follow the original
> `INSTALL.md` / `requirements.txt`. Below is **only what Dog-LRM adds on top.**

## 1. Environment

```bash
conda create -n dog-lrm python=3.10 -y
conda activate dog-lrm

# PyTorch (match your CUDA; example cu118)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Preprocessing deps
pip install "numpy<1.24"          # chumpy uses deprecated np aliases; >=1.24 breaks it
pip install chumpy                # BARC loads the SMBLD .pkl via chumpy
pip install pillow scipy trimesh
pip install transformers          # BiRefNet weights via HuggingFace (Stage 2)

# pytorch3d (silhouette renderer, Stage 4) — install matched to your torch/CUDA:
# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## 2. Clone BARC (dog SMAL model — NOT committed to this repo)

`third_party/` is git-ignored (the SMBLD `.pkl` is license-gated and ~34 MB). Clone it
yourself; the SMBLD dog model ships **inside** the clone, so no extra download is needed:

```bash
git clone --depth 1 https://github.com/runa91/barc_release.git third_party/barc_release
# provides: third_party/barc_release/data/smal_data/my_smpl_SMBLD_nbj_v3.pkl  (+ shape prior)
#           third_party/barc_release/src/smal_pytorch/...                      (SMAL layer)
```
Mind the underlying SMAL/SMBLD research license (MPI / Zuffi et al.).

## 3. Weights to download

| What | For | Source | Notes |
|---|---|---|---|
| **BiRefNet** | Stage 2 masks | HF `zhengpeng7/BiRefNet` (auto) | Downloads on first `extract_masks.py` run. Or pass a local `.pth` via `--weights`. |
| **SMBLD dog SMAL** | Stage 4 fit + body model | **bundled in BARC clone** | No separate download. |
| **DINOv3 ViT-L/16** | P1 image encoder (frozen) | Meta / HF (gated, accept license) | Needed when model code lands; not used by preprocessing. |
| BARC keypoint net | Stage 3 (deferred) | BARC project page | Only if/when we add keypoint refinement. |

Not needed: Sapiens (dropped), ArcFace/face-SR/StyleGAN (human-only, removed in plan).
LHM base weights are optional — only if we warm-start the transformer/decoder later.

## 4. Run preprocessing

See `preprocess/README.md`. One pass over a parent dir of per-pet COLMAP scenes:

```bash
ROOT=/path/to/all_pets
python preprocess/colmap_to_cameras.py --root $ROOT
python preprocess/extract_masks.py     --root $ROOT
python preprocess/fit_smal.py          --root $ROOT --debug
python preprocess/build_manifest.py    --root $ROOT
```
