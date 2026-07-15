# Dog-LRM Setup (environment + weights)

Animal (dog) adaptation of LHM: pet image → 3D Gaussian avatar + SMAL params.
See `ANIMAL_LHM_PLAN.md` for the design and `preprocess/README.md` for the data pipeline.

> The base LHM install (model/inference side) is unchanged — follow the original
> `INSTALL.md` / `requirements.txt`. Below is **only what Dog-LRM adds on top.**

## 1. Environment

Verified recipe (CUDA 12.x driver ≥535, Python 3.10). Order and pins matter — see notes.

```bash
conda create -n dog-lrm python=3.10 -y
conda activate dog-lrm

# 1) PyTorch — cu121 wheels (work with driver 535; use cu118 for older drivers)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 2) Other deps (transformers pinned to a torch-2.1-compatible release)
pip install pillow scipy trimesh "transformers==4.46.3"

# 3) numpy LAST, pinned (chumpy needs deprecated np aliases removed in >=1.24)
pip install "numpy<1.24"

# 4) chumpy with --no-build-isolation (its setup.py imports pip, hidden by PEP-517 isolation)
pip install chumpy --no-build-isolation

# 5) pytorch3d — prebuilt wheel for this exact combo (py310 / cu121 / torch2.1.2); no source build
pip install fvcore iopath
pip install --no-cache-dir --no-deps pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt212/download.html

# 6) BARC inference deps (its model imports these). Pin opencv<4.10 so it keeps numpy 1.x.
pip install "opencv-python-headless<4.10" FrEIA pymp-pypi matplotlib pandas pycocotools importlib_resources
pip install "numpy<1.24"   # re-pin: some of the above try to pull numpy 2.x
```

**BARC gotchas:** its vendored `pilutil` uses removed Pillow APIs → `barc_infer.py` ships
its own crop loader instead of BARC's `ImgCrops`. Its `SilhRenderer` is locked to
pytorch3d 0.2.5/0.6.1 → `barc_infer.py` stubs it (params are computed before rendering).

**Gotchas hit & fixed:**
- `transformers>=5` requires torch≥2.4 and silently disables PyTorch on 2.1.2 → pin `4.46.3`.
- `chumpy` build fails under PEP-517 isolation (`No module named 'pip'`) → `--no-build-isolation`.
- pytorch3d `--no-index` blocks its deps `fvcore`/`iopath` → install them first, then `--no-deps`.

Verified versions (`pip freeze`): torch 2.1.2+cu121, numpy 1.23.5, chumpy 0.70,
pytorch3d 0.7.5, transformers 4.46.3, scipy 1.15.3, trimesh 4.12.2.
Sanity: `python -c "import torch,chumpy,pytorch3d;print(torch.cuda.is_available())"` → `True`,
and BARC `SMAL()` loads (num_betas=54, 7774 faces).

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
