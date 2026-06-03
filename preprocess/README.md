# Preprocessing (P0): COLMAP pet data → training schema

One pet == one COLMAP scene: `<scene>/images/` + `<scene>/sparse/0/`.
All outputs land in `<scene>/preprocess/`.

Per-frame target schema (assembled by the final stage):
`{ image_path, mask_path, intrinsics(fx,fy,cx,cy), c2w(4x4), keypoints_2d, smal_params }`

## Stages

| # | Script | Deps | Output | Status |
|---|---|---|---|---|
| 1 | `colmap_to_cameras.py` | numpy | `cameras.json`, `scene_norm.json` | ✅ ready |
| 2 | `extract_masks.py` | torch, BiRefNet (HF weights) | `masks/<name>.png` (soft alpha) | ✅ ready |
| 3 | animal 2D keypoints | BARC 24-kp detector | `keypoints.json` | ⏸ deferred (silhouette-only first; add later to refine) |
| 4 | `fit_smal.py` | torch, pytorch3d, chumpy, BARC SMAL | `smal_params.json` (shared pseudo-GT) | ✅ ready (untested) |
| 5 | `build_manifest.py` | stdlib | `manifest.json` | ✅ ready |

Each pet's COLMAP is a **static** multi-view capture → one **shared** SMAL per scene
(single pose+shape), fit by silhouette matching across all views. Keypoints (Stage 3)
are deferred; the silhouette+multi-view fit runs without a keypoint detector.

## Run Stage 1

```bash
python preprocess/colmap_to_cameras.py --scene_dir /path/to/pet_001   # single
python preprocess/colmap_to_cameras.py --root /path/to/all_pets       # batch
```
Verify: `cameras.json.num_frames` == #images; each `c2w` is 4×4; `scale` ≠ 0.
A `[WARN: distortion]` line means COLMAP used a distortion model → undistort before fitting.

Output c2w convention is **OpenCV/COLMAP** (+X right, +Y down, +Z forward). Matching it
to the renderer's camera convention is a deliberate downstream step (plan §5 risk).

## Run Stage 2 (masks)

```bash
python preprocess/extract_masks.py --root /path/to/all_pets            # HF weights (auto)
python preprocess/extract_masks.py --root /path/to/all_pets --weights /path/to/BiRefNet.pth
```
Verify: `<scene>/preprocess/masks/` has one PNG per image; foreground ≈ white.
Needs `transformers` for the default HF route (first run downloads weights).

## Run Stage 4 (SMAL fit) + Stage 5 (manifest)

```bash
python preprocess/fit_smal.py --root /path/to/all_pets --debug   # writes smal_params.json
python preprocess/build_manifest.py --root /path/to/all_pets     # writes manifest.json
```
Stage 4 needs `pytorch3d` + `chumpy` (BARC loads the .pkl via chumpy; on numpy>=1.24
you may need `pip install "numpy<1.24"` or a chumpy patch). With `--debug` it saves
`<scene>/preprocess/smal_debug/<name>.png` — **the key verification**: the red SMAL
silhouette should overlap the dog. Tune `--iters_full`, `--w_pose`, `--render_res`
if alignment is poor. `final_loss` is printed per scene.

## End-to-end (one server pass)

```bash
ROOT=/path/to/all_pets
python preprocess/colmap_to_cameras.py --root $ROOT
python preprocess/extract_masks.py      --root $ROOT
python preprocess/fit_smal.py           --root $ROOT --debug
python preprocess/build_manifest.py     --root $ROOT
```

## Body model (in hand)

The SMBLD dog model ships with the BARC clone — no license download needed:
`third_party/barc_release/data/smal_data/my_smpl_SMBLD_nbj_v3.pkl` (+ shape prior).
SMAL PyTorch layer: `third_party/barc_release/src/smal_pytorch/smal_model/smal_torch_new.py`.
