# FUR V11 — NeuralFur-teacher / Splatter-Image-student, feed-forward decomposable fur

Last updated 2026-06-24. Synthesis of four sources, decided with the user 2026-06-24 (overnight).
Supersedes the "combine v9 + cascade" recommendation in `FUR_SUMMARY.md` §6/§8 with concrete
references. Read `FUR_SUMMARY.md` first for the settled facts (esp. §3 — do not relitigate).

## 0. The four sources and their roles

| source | what we take | role |
|---|---|---|
| **NeuralFur** (3DV'26, `Vanessik/NeuralFur`) | shrink-to-**furless mesh** + Gaussian-Haircut strands + VLM per-part length/thickness/gravity | **per-scene TEACHER** (ceiling) |
| **Splatter Image** (CVPR'24) | single-image → per-pixel feed-forward gaussians | **feed-forward STUDENT** backbone (already partly in v9 `splat_dec`) |
| **LHM** (our origin) | template (animatable, coarse geom) + pixel-aligned (appearance detail) hybrid | **architecture** glue |
| **our pet line** (v6→v9, cascade) | D-SMAL template, v6-flow strands, face `nofur`, decomposable PLY export, part labels | **substrate** |

**Fusion = distillation, not either/or.** "cascaded vs feed-forward" → teacher(per-scene) vs
student(single-image); the link is distillation. "two-stage vs one-stage" → an offline data
pipeline (generate GT → train FF net), NOT an in-model freeze (which is what killed approach B).

## 1. The core idea (one sentence)
**Use Splatter-Image's single-image feed-forward machinery to predict NeuralFur-style
surface-rooted strands (roots on a shrunk furless D-SMAL body), distilled from NeuralFur
per-scene reconstructions, with the body geometrically receding (shrink) so fur has room to
live and the undercoat is never exposed where fur is thin.**

## 2. The dark-undercoat-hole bug (user, 2026-06-24) — root cause + fix
**Symptom**: cascade left **dark base colour in regions not covered by fur** → bad visuals
(see `exps/fur_final/00003-nara_decomp.png`, `00104-milk_decomp.png`: undercoat panel ≈ black
over the whole body; thin-fur areas show it through the composite).

**Root cause**: recession darkened the body by a **binary region label** (`cover` = "nearest fur
root is a non-face root"), *independent of whether fur actually opaquely covers that spot*. Sparse
/short/translucent strands → darkened body shows through.

**Two complementary fixes (both = "recede only where fur actually is", the NeuralFur principle):**
- **(B) coverage-gated TONAL recession** — IMPLEMENTED tonight in `train_fur_final.py --cov_gate 1`.
  Per body anchor, `fcov` = opacity-weighted local fur density (kNN of fur roots, weight by
  `exp(-d²/2σ²) × root_opacity`). `bmult = 1 - fcov·(1-uc_floor)`. Thin fur → `fcov→0` → body
  keeps full skin colour → **no dark holes**. Deterministic (drops the free `body_recede` param +
  `w_recede` prior). `root_opacity()` added to `FurV6Flow`.
- **(A) geometric SHRINK recession** (NeuralFur `extract_furless_body.py`) — TO PORT. Move body
  gaussians inward along normal by **per-part thickness** (torso/neck/belly/tail thick; face/paws/
  ears thin; nose/pads 0 — values cribbed from `animal_config.effective_fur_thickness_cm`),
  Laplacian-smooth the field. Body sits inside the silhouette; fur fills the shell → fur has its
  own space (better for decomposition + dynamics), and uncovered = skin colour, not dark.

A and B compose: shrink the body AND darken only where densely covered.

## 3. Teacher (NeuralFur) — reality check
Full stack is HEAVY: GaussianHaircut + NeuralHaircut + NeuS + custom CUDA hair rasterizer +
Directional (C++) + SMALify + Blender, CUDA 11.8, 11-step preprocessing (NeuS recon, Gabor
orientation maps, SDF volumes, VLM annotations). **Standing the full pipeline up on our data is a
multi-day integration, not an overnight task.** Strategy:
- **Tonight / near-term**: port only the *furless-shrink idea* (§2A) into our cascade — cheap, and
  it is the mechanism that fixes the bug. Keep our per-scene cascade as the working teacher.
- **Later**: integrate real NeuralFur per-scene (we have ~90 views + masks + cameras + D-SMAL fit,
  which maps to their SMAL-based preprocessing) to get cleaner strand GT for distillation.

## 4. Student (feed-forward) — `train_dog_lrm_fur_v2.py` line, v9 backbone
- v9 already has the Splatter-Image branch (`splat_dec`, per-pixel gaussians from DINO features).
- **Change**: predict **surface-rooted strand params** (root face/bary on the *shrunk* furless
  D-SMAL, length/dir/curl/opacity) instead of free per-pixel gaussians → **animatable** (the v9
  splat's open problem) + decomposable. Body head predicts the **shrunk** undercoat.
- **Supervision**: multi-view photometric + adversarial (v9) + **distillation** to teacher
  {furless body, strand op/colour/geom} on dogs where we have per-scene results.
- Face excluded via `nofur` (Stage-1 skin, no fur).

## 5. Tonight's plan (verify-as-you-go)
1. **[done]** `--cov_gate` coverage-gated tonal recession in `train_fur_final.py`.
2. **[in progress]** validate on nara (short, was near-black undercoat) + milk (curly white):
   decomp.png undercoat panel should no longer be uniformly black; composite ≥ baseline.
3. Port **(A) geometric per-part shrink** of the body (D-SMAL normals + part labels), compare.
4. Sweep dogs (nara/itsuki/milk/hanabi/paul) → montage, confirm holes gone across coats.
5. If holes fixed: scaffold the **student** strand-root head over the shrunk body (distill from the
   fixed cascade). Else: diagnose & fix, then proceed.

## 5b. CANONICAL per-scene config (locked 2026-06-24, all user art-direction addressed)
`train_fur_final.py --dog <D> --iters 1500 --uniform_n 160000 --radius_frac 0.0005` (finer, dense)
`--len_scale 0.45 --w_sil 0.9` (shorter, hug mask) `--tmix 0.95 --off 0.03 --tip_fade 0.5` (flat, no
stand-up) `--comb_iters 8` (combed flow — answers "strand 方向约束") `--face_keep_thr 0.2 --face_collar
0.7 --head_clear 0.8 --head_r 0.15` (head not occluded) `--cov_gate 1 --shrink 0.03 --w_geo 0.5
--fur_op 0.45 --op_keep 0.02 --len_floor 0.03 --hard_sil 1 --sil_thr 0.5`. **hard_sil** = export-time
visual-hull clip: drop fur gaussians projecting outside the GT mask in <thr of visible views → hard
guarantee fur silhouette ≤ mask (user req "strand 后的 mesh 不要超过 mask"). `visual_hull_keep()` in
train_fur_final.py; itsuki dropped ~1% edge fly-aways, L1 unchanged. Higher sil_thr = stricter. New flags all in `train_fur_final.py`; strand tip
fade in `FurV6Flow`. Robustness: `nn_argmin()` chunks the root↔body cdist (was 35GB one-shot → OOM
under GPU contention). Results: `exps/fur_v11_canon/` (12 dogs, L1 0.0075–0.0135; curly paul 0.0219).
**Direction constraints on strands** (user Q): surface tangent/normal mix `d0=tmix·t+(1-tmix)·n`,
gravity droop, ear-flat, `w_geo` L2 on the `d_dir` residual, + NEW `comb_iters` neighbour-smoothing
of the tangent field (≈ NeuralFur parallel-transport). Stronger options if needed: training-time
neighbour-alignment loss; image orientation maps (Gabor) as in NeuralFur.

## 6. Judge criteria (from FUR_SUMMARY §3 — unchanged)
Body-shell on L1<0.01 (achievable); **fur on decomposability + realism + dynamics + NO dark holes**.
Do NOT gate fur on static L1.
