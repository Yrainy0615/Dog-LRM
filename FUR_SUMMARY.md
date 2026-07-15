# Dog-LRM Stage-2 Fur — Findings & Conclusions (handoff for other agents)

Last updated 2026-06-24. Scope: how to give a Dog-LRM 3DGS dog avatar **fur** that is (ideally)
**decomposable** (separable skin/undercoat vs a **simulatable** fur layer) AND visually good.
Three approach families were explored. This doc compares them and records what is settled vs open.

> Companion memory notes (auto-loaded): `fur-dense-route`, `fur-difflocks-route`, `smal-part-labels`,
> `nfs-stale-read`. Read those for the gotchas. This doc is the architecture-level synthesis.

---

## 0. TL;DR

- **Three families**: (A) **feedforward joint** body+fur (v6→v10), (B) **two-stage** (Stage-1 frozen → Stage-2 fur), (C) **cascade** (coupled skin↔fur with recession, per-scene).
- **A (v6→v9)**: generalizes (single image→fur), **best fur TEXTURE** (v9 = pixel-aligned splatter + adversarial). But **NOT decomposable** (coat baked into the body) and had a **face-fur** problem (fixed by a `nofur` mask).
- **B (two-stage)**: froze Stage-1 (coat baked) then added fur → fur is **redundant / nowhere to live** → struggled ("fur completely wrong"; adding fur *raises* static L1). Dead end as-is.
- **C (cascade)**: body **recedes to a dark undercoat** where fur is committed, fur **carries the coat** → **decomposable + simulatable**, composite≈GT. Currently **per-scene optimization** (ceiling probe), not feedforward. Static L1 ~0.010–0.015 (curly worse).
- **PROVEN tradeoff**: structured/simulatable fur **cannot** beat the smooth body-shell on static L1; `L1<0.01` is a **body-shell** metric. Fur's value = decomposability + dynamics + perceptual sharpness, NOT static L1.
- **Current best visual texture = v9 (feedforward)**. **Current best decomposition = cascade**. They have not yet been combined; that combination is the recommended direction (see §6).

---

## 1. The three approaches

### A) Feedforward joint (v6 → v10)  — `train_dog_lrm_fur_v2.py`, `dog_lrm/model_fur.py`
One network, one forward, **shared DINOv2 backbone → body head + fur head** (predicted together).
Body = D-SMAL-surface gaussians; fur = per-root strands (root_face/bary on the D-SMAL mesh, length L,
direction in TBN, droop toward gravity, helical curl, per-root opacity gate, K segments).
- **v6** (`exps/dog_lrm_fur_v6`): Blender synthetic groom/drape prior + **coat "fur embedding"** (`curl_emb`/`curl_mlp` indexed by VLM `curl_id` → curl_amp/freq/length per coat class). `advweak` variants add weak PatchGAN.
- **v7** (`exps/dog_lrm_fur_v7`, 1.1GB): **triplane** feature field + **level-2 body** (62k verts) + bigger backbone (dim 768, 12 layers/heads). Capacity upgrade for sharpness.
- **v8** (`exps/dog_lrm_fur_v8`): **free thin-surfel body** gaussians, **per-strand image-conditioned curl** head (fur_head 21→23 dims), and **`nofur` face/paw mask** (`op *= (1-nofur)`; usable at inference as `anc["nofur"]` override, NO retrain).
- **v9** (`exps/dog_lrm_fur_v9`, 1.1GB): + **single-image pixel-aligned "Splatter" residual branch** (`splat_dec`, GS-LRM-style) + adversarial/higher-LPIPS → **the best fur texture so far** (sharp, realistic). Build args: dim=768, n_layers=12, n_heads=12, K=11, n_root=26000, tri_res=64, tri_ch=32, splat_res=128, splat_base_sc=0.004, splat_dres=0.05.
- **v10/v11** (`exps/dog_lrm_fur_v10*`, `v11_proj`): bigger / floored ref-projection variants.
- **Render a v9 ckpt + face-nofur**: `render_v9_nofur.py` (validated — loads DogLRMFurV9, sets `anc["nofur"]`=face mask, face → Stage-1 skin).
- **Pros**: generalizes; **best texture**. **Cons**: **coat baked into body → NOT decomposable** (can't cleanly pull a movable fur layer); face fur needs the nofur mask.

### B) Two-stage (Stage-1 frozen → Stage-2 fur)
Train the body (Stage-1, `train_dog_lrm_ddp.py`, `/tmp/stage1_final.pt`, subdiv-2 ~62k anchors), **freeze**,
then train a Stage-2 fur layer on top (`train_fur_stage2.py`, `train_fur_undercoat.py`, etc.).
- **Cons / why it stalled**: Stage-1 is trained on photos that **include fur** → the "body" already
  **bakes the coat appearance**. With Stage-1 frozen, Stage-2 fur has **nowhere to live** — it's
  redundant on an already-correct body → either cosmetic or it **degrades** the static fit.
  This is the root of the long "毛完全不对 / adding fur raises L1" struggle.
- **Key realization**: decoupling **without letting the skin recede** makes fur redundant. → led to (C).

### C) Cascade (coupled skin↔fur, recession)  — `train_fur_final.py` (per-scene)
Body and fur are **coupled and jointly optimized**:
- **recession**: in fur regions the body **darkens to a dark undercoat** (learnable per-anchor brightness, pushed by a `w_recede` prior on fur-covered anchors).
- **fur carries the coat**: strands inherit/optimize the coat colour; semi-transparent so the undercoat shows through partings (physically correct; the basis for dynamics).
- geometry = **v6-flow strand prior** (combed tangent flow `d0 = tmix·t + (1-tmix)·n`, droop, curl, offset-shell), face excluded via `w_face`.
- **Result**: **decomposable** (undercoat ↔ fur), **simulatable** (strands are real chains; sway shows undercoat through partings), composite≈GT. **Per-scene** (multi-view) optimization — it's the **ceiling probe**, not a deployable model yet.
- **Coupling needs an explicit prior**: a prototype showed the body does **NOT** spontaneously recede (fur & skin same colour → composite fine either way → underdetermined). The `w_recede` prior is what forces clean separation = a soft/learnable version of the hand-coded undercoat-split.

---

## 2. Comparison

| axis | A) feedforward v6–v9 | B) two-stage frozen | C) cascade (per-scene) |
|---|---|---|---|
| skin↔fur relation | parallel heads, no coupling | frozen, hard-decoupled | **coupled + recession** |
| where appearance lives | **body** (coat baked) | **body** (coat baked) | **split: undercoat + fur** |
| decomposable? | ✗ | ✗ | **✓** |
| fur role | decorative | redundant | **load-bearing (carries coat)** |
| simulatable fur | weak | weak | **✓ (sway works)** |
| texture quality | **best (v9)** | n/a | softer; needs sharpening |
| generalizes (1 img→fur)? | **✓ feedforward** | partial | ✗ (per-scene optim) |
| static composite L1 | ~body-shell (low) | fur raises it | 0.010–0.015 (curly ~0.03) |
| face | nofur mask | — | `w_face` excluded (Stage-1 skin) |

---

## 3. Settled facts (do NOT relitigate)

1. **Structured fur can't beat the body-shell on static L1.** Per-scene oracle (`train_fur_oracle.py`, held-out views, geometry frozen, appearance free): **body-only** L1 = 0.005–0.011; **body + structured fur** = 0.008–0.015 (adding fur only *raises* L1). So **`L1<0.01` is a body-shell metric**; gating fur on it forces you to abandon the very strand structure that is the goal. → Gate the **body-shell** on L1<0.01 (achievable); judge the **fur** on decomposability + realism + dynamics.
2. **Per-scene optimization ceiling plateaus by ~1–2k iters** (7k gives no held-out gain; mild overfit after). More iters ≠ lower ceiling; the ceiling is set by the *representation*.
3. **DiffLocks synthetic-strand route was ABANDONED** (user call): Blender synthetic fur looked wrong; the image→strand diffusion's synth→real transfer was poor (`transfer_real.py`). Infra still exists (`preprocess/blender_fur_dataset.py`, `train_strand_diffusion.py`) if revisited, but de-prioritized.
4. **Headless Blender fur tuning does not converge to photoreal** — it's an art-heavy task without interactive feedback (children/clumping/Principled-Hair-BSDF all tried). Don't sink time into blind param sweeps.
5. **SMAL part labels** (`smal-part-labels` memory): `SMALModel` uses **BARC** joints (muzzle=16,32; tail=25–31; ears=33,34; paws=10,14,20,24). `smal.subdivide(weights)` **MISALIGNS** labels vs subdivided verts — compute on the 3889-base mesh then propagate by **canonical nearest-neighbour**. Also: rendering thin/transparent gaussians has **see-through** that *fakes* scrambled labels — verify numerically, not by eye.
6. **NFS stale reads** (`nfs-stale-read` memory): workspace is NFS; a separate process can `torch.load` a **stale** version of a just-written file → spurious artifacts. Render/eval **in the same process** that produced the data, or sanity-check a known field after load.

---

## 4. Cascade refined recipe (the per-scene ceiling) — `train_fur_final.py`

Tuned via user feedback into the current best per-scene decomposable fur:
- **hug skin (not 炸毛)**: `--tmix 0.9` (combed along surface) `--off 0.05` (low shell lift) `--len_scale 0.6` (shorter). Also lowers L1 (tighter silhouette).
- **soft/fine + dense (not 毛刺, not bald-gray)**: fineness = **many thin strands + low opacity**, NOT fat strands. `--radius_frac 0.0008` (thin) `--fur_op 0.5 --op_keep 0.02` (translucent blend) `--uniform_n ~120000` (uniform area-weighted roots covering everything except face) `--len_floor 0.03` (low-density areas still grow fur → no gray bald spots). NOTE: density-gated `fur_anchors` (40k roots, L≈0 in sparse areas) leaves bald spots — uniform resample fixes it.
- **colour = multi-view OPTIMIZED** (better than one-shot sampling): `--gt_color 1` samples a per-strand colour from GT photos (project root→sample→avg visible) as **init**, then a per-strand learnable residual (±0.3) is **optimized against multi-view**, with `--w_col` L2 toward the sampled init to suppress rainbow. (White coats still pick up some iridescent speckle from sub-pixel thin strands — open issue.)
- **mild geometry opt**: unfroze per-root `d_root` (bounded ±0.04·diag) / `d_dir` / per-strand `d_logL`, kept small by `--w_geo 0.3` (avoids collapse/strays).
- **silhouette** `--w_sil` (penalize fur alpha beyond GT mask), **curly** coat → Kp 10 + tighter CURL + ×0.7 length + `--op_floor 0.15`, **face** = `w_face>0.3` excluded (Stage-1 skin only).
- Result: itsuki(long) 0.011, nara(short) ~0.0095, milk(curly) ~0.015, paul(curly) ~0.03 (weak). Soft, dense, real colour, decomposable, simulatable.

---

## 5. The recurring user intent (target spec)

"**可分解的毛 + 视觉质量(L1<0.01 级)**" → unified vision stated 2026-06-23:
**Stage-1 predicts the no-fur skin GS** → **Stage-2 converts non-face skin GS into v9-logic fur** → **face stays Stage-1 skin (no fur)**. I.e. combine the **cascade recession** (so Stage-1 is true bare skin / undercoat) with the **v9 fur logic** (for texture), face excluded.
Also feedback on look: fur should **hug the skin (not 炸毛)**, be **fine (many thin hairs)**, **densely cover everything except the face**, and **colour optimized multi-view** (not just sampled).

---

## 6. Open problems & recommended next steps

1. **Combine v9 texture + cascade decomposition** (the target). v9 = best texture but coat-baked; cascade = decomposable but soft. Path: take the v9 fur head/representation (+ pixel-aligned/adversarial sharpness) as the **Stage-2 fur layer**, root it on the Stage-1 skin, add the **recession** so the body becomes bare undercoat, exclude face (`nofur`). This gives v9 texture + decomposable + bald face.
2. **Feedforward the cascade** (deploy). Per-scene cascade is the ceiling; train an image-conditioned head (on the v6/v7 backbone) to predict {fur op/colour/geometry-residual + body recession} from a single image, supervised by the per-scene cascade results + multi-view photometric. = generalization of (A) + decomposability of (C).
3. **White-coat rainbow**: thin strands on white → iridescent speckle. Try light-coat-adaptive radius (fatter for light coats), stronger colour uniformity, or higher-res anti-aliased export.
4. **Curly coats (e.g., paul/poodle)**: the strand prior lacks a tight-curl/clump mode → over-grown spiky shag, highest L1 (~0.03). Needs a real curl/clump mode + opacity floor.
5. **Sharpen the cascade fur** toward v9: add the pixel-aligned residual / PatchGAN (v9's levers) to the cascade objective.
6. **Canonical export**: `export_canonical.py` un-poses Stage-1 to canonical via per-anchor `T⁻¹` (T = BARC LBS transform, captured via `smal.smal.last_T` — a 1-line patch in `third_party/.../smal_torch_new.py`). Works (coherent canonical rest pose) but **orientations are rough** (quats rotated by T⁻¹ + subdivide(T) approximation). Refine by re-deriving quats from the canonical surface. `--pose posed` is the clean correct product; subdiv-3 ≈ 248k gaussians.

---

## 7. Key files / checkpoints

- **Stage-1 body**: `train_dog_lrm_ddp.py`; ckpt `/tmp/stage1_final.pt` (subdiv-2, K=1). DogLRM in `dog_lrm/model.py`.
- **Feedforward fur (v6–v10)**: `train_dog_lrm_fur_v2.py`, `dog_lrm/model_fur.py` (DogLRMFurV2/V7/V8/V9/V10). ckpts `exps/dog_lrm_fur_v6|v7|v8|v9|v10/model.pt`. **v9 = best texture.** Render+nofur: `render_v9_nofur.py`. Eval: `eval_fur_ff.py`. Export: `export_sample_format.py` (strands_lines.ply / body_rig.glb / gaussians ±fur).
- **Cascade (best decomposable)**: `train_fur_final.py` (+ `train_fur_v6flow.py` FurV6Flow). Outputs decomp / ply / sway / fur.pt. Results under `exps/overnight/{cascade,cover,soft2,fine,fine2}`.
- **Oracle / ceiling**: `train_fur_oracle.py`. **Per-scene fur-led** baseline: `train_fur_furled.py`.
- **DiffLocks (abandoned)**: `preprocess/blender_fur_dataset.py`, `train_strand_diffusion.py`, `sample_strand_diffusion.py`, `transfer_real.py`.
- **Region/labels**: `diag_fur_region2.py` (correct base+canonical-NN part labels). `fur_anchors.npz` per scene (roots/TBN/L/w_face/w_ear/curl_id).
- **Canonical export**: `export_canonical.py` (+ `last_T` patch in BARC smal).
- **Overnight report**: `exps/overnight/REPORT.md`.

## 8. One-paragraph recommendation
The cascade (C) is the right *structure* (decomposable, simulatable) and the recession prior is the
key mechanism; v9 (A) is the right *texture*. Neither alone meets "decomposable + v9-quality". The
highest-value next step is **§6.1 + §6.2**: put the v9 fur logic (with its pixel-aligned/adversarial
sharpness) into the cascade's Stage-2 fur layer over a receding Stage-1 skin, then train it
feedforward. Judge the body-shell on L1<0.01 (achievable) and the fur on realism/decomposability/
dynamics — do NOT gate the fur on static L1 (proven impossible for structured fur).

---

## 9. v11 progress (2026-06-24 overnight) — see FUR_V11_PLAN.md
Direction decided with user: **NeuralFur** (3DV'26, `NeuralFur/`) = per-scene TEACHER; **Splatter
Image** (already in v9 `splat_dec`) = feed-forward STUDENT; **LHM** hybrid = architecture;
fusion = **distillation**. "cascaded vs feed-forward" = teacher vs student; "two-stage" = an offline
data pipeline, NOT an in-model freeze.

**Shipped tonight in `train_fur_final.py` (per-scene cascade, the teacher):**
1. **Dark-undercoat-hole bug FIXED** (`--cov_gate 1`, default on). Old recession darkened the body by a
   binary region label (`cover`) independent of actual fur → black showed through thin fur. Now
   `bmult = 1 - fcov·(1-uc_floor)` where `fcov` = opacity-weighted local fur coverage (kNN of roots,
   `root_opacity()` added to FurV6Flow). Deterministic (drops `body_recede` param + `w_recede`
   prior). Verified: undercoat no longer black, **L1 preserved** (nara 0.0077, itsuki 0.0107, milk
   0.0140), **0 dark pixels in sway**.
2. **NeuralFur geometric furless-shrink** (`--shrink 0.03`). Body+roots move inward along normal by
   per-part thickness (face 0, ears 0.4×, body 1×), fur length extended to keep silhouette → fur
   gets its own **shell** over a smooth **furless body** = sim-ready decomposition (verified on itsuki:
   panel1 furless body + panel3 separable fur shell = composite≈GT, L1 0.0104).
3. **Finer / shorter / flatter** per user art-direction (毛细一点 / 太长 / 不要竖起来):
   `radius_frac 0.0005`, `uniform_n 160000`, `len_scale 0.45`, `tmix 0.95`, `off 0.03`, `w_sil 0.9`.
   Swept 12 dogs (`exps/fur_v11_sweep/`), L1 0.007–0.014 (curly paul 0.0215 / bear 0.0184 worst, as
   documented). Lengths now track GT silhouette. **Residual**: some dogs still show edge-strand
   stick-out at the rump silhouette (kuma) → next lever `tmix↑/off↓/len↓`.

**Artifacts**: `exps/fur_v11/` (cov_gate, len0.6), `exps/fur_v11_fsf/` + `exps/fur_v11_sweep/`
(fine/short/flat/shrink). Montages: `beforeafter_montage.png`, `cmp_montage.png`, `grid12.png`.
Env: `dog-lrm` conda + `PATH=.../dog-lrm/bin` + `TORCH_EXTENSIONS_DIR=.torch_ext_lhm` (ninja+gsplat).

**Next (gated on user OK of the look)**: scaffold the feed-forward STUDENT — single image → per-root
strand params on the shrunk D-SMAL + body recession, distilled from these cascade `*_final.pt`.
