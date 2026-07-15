# Dog-LRM Fur — Three-Approach Comparison

Three approach families were explored for giving a Dog-LRM 3DGS dog avatar **fur** that is both
**decomposable** (a separable skin/undercoat layer vs. a simulatable fur layer) and visually faithful.

---

## Table 1 — Overview

| Approach | How it works | Verdict |
|---|---|---|
| **A. Feed-forward (joint)**<br>(v6 → v9) | A single network, a single forward pass. A shared DINOv2 backbone feeds **two heads predicted together**: a *body head* (D-SMAL-surface gaussians) and a *fur head* (per-root strands: length, TBN direction, gravity droop, curl, opacity gate). v9 adds a pixel-aligned "splatter" residual + adversarial loss for sharpness. | **Best texture and the only one that generalizes** (single image → fur, no per-scene optimization). **But the coat is baked into the body → NOT decomposable** (you cannot cleanly pull out a movable fur layer); the face needs an explicit `nofur` mask. |
| **B. Two-stage (frozen)** | Train the body first (Stage-1, on photos that already contain fur), **freeze it**, then train a separate fur layer on top (Stage-2). | **Dead end.** Because Stage-1 was trained on furry photos, the "body" has **already baked in the coat appearance**. With the body frozen, the Stage-2 fur is **redundant — it has nowhere to live**; adding it is either purely cosmetic or it actively **raises the static reconstruction error**. This is the root cause of the long "fur is completely wrong / adding fur hurts L1" struggle. |
| **C. Cascade (coupled + recession)**<br>(v11, per-scene) | Body and fur are **jointly optimized and coupled**: wherever fur is committed, the body **recedes into a darker undercoat** (forced by a `w_recede` prior), and the **semi-transparent strands carry the coat colour** so the undercoat shows through the partings. Geometry comes from a combed v6-flow strand prior; the face is excluded via `w_face`. | **Decomposable + simulatable** (sway reveals the undercoat through partings) with composite ≈ ground truth. **But it is currently per-scene optimization** (multi-view, ~1–2k iters) — a *ceiling probe*, not a deployable feed-forward model — and the texture is softer than v9. |

> **New this round — NeuralFur (the real GaussianHaircut optimizer)** belongs to the cascade family
> (per-scene strand optimization). We took **only its strand geometry** and recoloured it with our
> cascade's neighbour-smoothed albedo-query (NeuralFur's native per-strand colour is independent →
> speckly/over-bright). The fusion — NeuralFur geometry + our colour — works cleanly.

---

## Table 2 — Axis-by-axis comparison

| Axis | A) Feed-forward (v6–v9) | B) Two-stage (frozen) | C) Cascade (per-scene) |
|---|---|---|---|
| **Skin ↔ fur relation** | Parallel prediction heads, no coupling between them | Hard-decoupled: body frozen, fur bolted on afterward | **Coupled**: body actively recedes where fur is committed |
| **Where appearance lives** | In the **body** (coat baked into body gaussians) | In the **body** (coat baked in before freezing) | **Split**: dark undercoat (body) + coat colour (fur) |
| **Decomposable?** | ✗ — coat and body are entangled | ✗ — coat already baked before fur is added | **✓ — clean undercoat ↔ fur separation** |
| **Role of the fur** | Decorative (body already looks right) | Redundant (nothing left for it to explain) | **Load-bearing — the fur carries the coat appearance** |
| **Simulatable fur** | Weak (no real movable layer) | Weak (fur is cosmetic) | **✓ — sway works; partings reveal the undercoat** |
| **Texture quality** | **Best (v9: sharp, realistic)** | n/a (never produced usable fur) | Softer; still needs sharpening toward v9 |
| **Single-image generalization** | **✓ — feed-forward, one image → fur** | Partial | ✗ — per-scene multi-view optimization only |
| **Static composite L1** | ≈ body-shell (low) | Adding fur **raises** it | 0.010–0.015 (curly coats ~0.03) |
| **Face handling** | `nofur` mask (face → Stage-1 skin) | — | `w_face` excluded (face stays bare skin) |

> **A myth that has been disproven:** structured, simulatable fur **cannot beat** a smooth body-shell on
> static L1 — so **`L1 < 0.01` is really a body-shell metric**. The value of fur is **decomposability +
> dynamics + perceptual sharpness**, *not* static L1. Gate the body-shell on L1; judge the fur on realism,
> separability, and motion.
