#!/usr/bin/env python3
"""v6 stage-2 step 2: fit our (droop, gamma) strand params to the Blender cloth-draped
strands, then scatter into a per-canonical-vertex GT field (droop_gt, gamma_gt) any dog's
roots can read via root_face/bary -- the supervision target for the model's droop/gamma heads.

  PATH=$ENV/bin:$PATH python preprocess/fit_drape_gt.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("."))


def norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def resample_to_len(S, targetL):
    """Trim each strand [M,K,3] to arc-length targetL[M] and resample to K points by arc
    length. Short-fur regions keep the near-straight top of a long draped strand (low droop);
    long-fur regions keep the full curve (high droop) -> region-varying droop GT."""
    M, K, _ = S.shape
    seg = np.linalg.norm(np.diff(S, axis=1), axis=2)
    arc = np.concatenate([np.zeros((M, 1)), np.cumsum(seg, axis=1)], axis=1)
    out = np.empty((M, K, 3))
    tpos = np.linspace(0, 1, K)[None, :] * targetL[:, None]
    for j in range(K):
        t = tpos[:, j]
        idx = np.clip((arc <= t[:, None]).sum(1) - 1, 0, K - 2)
        a0 = np.take_along_axis(arc, idx[:, None], 1)[:, 0]
        sl = np.take_along_axis(seg, idx[:, None], 1)[:, 0] + 1e-9
        fr = np.clip((t - a0) / sl, 0, 1)[:, None]
        p0 = np.take_along_axis(S, idx[:, None, None].repeat(3, 2), 1)[:, 0]
        p1 = np.take_along_axis(S, (idx + 1)[:, None, None].repeat(3, 2), 1)[:, 0]
        out[:, j] = p0 + fr * (p1 - p0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="synth_fur/drape_raw.npz")
    ap.add_argument("--template", default="synth_fur/blender_input.npz")
    ap.add_argument("--out", default="synth_fur/drape_gt.npz")
    ap.add_argument("--trim", action="store_true",
                    help="trim each (uniform-length) strand to the per-region L_geo at its root, "
                         "so the droop GT varies by region instead of being a degenerate constant")
    args = ap.parse_args()

    from scipy.spatial import cKDTree
    tpl = np.load(args.template)
    verts = tpl["verts"].astype(np.float64)                          # [V,3] canonical
    L_geo_v = tpl["L_geo"].astype(np.float64)
    tree = cKDTree(verts)

    z = np.load(args.raw)
    S = z["drape"].astype(np.float64)                                # [M,K,3] draped
    S0 = z["undraped"].astype(np.float64)
    M, K, _ = S.shape
    if args.trim:                                                    # uniform sim -> per-region length
        _, vi0 = tree.query(S[:, 0])
        targetL = np.clip(L_geo_v[vi0], 1e-3, None)
        S = resample_to_len(S, targetL)
        S0 = resample_to_len(S0, targetL)
        print(f"[trim] per-region length: target p10/50/90 "
              f"{np.round(np.percentile(targetL,[10,50,90]),4)}", flush=True)
    root = S[:, 0]
    d0 = norm(S0[:, 1] - S0[:, 0])                                   # natural (undraped) root dir
    L = np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1)            # draped length (~inextensible)
    g = np.array([0, 0, -1.0])
    sfrac = np.arange(1, K) / (K - 1)

    def forward(droop, gamma):
        beta = droop[:, None] * sfrac[None, :] ** gamma[:, None]
        out = np.empty((M, K, 3)); out[:, 0] = root; cur = root.copy(); step = (L / (K - 1))[:, None]
        for ki in range(1, K):
            b = beta[:, ki - 1:ki]
            cur = cur + norm((1 - b) * d0 + b * g) * step
            out[:, ki] = cur
        return out

    # 2-DOF grid fit: (droop, gamma) per strand
    dg = np.linspace(0, 1, 31); gg = np.linspace(0.5, 4.5, 17)
    best = np.full(M, 1e9); bd = np.full(M, 0.0); bgm = np.full(M, 1.5)
    for dq in dg:
        for gm in gg:
            e = np.linalg.norm(forward(np.full(M, dq), np.full(M, gm)) - S, axis=2).mean(1)
            m = e < best; best[m] = e[m]; bd[m] = dq; bgm[m] = gm
    rel = best / (L + 1e-9)
    print(f"fit {M} strands | residual mean {rel.mean():.3f} p90 {np.percentile(rel,90):.3f} "
          f"<5%:{(rel<.05).mean()*100:.0f}% | droop mean {bd.mean():.2f} gamma mean {bgm.mean():.2f}", flush=True)

    # scatter per-strand fits onto canonical vertices (nearest vert to each strand root)
    verts = np.load(args.template)["verts"].astype(np.float64)       # [V,3]
    V = verts.shape[0]
    from scipy.spatial import cKDTree
    tree = cKDTree(verts)
    _, vi = tree.query(root)                                         # nearest vert per strand
    droop_gt = np.zeros(V); gamma_gt = np.full(V, 1.5); cnt = np.zeros(V)
    np.add.at(droop_gt, vi, bd); np.add.at(cnt, vi, 1.0)
    gsum = np.zeros(V); np.add.at(gsum, vi, bgm)
    seen = cnt > 0
    droop_gt[seen] /= cnt[seen]; gamma_gt[seen] = gsum[seen] / cnt[seen]
    # fill unseen verts from nearest seen vert
    if (~seen).any():
        st = cKDTree(verts[seen]); _, j = st.query(verts[~seen])
        droop_gt[~seen] = droop_gt[seen][j]; gamma_gt[~seen] = gamma_gt[seen][j]
    print(f"per-vertex GT: {int(seen.sum())}/{V} directly hit | "
          f"droop[{droop_gt.min():.2f},{droop_gt.max():.2f}] gamma[{gamma_gt.min():.2f},{gamma_gt.max():.2f}]")

    np.savez(args.out, droop_gt=droop_gt.astype(np.float32), gamma_gt=gamma_gt.astype(np.float32),
             fit_residual=np.float32(rel.mean()))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
