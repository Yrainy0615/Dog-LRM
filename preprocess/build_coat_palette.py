#!/usr/bin/env python3
"""Build synth_fur/coat_palette.npz: per-style coat colours sampled from masked real dogs.

Style families follow blender_fur_dataset.py STYLES; real dogs are grouped by their
vlm_priors curl_class. Pixels are sampled from eroded masks at cache_s8, then k-means
per style gives k representative sRGB colours.

  python preprocess/build_coat_palette.py
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

CLASS2STYLE = {"short_smooth": "short", "double_coat": "short",
               "long_straight": "long", "wavy": "wavy", "wire": "wavy",
               "curly": "curly"}


def erode(m, it=3):
    for _ in range(it):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m


def kmeans(x, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        d = ((x[:, None] - c[None]) ** 2).sum(-1)
        a = d.argmin(1)
        c = np.stack([x[a == i].mean(0) if (a == i).any() else c[i] for i in range(k)])
    # order by cluster size, largest first
    sizes = np.bincount(a, minlength=k)
    return c[np.argsort(-sizes)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--priors", default="exps/vlm_priors")
    ap.add_argument("--out", default="synth_fur/coat_palette.npz")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--views_per_dog", type=int, default=6)
    args = ap.parse_args()

    style_px = {}
    for jp in sorted(glob.glob(os.path.join(args.priors, "*.json"))):
        meta = json.load(open(jp))
        style = CLASS2STYLE.get(meta.get("curl_class", ""), None)
        if style is None:
            continue
        dog = meta["dog"]
        cache = os.path.join(args.root, dog, "colmap", "preprocess", "cache_s8")
        jpgs = sorted(glob.glob(os.path.join(cache, "*.jpg")))
        if not jpgs:
            continue
        px = []
        for fp in jpgs[:: max(1, len(jpgs) // args.views_per_dog)][: args.views_per_dog]:
            mp = fp[:-4] + ".png"
            if not os.path.exists(mp):
                continue
            img = np.asarray(Image.open(fp).convert("RGB"), np.float32) / 255.0
            m = np.asarray(Image.open(mp).convert("L"), np.float32) > 127
            m = erode(m)
            if m.sum() < 100:
                continue
            px.append(img[m])
        if px:
            style_px.setdefault(style, []).append(np.concatenate(px))
            print(f"{dog}: {meta['curl_class']} -> {style}, {sum(len(p) for p in px)} px", flush=True)

    rng = np.random.default_rng(0)
    out = {}
    for style, chunks in style_px.items():
        # equal pixel budget per dog so one big dog doesn't dominate the palette
        per = 20000
        x = np.concatenate([c[rng.choice(len(c), min(per, len(c)), replace=False)] for c in chunks])
        out[style] = kmeans(x, args.k).astype(np.float32)
        print(f"[palette] {style}: {len(chunks)} dogs, {len(x)} px -> {args.k} colours\n{np.round(out[style], 3)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
