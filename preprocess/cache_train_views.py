#!/usr/bin/env python3
"""Pre-downscale each view's rgb + mask to 1/scale_div and cache to disk, so training
reads tiny files (full-res 16MP PNG masks otherwise dominate per-step IO). CPU-parallel
over (scene, frame). Output: <scene>/preprocess/cache_s<S>/<stem>.{jpg,png}.

  python preprocess/cache_train_views.py --root <root> --scale_div 8 --procs 32
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

from PIL import Image

SCALE = 8


def one(task):
    scene, fr, out = task
    W, H = fr["width"] // SCALE, fr["height"] // SCALE
    stem = os.path.splitext(fr["name"])[0]
    rp = os.path.join(out, stem + ".jpg")
    mp = os.path.join(out, stem + ".png")
    if os.path.exists(rp) and os.path.exists(mp):
        return 0
    img = Image.open(os.path.join(scene, fr["image_path"]))
    img.draft("RGB", (W, H))
    img.convert("RGB").resize((W, H)).save(rp, quality=95)
    m = Image.open(os.path.join(scene, "preprocess", "masks", stem + ".png")).convert("L")
    m.resize((W, H)).save(mp)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scale_div", type=int, default=8)
    ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()
    global SCALE
    SCALE = args.scale_div

    tasks = []
    for scene in sorted(glob.glob(os.path.join(args.root, "*", "colmap"))):
        cj = os.path.join(scene, "preprocess", "cameras.json")
        if not os.path.exists(cj):
            continue
        out = os.path.join(scene, "preprocess", f"cache_s{SCALE}")
        os.makedirs(out, exist_ok=True)
        for fr in json.load(open(cj))["frames"]:
            tasks.append((scene, fr, out))
    print(f"{len(tasks)} views to cache (scale 1/{SCALE}) on {args.procs} procs", flush=True)
    with Pool(args.procs) as p:
        done = sum(p.map(one, tasks, chunksize=16))
    print(f"cached {done} new views ({len(tasks)} total)", flush=True)


if __name__ == "__main__":
    main()
