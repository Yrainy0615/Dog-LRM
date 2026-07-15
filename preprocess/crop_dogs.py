#!/usr/bin/env python3
"""Crop centered square dog images for BARC inference (its ImgCrops demo input).

BARC expects crops that show the dog roughly centered. We use the Stage-2 mask bbox,
expand to a square with margin, and crop. BARC's loader pads/resizes to 256 itself.

Input : <scene>/preprocess/cameras.json + <scene>/preprocess/masks/<name>.png
Output: <scene>/preprocess/barc_crops/<name>.jpg  (+ crop_boxes.json for back-mapping)

Deps: numpy, pillow.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def crop_scene(scene_dir, out_subdir, margin, stride):
    pre = os.path.join(scene_dir, out_subdir)
    frames = json.load(open(os.path.join(pre, "cameras.json")))["frames"][::stride]
    out_dir = os.path.join(pre, "barc_crops")
    os.makedirs(out_dir, exist_ok=True)

    boxes = {}
    n = 0
    for fr in frames:
        stem = os.path.splitext(fr["name"])[0]
        mpath = os.path.join(pre, "masks", stem + ".png")
        if not os.path.exists(mpath):
            continue
        img = Image.open(os.path.join(scene_dir, fr["image_path"])).convert("RGB")
        m = np.array(Image.open(mpath).convert("L"))
        ys, xs = np.where(m > 127)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        side = max(x1 - x0, y1 - y0) * margin
        box = (int(cx - side / 2), int(cy - side / 2),
               int(cx + side / 2), int(cy + side / 2))
        img.crop(box).save(os.path.join(out_dir, stem + ".jpg"), quality=95)
        boxes[stem + ".jpg"] = {"box": box, "name": fr["name"]}
        n += 1

    json.dump(boxes, open(os.path.join(out_dir, "crop_boxes.json"), "w"), indent=2)
    print(f"[ok] {scene_dir}: {n} crops")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene_dir")
    g.add_argument("--root")
    ap.add_argument("--out_subdir", default="preprocess")
    ap.add_argument("--margin", type=float, default=1.3, help="bbox->square expansion")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth view")
    args = ap.parse_args()

    scenes = ([args.scene_dir] if args.scene_dir else
              [os.path.join(args.root, d) for d in sorted(os.listdir(args.root))
               if os.path.isdir(os.path.join(args.root, d))])
    total = 0
    for scene in scenes:
        try:
            total += crop_scene(scene, args.out_subdir, args.margin, args.stride)
        except Exception as e:
            print(f"[skip] {scene}: {e}")
    print(f"[done] {len(scenes)} scene(s), {total} crops")


if __name__ == "__main__":
    main()
