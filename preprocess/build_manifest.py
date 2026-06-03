#!/usr/bin/env python3
"""Stage 5 preprocessing: assemble a per-scene manifest the training dataset consumes.

Merges Stage 1 (cameras) + Stage 2 (masks) + Stage 4 (shared SMAL) into one file.
SMAL is scene-shared (static multi-view capture), so it lives at the top level, not
per frame.

Output: <scene>/preprocess/manifest.json
  {
    "smal": {...},                      # shared SMAL params (or null if not fitted)
    "scene_norm": {...},
    "frames": [ {image_path, mask_path, fx,fy,cx,cy, width,height, c2w}, ... ]
  }

Deps: stdlib only.
"""
import argparse
import json
import os


def build_scene(scene_dir, out_subdir):
    pre = os.path.join(scene_dir, out_subdir)
    cameras = json.load(open(os.path.join(pre, "cameras.json")))
    scene_norm = _maybe_load(os.path.join(pre, "scene_norm.json"))
    smal = _maybe_load(os.path.join(pre, "smal_params.json"))

    frames = []
    for fr in cameras["frames"]:
        stem = os.path.splitext(fr["name"])[0]
        mask_rel = os.path.join(out_subdir, "masks", stem + ".png")
        has_mask = os.path.exists(os.path.join(scene_dir, mask_rel))
        frames.append({
            "image_path": fr["image_path"],
            "mask_path": mask_rel if has_mask else None,
            "width": fr["width"], "height": fr["height"],
            "fx": fr["fx"], "fy": fr["fy"], "cx": fr["cx"], "cy": fr["cy"],
            "has_distortion": fr["has_distortion"],
            "c2w": fr["c2w"],
        })

    manifest = {
        "convention": cameras.get("convention", "opencv"),
        "smal": smal,
        "scene_norm": scene_norm,
        "num_frames": len(frames),
        "frames": frames,
    }
    json.dump(manifest, open(os.path.join(pre, "manifest.json"), "w"), indent=2)

    n_mask = sum(f["mask_path"] is not None for f in frames)
    flags = ("smal" if smal else "NO-smal") + f", {n_mask}/{len(frames)} masks"
    print(f"[ok] {scene_dir}: {len(frames)} frames ({flags})")
    return len(frames)


def _maybe_load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene_dir")
    g.add_argument("--root")
    ap.add_argument("--out_subdir", default="preprocess")
    args = ap.parse_args()

    scenes = ([args.scene_dir] if args.scene_dir else
              [os.path.join(args.root, d) for d in sorted(os.listdir(args.root))
               if os.path.isdir(os.path.join(args.root, d))])
    total = 0
    for scene in scenes:
        try:
            total += build_scene(scene, args.out_subdir)
        except Exception as e:
            print(f"[skip] {scene}: {e}")
    print(f"[done] {len(scenes)} scene(s), {total} frames")


if __name__ == "__main__":
    main()
