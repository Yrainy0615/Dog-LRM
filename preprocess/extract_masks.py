#!/usr/bin/env python3
"""Stage 2 preprocessing: foreground masks via BiRefNet.

Reuses BiRefNet (the matting net already vendored under engine/BiRefNet). Weights are
not shipped; by default this pulls the public checkpoint from HuggingFace
(`zhengpeng7/BiRefNet`, same model). Use --weights to load a local .pth via the
in-repo model code instead.

Input : <scene>/images/...
Output: <scene>/preprocess/masks/<name>.png   (soft alpha, 0..255, original resolution)

Deps: torch, torchvision, pillow  (+ transformers for the default HF route).
"""
import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# BiRefNet's exact preprocessing (engine/BiRefNet/inference_img.py)
_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(weights, device):
    if weights is None:
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(
            "zhengpeng7/BiRefNet", trust_remote_code=True)
    else:
        # in-repo model code path
        birefnet_dir = os.path.join(os.path.dirname(__file__), "..", "engine", "BiRefNet")
        sys.path.insert(0, os.path.abspath(birefnet_dir))
        from models.birefnet import BiRefNet
        from utils import check_state_dict
        model = BiRefNet(bb_pretrained=False)
        sd = check_state_dict(torch.load(weights, map_location="cpu"))
        model.load_state_dict(sd)
    model.eval().to(device)
    return model


@torch.no_grad()
def predict_mask(model, image, device):
    x = _transform(image.convert("RGB")).unsqueeze(0).to(device)
    pred = model(x)[-1].sigmoid().cpu()[0].squeeze()  # [1024,1024] in 0..1
    mask = transforms.ToPILImage()(pred).resize(image.size)  # back to original
    return mask


def process_scene(scene_dir, out_subdir, model, device):
    img_dir = os.path.join(scene_dir, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"{img_dir} not found")
    out_dir = os.path.join(scene_dir, out_subdir, "masks")
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(IMG_EXTS))
    for n in names:
        image = Image.open(os.path.join(img_dir, n))
        mask = predict_mask(model, image, device)
        stem = os.path.splitext(n)[0]
        mask.save(os.path.join(out_dir, stem + ".png"))
    print(f"[ok] {scene_dir}: {len(names)} masks")
    return len(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene_dir", help="single COLMAP scene dir")
    g.add_argument("--root", help="parent dir; each immediate subdir is one scene")
    ap.add_argument("--out_subdir", default="preprocess")
    ap.add_argument("--weights", default=None,
                    help="local BiRefNet .pth (default: download from HuggingFace)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = load_model(args.weights, args.device)
    print("BiRefNet ready.")

    scenes = ([args.scene_dir] if args.scene_dir else
              [os.path.join(args.root, d) for d in sorted(os.listdir(args.root))
               if os.path.isdir(os.path.join(args.root, d))])

    total = 0
    for scene in scenes:
        try:
            total += process_scene(scene, args.out_subdir, model, args.device)
        except Exception as e:
            print(f"[skip] {scene}: {e}")
    print(f"[done] {len(scenes)} scene(s), {total} masks total")


if __name__ == "__main__":
    main()
