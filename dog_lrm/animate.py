#!/usr/bin/env python3
"""Animate a trained Dog-LRM avatar with an AnimalML3D (OmniMotionGPT) motion.

Pipeline: IK-fit the motion's joint positions -> per-frame SMAL body pose `theta`,
transfer that articulation onto the avatar's identity (keep its betas / global
placement / scale), regenerate Gaussians per frame via the trained model, render
from a fixed camera -> mp4. Appearance offsets are pose-independent (conditioned on
the canonical points + input image), so the same trained model re-poses for free.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.model import DogLRM
from dog_lrm.render import intrinsics, render_gaussians
from dog_lrm.smal_model import SMALModel, load_pseudo_gt
from dog_lrm.motion import load_motion_joints, load_motion_betas, fit_theta_to_joints


def load_cam(scene_dir, fr, s, device):
    K = intrinsics(fr["fx"] / s, fr["fy"] / s, fr["cx"] / s, fr["cy"] / s, device)
    return dict(K=K, c2w=torch.tensor(fr["c2w"], device=device).float(),
                W=fr["width"] // s, H=fr["height"] // s, name=fr["name"],
                path=os.path.join(scene_dir, fr["image_path"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help=".../<pet>/colmap")
    ap.add_argument("--ckpt", required=True, help="model.pt from training")
    ap.add_argument("--motion", required=True, help="animals_smal_joints/<name>.npy")
    ap.add_argument("--template_dir",
                    default="third_party/OmniMotionGPT/data/unzipped/animals_smal_template")
    ap.add_argument("--input_view", type=int, default=0, help="view index for appearance")
    ap.add_argument("--render_view", type=int, default=-1, help="camera index (-1 = middle)")
    ap.add_argument("--scale_div", type=int, default=2)
    ap.add_argument("--ik_iters", type=int, default=400)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="exps/anim")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out, exist_ok=True)
    s = args.scale_div

    smal = SMALModel(dev)
    frames = json.load(open(os.path.join(args.scene_dir, "preprocess", "cameras.json")))["frames"]
    rv = len(frames) // 2 if args.render_view < 0 else args.render_view
    cam = load_cam(args.scene_dir, frames[rv], s, dev)

    # avatar identity + resting placement (rinda's offline fit)
    gt = load_pseudo_gt(args.scene_dir, "preprocess", smal.num_betas, dev)
    canonical = smal.canonical_verts(gt["betas"], gt["limbs"])              # [1,N,3]
    glob_orient = gt["theta"][:, :1]                                        # keep avatar's facing

    # appearance input image (224)
    img = Image.open(os.path.join(args.scene_dir, frames[args.input_view]["image_path"])).convert("RGB")
    inp = torch.from_numpy(np.asarray(img.resize((224, 224))).astype(np.float32) / 255.)
    inp = inp.permute(2, 0, 1)[None].to(dev)

    model = DogLRM().to(dev)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location=dev), strict=False)
    assert not unexpected, unexpected
    model.eval()
    print(f"loaded {args.ckpt} (dino re-init from pretrained, {len(missing)} missing = dino.*)")

    # retarget motion -> body pose theta
    name = os.path.splitext(os.path.basename(args.motion))[0]
    tgt = load_motion_joints(args.motion, dev)
    mbetas = load_motion_betas(args.template_dir, name, smal.num_betas, dev)
    fit = fit_theta_to_joints(smal, mbetas, tgt, iters=args.ik_iters, device=dev)
    T = tgt.shape[0]
    print(f"animating {T} frames of '{name}' onto {os.path.basename(os.path.dirname(args.scene_dir))}")

    bg = torch.ones(3, device=dev)
    out_frames = []
    with torch.no_grad():
        for t in range(T):
            theta = torch.cat([glob_orient, fit["theta"][t:t + 1, 1:]], dim=1)   # avatar facing + motion body
            posed = smal.posed_verts(gt["betas"], gt["limbs"], theta, gt["trans"], gt["scale"])
            gs = model(inp, canonical, posed, subdivide=smal.subdivide)
            rgb, _ = render_gaussians(gs["means"][0], gs["quats"][0], gs["scales"][0],
                                      gs["opacities"][0], gs["rgb"][0],
                                      cam["c2w"], cam["K"], cam["W"], cam["H"], bg=bg)
            out_frames.append((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    import imageio
    mp4 = os.path.join(args.out, f"{os.path.basename(os.path.dirname(args.scene_dir))}_{name}.mp4")
    imageio.mimsave(mp4, out_frames, fps=args.fps, quality=8)
    print(f"saved {mp4}  ({T} frames @ {args.fps}fps, {out_frames[0].shape[1]}x{out_frames[0].shape[0]})")


if __name__ == "__main__":
    main()
