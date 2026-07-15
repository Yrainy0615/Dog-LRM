#!/usr/bin/env python3
"""Animate a baked Gaussian .ply (e.g. v5 rinda export) with an OmniMotionGPT motion.

No model weights needed: the .ply Gaussians correspond to subdivided SMAL vertices
(g -> vertex g//K). We IK-fit the motion to per-frame SMAL pose, then re-pose every
Gaussian by the relative per-vertex LBS transform M = affine(theta_t)·inv(affine(theta_0)),
transforming center + orientation. Renders 4 COLMAP views as a 2x2 grid -> mp4.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.render import intrinsics, render_gaussians, load_ply
from dog_lrm.smal_model import SMALModel, load_pseudo_gt
from dog_lrm.motion import (load_motion_joints, load_motion_betas, fit_theta_to_joints,
                            lbs_world_affine, mat_to_quat, quat_mul)


def load_cam(fr, s, device):
    return dict(K=intrinsics(fr["fx"] / s, fr["fy"] / s, fr["cx"] / s, fr["cy"] / s, device),
                c2w=torch.tensor(fr["c2w"], device=device).float(),
                W=fr["width"] // s, H=fr["height"] // s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--ply", required=True, help="baked Gaussian .ply (v5 export)")
    ap.add_argument("--motion", required=True)
    ap.add_argument("--template_dir",
                    default="third_party/OmniMotionGPT/data/unzipped/animals_smal_template")
    ap.add_argument("--views", nargs="+", default=["148", "153"],
                    help="colmap image-name stems to render (side by side)")
    ap.add_argument("--abs_pose", action="store_true",
                    help="use the motion's absolute pose (default: start from canonical "
                         "rest stance and deform by the motion delta)")
    ap.add_argument("--scale_div", type=int, default=6)
    ap.add_argument("--ik_iters", type=int, default=400)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="exps/anim")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out, exist_ok=True)

    smal = SMALModel(dev)                                  # n_subdiv=1 (matches v5 export)
    Vsub = smal.subdiv_M.shape[0]
    frames = json.load(open(os.path.join(args.scene_dir, "preprocess", "cameras.json")))["frames"]
    name2idx = {f["name"].split(".")[0]: i for i, f in enumerate(frames)}
    vidx = [name2idx[v] for v in args.views]
    cams = [load_cam(frames[i], args.scale_div, dev) for i in vidx]
    print(f"render views {args.views} (idx {vidx}) @ {cams[0]['W']}x{cams[0]['H']}")

    gt = load_pseudo_gt(args.scene_dir, "preprocess", smal.num_betas, dev)
    gs = load_ply(args.ply, dev)
    G = gs["means"].shape[0]
    assert G % Vsub == 0, (G, Vsub)
    K = G // Vsub
    vert = torch.arange(G, device=dev) // K               # gaussian -> subdiv vertex
    print(f"{G} gaussians = {Vsub} verts x {K}")

    # anchor pose = rinda's fit (the pose the .ply was baked at); invert once
    M0 = lbs_world_affine(smal, gt["betas"], gt["limbs"], gt["theta"], gt["trans"], gt["scale"])
    M0inv = torch.inverse(M0)                              # [Vsub,4,4]
    glob = gt["theta"][:, :1]

    # retarget motion -> body pose
    name = os.path.splitext(os.path.basename(args.motion))[0]
    tgt = load_motion_joints(args.motion, dev)
    mbetas = load_motion_betas(args.template_dir, name, smal.num_betas, dev)
    fit = fit_theta_to_joints(smal, mbetas, tgt, iters=args.ik_iters, device=dev)
    T = tgt.shape[0]
    print(f"animating {T} frames of '{name}'")

    body0 = fit["theta"][0:1, 1:]                 # motion's first-frame body pose
    bg = torch.ones(3, device=dev)
    out_frames = []
    with torch.no_grad():
        for t in range(T):
            body = fit["theta"][t:t + 1, 1:]
            if not args.abs_pose:                 # canonical start: zero at t=0, deform by delta
                body = body - body0
            theta = torch.cat([glob, body], dim=1)
            Mt = lbs_world_affine(smal, gt["betas"], gt["limbs"], theta, gt["trans"], gt["scale"])
            rel = (Mt @ M0inv)[vert]                       # [G,4,4]
            means = (rel[:, :3, :3] @ gs["means"][..., None])[..., 0] + rel[:, :3, 3]
            quats = quat_mul(mat_to_quat(rel[:, :3, :3]), gs["quats"])
            tiles = []
            for cam in cams:
                rgb, _ = render_gaussians(means, quats, gs["scales"], gs["opacities"],
                                          gs["rgb"], cam["c2w"], cam["K"], cam["W"], cam["H"], bg=bg)
                tiles.append((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            out_frames.append(np.concatenate(tiles, axis=1))     # views side by side

    import imageio
    pet = os.path.basename(os.path.dirname(args.scene_dir))
    mp4 = os.path.join(args.out, f"{pet}_{name}_{'_'.join(args.views)}.mp4")
    imageio.mimsave(mp4, out_frames, fps=args.fps, quality=8)
    print(f"saved {mp4}  ({T} frames, grid {out_frames[0].shape[1]}x{out_frames[0].shape[0]})")


if __name__ == "__main__":
    main()
