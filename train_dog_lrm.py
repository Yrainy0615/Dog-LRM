#!/usr/bin/env python3
"""Minimal Dog-LRM training: batch over scenes (= the baseline's identity batch dim).

Each scene contributes 1 input view -> Gaussians on its teacher-forced posed SMAL ->
rendered to K random other views of that scene -> masked photometric loss. The model
forward is batched [B,N,...] across scenes (same 3889-vertex topology); rendering loops
per scene (each Gaussian set -> its own cameras). BARC/SMAL/pose frozen.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.model import DogLRM
from dog_lrm.render import intrinsics, render_gaussians, save_ply
from dog_lrm.smal_model import SMALModel, load_pseudo_gt


def load_view(scene_dir, fr, s, device):
    img = Image.open(os.path.join(scene_dir, fr["image_path"])).convert("RGB")
    W, H = fr["width"] // s, fr["height"] // s
    rgb = torch.from_numpy(np.asarray(img.resize((W, H))).astype(np.float32) / 255.).to(device)
    stem = os.path.splitext(fr["name"])[0]
    m = Image.open(os.path.join(scene_dir, "preprocess", "masks", stem + ".png")).convert("L")
    mask = torch.from_numpy(np.asarray(m.resize((W, H))).astype(np.float32) / 255.).to(device)
    K = intrinsics(fr["fx"] / s, fr["fy"] / s, fr["cx"] / s, fr["cy"] / s, device)
    return dict(rgb=rgb, mask=mask[..., None], K=K,
                c2w=torch.tensor(fr["c2w"], device=device).float(), W=W, H=H)


def load_scene(scene_dir, smal, s, device):
    frames = json.load(open(os.path.join(scene_dir, "preprocess", "cameras.json")))["frames"]
    views = [load_view(scene_dir, fr, s, device) for fr in frames]
    gt = load_pseudo_gt(scene_dir, "preprocess", smal.num_betas, device)
    canonical = smal.canonical_verts(gt["betas"], gt["limbs"])[0]
    posed = smal.posed_verts(gt["betas"], gt["limbs"], gt["theta"], gt["trans"], gt["scale"])[0]
    inputs_all = torch.stack([                                    # 224 input for every view
        F.interpolate(v["rgb"].permute(2, 0, 1)[None], size=(224, 224),
                      mode="bilinear", align_corners=False)[0] for v in views])
    return dict(name=os.path.basename(os.path.dirname(scene_dir)), views=views,
                canonical=canonical, posed=posed, inputs_all=inputs_all)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", help="train on every <root>/*/colmap scene")
    g.add_argument("--scene_dirs", nargs="+")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lpips_weight", type=float, default=0.1)
    ap.add_argument("--lpips_start", type=int, default=200, help="warm up without LPIPS")
    ap.add_argument("--offset_reg", type=float, default=0.1, help="ACAP: pull Gaussians to anchor")
    ap.add_argument("--offset_free", type=float, default=0.02, help="free offset distance (no penalty)")
    ap.add_argument("--scale_reg", type=float, default=0.1, help="ball: penalize anisotropic Gaussians")
    ap.add_argument("--scale_ratio", type=float, default=4.0, help="free max/min axis ratio")
    ap.add_argument("--scale_clip_start", type=float, default=0.02, help="max scale at it 0")
    ap.add_argument("--scale_clip_end", type=float, default=0.08, help="max scale after warmup")
    ap.add_argument("--scale_clip_warmup", type=int, default=300, help="iters to ramp max scale (0=off)")
    ap.add_argument("--vis_every", type=int, default=250)
    ap.add_argument("--n_subdiv", type=int, default=2, help="loop subdiv levels (2~62k verts; 3 is heavy)")
    ap.add_argument("--scale_div", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="exps/dog_lrm_dbg")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out, exist_ok=True)

    if args.root:
        scene_dirs = sorted(glob.glob(os.path.join(args.root, "*", "colmap")))
    else:
        scene_dirs = args.scene_dirs

    smal = SMALModel(dev, n_subdiv=args.n_subdiv)
    scenes = [load_scene(sd, smal, args.scale_div, dev) for sd in scene_dirs]
    B = len(scenes)
    print(f"{B} scenes: " + ", ".join(s["name"] for s in scenes))

    canonical = torch.stack([s["canonical"] for s in scenes])  # [B,N,3]
    posed = torch.stack([s["posed"] for s in scenes])

    model = DogLRM().to(dev)
    print(f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(dev)
    for p in lpips_fn.parameters():
        p.requires_grad = False
    white = torch.ones(3, device=dev)

    for it in range(args.iters):
        if args.scale_clip_warmup > 0:                            # ramp max scale to avoid early floaters
            t = min(1.0, it / args.scale_clip_warmup)
            scale_clip = args.scale_clip_start + t * (args.scale_clip_end - args.scale_clip_start)
        else:
            scale_clip = args.scale_clip_end
        ref = [int(np.random.randint(len(s["views"]))) for s in scenes]  # random reference view
        inputs = torch.stack([scenes[b]["inputs_all"][ref[b]] for b in range(B)])
        gs = model(inputs, canonical, posed, subdivide=smal.subdivide, scale_clip=scale_clip)
        loss_rgb = loss_mask = loss_perc = 0.0
        n = 0
        for b, scene in enumerate(scenes):
            for j in range(len(scene["views"])):     # supervise on every other colmap view
                if j == ref[b]:
                    continue
                v = scene["views"][j]
                rgb, alpha = render_gaussians(gs["means"][b], gs["quats"][b], gs["scales"][b],
                                              gs["opacities"][b], gs["rgb"][b], v["c2w"], v["K"],
                                              v["W"], v["H"], bg=white)
                gt_w = v["rgb"] * v["mask"] + (1 - v["mask"]) * white
                # RGB loss on the foreground only: a collapsed (transparent->white) render
                # then mismatches the dog and is penalized (avoids the small-fg vanish trap).
                loss_rgb = loss_rgb + F.l1_loss(rgb * v["mask"], v["rgb"] * v["mask"])
                loss_mask = loss_mask + F.l1_loss(alpha, v["mask"])
                if it >= args.lpips_start:
                    r = F.interpolate(rgb.permute(2, 0, 1)[None] * 2 - 1, 256, mode="bilinear", align_corners=False)
                    g = F.interpolate(gt_w.permute(2, 0, 1)[None] * 2 - 1, 256, mode="bilinear", align_corners=False)
                    loss_perc = loss_perc + lpips_fn(r, g).mean()
                n += 1
        # attribute regularizers (no per-view loop; act on the whole batched cloud)
        off = gs["offset"].norm(dim=-1)                           # ACAP: keep near anchor
        loss_off = (off.clamp(min=args.offset_free) - args.offset_free).mean()
        sc = gs["scales"]                                         # ball: discourage spiky Gaussians
        ratio = sc.max(dim=-1).values / (sc.min(dim=-1).values + 1e-6)
        loss_ball = (ratio.clamp(min=args.scale_ratio) - args.scale_ratio).mean()
        loss = ((loss_rgb + loss_mask + args.lpips_weight * loss_perc) / n
                + args.offset_reg * loss_off + args.scale_reg * loss_ball)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if it % args.vis_every == 0 or it == args.iters - 1:
            lp = float(loss_perc) / n if it >= args.lpips_start else 0.0
            print(f"it{it:4d} loss={float(loss):.4f} rgb={float(loss_rgb)/n:.4f} "
                  f"mask={float(loss_mask)/n:.4f} lpips={lp:.4f} "
                  f"off={float(loss_off):.4f} ball={float(loss_ball):.4f} sclip={scale_clip:.3f}")
            with torch.no_grad():
                tiles = []
                for b, scene in enumerate(scenes):
                    v = scene["views"][len(scene["views"]) // 2]
                    rgb, _ = render_gaussians(gs["means"][b], gs["quats"][b], gs["scales"][b],
                                              gs["opacities"][b], gs["rgb"][b], v["c2w"], v["K"],
                                              v["W"], v["H"], bg=torch.ones(3, device=dev))
                    pair = np.concatenate([(rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                                           (v["rgb"].cpu().numpy() * 255).astype(np.uint8)], axis=1)
                    tiles.append(pair)
                hmin = min(t.shape[0] for t in tiles)
                Image.fromarray(np.concatenate([t[:hmin] for t in tiles], axis=1)).save(
                    os.path.join(args.out, f"it{it:04d}.png"))

    # export final Gaussians (3DGS .ply) per scene
    model.eval()
    with torch.no_grad():
        gs = model(inputs, canonical, posed, subdivide=smal.subdivide)
    for b, scene in enumerate(scenes):
        save_ply(os.path.join(args.out, scene["name"] + ".ply"),
                 gs["means"][b], gs["scales"][b], gs["quats"][b], gs["opacities"][b], gs["rgb"][b])
    print(f"saved {B} .ply -> {args.out}")

    # save trainable weights only (skip frozen DINO) for reuse (e.g. animation)
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("dino.")}
    torch.save(sd, os.path.join(args.out, "model.pt"))
    print(f"saved model.pt ({len(sd)} tensors) -> {args.out}")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
