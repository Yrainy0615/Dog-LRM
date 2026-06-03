#!/usr/bin/env python3
"""Stage 4 preprocessing: multi-view silhouette fit of ONE shared SMAL per scene.

Each pet's COLMAP scene is a static multi-view capture -> a single SMAL instance
(shape + pose + global_orient + trans + scale) explains all views. We fit it by
rendering the SMAL mesh silhouette (pytorch3d soft rasterizer) into every view and
matching the foreground masks from Stage 2.

This is the FIRST-STEP fit that produces pseudo-GT SMAL; there is no trained Gaussian
model to re-render yet (that comes later, inside LHM training). Hence mesh silhouette.

Inputs : <scene>/preprocess/cameras.json   (Stage 1)
         <scene>/preprocess/masks/<name>.png (Stage 2)
Output : <scene>/preprocess/smal_params.json
         <scene>/preprocess/smal_debug/<name>.png  (with --debug)

Body model: BARC SMBLD dog SMAL (third_party/barc_release).
Deps: torch, pytorch3d, numpy, pillow, chumpy (BARC loads the .pkl via chumpy).
NOTE: not runnable in this repo's bare env; run on the GPU server.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def load_smal(device):
    barc_src = os.path.join(os.path.dirname(__file__), "..", "third_party",
                            "barc_release", "src")
    sys.path.insert(0, os.path.abspath(barc_src))
    from smal_pytorch.smal_model.smal_torch_new import SMAL
    return SMAL().to(device)


# --------------------------------------------------------------------------- #
# Cameras & masks
# --------------------------------------------------------------------------- #
def build_cameras(frames, render_hw, device):
    from pytorch3d.utils import cameras_from_opencv_projection
    h, w = render_hw
    R, t, K, sizes = [], [], [], []
    for fr in frames:
        c2w = torch.tensor(fr["c2w"], dtype=torch.float32)
        w2c = torch.inverse(c2w)
        sx, sy = w / fr["width"], h / fr["height"]  # intrinsics -> render res
        R.append(w2c[:3, :3])
        t.append(w2c[:3, 3])
        K.append(torch.tensor([[fr["fx"] * sx, 0, fr["cx"] * sx],
                               [0, fr["fy"] * sy, fr["cy"] * sy],
                               [0, 0, 1]], dtype=torch.float32))
        sizes.append([h, w])
    return cameras_from_opencv_projection(
        torch.stack(R).to(device), torch.stack(t).to(device),
        torch.stack(K).to(device),
        torch.tensor(sizes, dtype=torch.float32, device=device)).to(device)


def load_masks(scene_dir, frames, out_subdir, render_hw, device):
    h, w = render_hw
    masks = []
    for fr in frames:
        stem = os.path.splitext(fr["name"])[0]
        mp = os.path.join(scene_dir, out_subdir, "masks", stem + ".png")
        m = Image.open(mp).convert("L").resize((w, h))
        masks.append(torch.from_numpy(np.array(m)).float() / 255.0)
    return torch.stack(masks).to(device)  # [N,h,w]


def make_renderer(cameras, render_hw, device):
    from pytorch3d.renderer import (BlendParams, MeshRasterizer, MeshRenderer,
                                    RasterizationSettings, SoftSilhouetteShader)
    blend = BlendParams(sigma=1e-4, gamma=1e-4)
    raster = RasterizationSettings(
        image_size=tuple(render_hw),
        blur_radius=float(np.log(1.0 / 1e-4 - 1.0) * blend.sigma),
        faces_per_pixel=50)
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftSilhouetteShader(blend_params=blend))


# --------------------------------------------------------------------------- #
# SMAL forward -> per-view silhouette
# --------------------------------------------------------------------------- #
def render_silhouette(smal, p, renderer, cameras, device):
    from pytorch3d.structures import Meshes
    theta = torch.cat([p["global_orient"], p["body_pose"]], dim=1)  # [1,35,3]
    trans0 = torch.zeros(1, 3, device=device)
    verts, _, _ = smal(beta=p["betas"], betas_limbs=p["betas_limbs"],
                       theta=theta, trans=trans0, get_skin=True)
    world = torch.exp(p["log_scale"]) * verts + p["trans"][:, None, :]  # similarity
    n = cameras.R.shape[0]
    faces = smal.faces.long().to(device)
    meshes = Meshes(verts=world.repeat(n, 1, 1), faces=faces[None].repeat(n, 1, 1))
    return renderer(meshes, cameras=cameras)[..., 3]  # alpha [N,h,w]


def silhouette_loss(pred, masks):
    return ((pred - masks) ** 2).mean()


def init_params(smal, device):
    return {
        "global_orient": torch.zeros(1, 1, 3, device=device, requires_grad=True),
        "body_pose": torch.zeros(1, 34, 3, device=device, requires_grad=True),
        "betas": torch.zeros(1, smal.num_betas, device=device, requires_grad=True),
        "betas_limbs": torch.zeros(1, smal.num_betas_logscale, device=device,
                                   requires_grad=True),
        "trans": torch.zeros(1, 3, device=device, requires_grad=True),
        "log_scale": torch.zeros(1, device=device, requires_grad=True),
    }


def prior_loss(p, w_beta, w_limb, w_pose):
    return (w_beta * (p["betas"] ** 2).mean()
            + w_limb * (p["betas_limbs"] ** 2).mean()
            + w_pose * (p["body_pose"] ** 2).mean())


def optimize(smal, p, renderer, cameras, masks, device, args, opt_params, iters, lr):
    opt = torch.optim.Adam([p[k] for k in opt_params], lr=lr)
    last = None
    for it in range(iters):
        opt.zero_grad()
        pred = render_silhouette(smal, p, renderer, cameras, device)
        loss = (silhouette_loss(pred, masks)
                + prior_loss(p, args.w_beta, args.w_limb, args.w_pose))
        loss.backward()
        opt.step()
        last = float(loss)
    return last


def fit_scene(scene_dir, smal, args, device):
    cam_json = os.path.join(scene_dir, args.out_subdir, "cameras.json")
    frames = json.load(open(cam_json))["frames"]
    render_hw = (args.render_res, args.render_res)

    cameras = build_cameras(frames, render_hw, device)
    masks = load_masks(scene_dir, frames, args.out_subdir, render_hw, device)
    renderer = make_renderer(cameras, render_hw, device)

    # multi-start over a few global-orientation inits (silhouette fits are
    # orientation-ambiguous); keep the rigid-aligned start with lowest loss.
    yaw_inits = [0.0, 1.5708, 3.1416, 4.7124]
    best_p, best_loss = None, float("inf")
    for yaw in yaw_inits:
        p = init_params(smal, device)
        with torch.no_grad():
            p["global_orient"][0, 0, 1] = yaw  # init rotation about model up-ish axis
        loss = optimize(smal, p, renderer, cameras, masks, device,
                        args, ["global_orient", "trans", "log_scale"],
                        iters=args.iters_rigid, lr=args.lr_rigid)
        if loss < best_loss:
            best_loss, best_p = loss, p

    p = best_p
    final = optimize(smal, p, renderer, cameras, masks, device,
                     args, list(p.keys()), iters=args.iters_full, lr=args.lr_full)

    out = {
        "shared": True,
        "smal_model": "my_smpl_SMBLD_nbj_v3.pkl",
        "render_res": render_hw,
        "final_loss": final,
        "n_betas": smal.num_betas,
        "global_orient": p["global_orient"].detach().cpu().reshape(-1).tolist(),
        "body_pose": p["body_pose"].detach().cpu().reshape(34, 3).tolist(),
        "betas": p["betas"].detach().cpu().reshape(-1).tolist(),
        "betas_limbs": p["betas_limbs"].detach().cpu().reshape(-1).tolist(),
        "trans": p["trans"].detach().cpu().reshape(-1).tolist(),
        "scale": float(torch.exp(p["log_scale"]).detach().cpu()),
    }
    os.makedirs(os.path.join(scene_dir, args.out_subdir), exist_ok=True)
    json.dump(out, open(os.path.join(scene_dir, args.out_subdir,
                                     "smal_params.json"), "w"), indent=2)

    if args.debug:
        save_debug(scene_dir, frames, smal, p, renderer, cameras, masks,
                   args.out_subdir, device)
    print(f"[ok] {scene_dir}: final_loss={final:.4f}, scale={out['scale']:.3f}")
    return final


@torch.no_grad()
def save_debug(scene_dir, frames, smal, p, renderer, cameras, masks, out_subdir, device):
    pred = render_silhouette(smal, p, renderer, cameras, device).cpu().numpy()
    dbg = os.path.join(scene_dir, out_subdir, "smal_debug")
    os.makedirs(dbg, exist_ok=True)
    h, w = pred.shape[1:]
    for i, fr in enumerate(frames):
        img = Image.open(os.path.join(scene_dir, fr["image_path"])).convert("RGB").resize((w, h))
        ov = np.array(img).astype(np.float32)
        ov[..., 0] = np.clip(ov[..., 0] + 120 * pred[i], 0, 255)  # red = SMAL silhouette
        Image.fromarray(ov.astype(np.uint8)).save(
            os.path.join(dbg, os.path.splitext(fr["name"])[0] + ".png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene_dir")
    g.add_argument("--root")
    ap.add_argument("--out_subdir", default="preprocess")
    ap.add_argument("--render_res", type=int, default=256)
    ap.add_argument("--iters_rigid", type=int, default=150)
    ap.add_argument("--iters_full", type=int, default=400)
    ap.add_argument("--lr_rigid", type=float, default=0.05)
    ap.add_argument("--lr_full", type=float, default=0.01)
    ap.add_argument("--w_beta", type=float, default=1.0)
    ap.add_argument("--w_limb", type=float, default=1.0)
    ap.add_argument("--w_pose", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug", action="store_true", help="save silhouette overlays")
    args = ap.parse_args()

    device = args.device
    smal = load_smal(device)
    print(f"SMAL ready: num_betas={smal.num_betas}, "
          f"num_betas_logscale={smal.num_betas_logscale}")

    scenes = ([args.scene_dir] if args.scene_dir else
              [os.path.join(args.root, d) for d in sorted(os.listdir(args.root))
               if os.path.isdir(os.path.join(args.root, d))])
    for scene in scenes:
        try:
            fit_scene(scene, smal, args, device)
        except Exception as e:
            print(f"[skip] {scene}: {e}")
    print(f"[done] {len(scenes)} scene(s)")


if __name__ == "__main__":
    main()
