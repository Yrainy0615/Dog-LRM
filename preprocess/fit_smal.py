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
import time

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
    # Soft Dice per view (mean over views). Unlike MSE, an empty prediction gives
    # Dice=1 (max loss), so the optimizer can't cheat by shrinking the mesh away
    # when the foreground is small. Small MSE term added for edge refinement.
    num = 2.0 * (pred * masks).sum((1, 2))
    den = pred.sum((1, 2)) + masks.sum((1, 2)) + 1e-6
    dice = (1.0 - num / den).mean()
    return dice + 0.2 * ((pred - masks) ** 2).mean()


# --------------------------------------------------------------------------- #
# Multi-view 2D keypoint reprojection (semantic supervision)
# --------------------------------------------------------------------------- #
def load_keypoints(scene_dir, frames, out_subdir, render_hw, device, conf_thr):
    """BARC hourglass 2D keypoints (24) per view, mapped crop-256 -> render-px.
    Returns target [N,24,2] (render px) and weights [N,24] (confidence, gated)."""
    from configs.SMAL_configs import KEY_VIDS
    h, w = render_hw
    pre = os.path.join(scene_dir, out_subdir)
    boxes = json.load(open(os.path.join(pre, "barc_crops", "crop_boxes.json")))
    params = json.load(open(os.path.join(pre, "barc_init", "params.json")))
    n = len(frames)
    tgt = torch.zeros(n, KEY_VIDS.shape[0], 2, device=device)
    wt = torch.zeros(n, KEY_VIDS.shape[0], device=device)
    for i, fr in enumerate(frames):
        key = os.path.splitext(fr["name"])[0] + ".jpg"
        if key not in boxes or key not in params or "keypoints_norm" not in params[key]:
            continue
        x0, y0, x1, y1 = boxes[key]["box"]
        cw, ch = x1 - x0, y1 - y0
        s = max(cw, ch)                                   # load_barc_crop pads to square
        padx, pady = (s - cw) // 2, (s - ch) // 2
        kp = np.array(params[key]["keypoints_norm"], dtype=np.float32)      # (24,2) [-1,1]
        sc = np.array(params[key]["keypoints_scores"], dtype=np.float32)    # (24,)
        sq = (kp + 1.0) / 2.0 * (256 - 1) * (s / 256.0)   # crop-256 -> padded-square px
        orig_x = x0 + sq[:, 0] - padx                     # padded-square -> original px
        orig_y = y0 + sq[:, 1] - pady
        tgt[i, :, 0] = torch.from_numpy(orig_x * (w / fr["width"])).to(device)
        tgt[i, :, 1] = torch.from_numpy(orig_y * (h / fr["height"])).to(device)
        wt[i] = torch.from_numpy(sc).to(device)
    wt = torch.where(wt < conf_thr, torch.zeros_like(wt), wt)
    return tgt, wt


def build_kp_regressor(smal, device):
    """[24, V] averaging matrix: each SMAL keypoint = mean of its KEY_VIDS vertex group
    (groups have variable size, so KEY_VIDS is a ragged object array)."""
    from configs.SMAL_configs import KEY_VIDS
    v = int(smal.faces.max().item()) + 1
    reg = torch.zeros(len(KEY_VIDS), v, device=device)
    for j, g in enumerate(KEY_VIDS):
        g = np.atleast_1d(g).astype(np.int64)
        reg[j, g] = 1.0 / len(g)
    return reg


def project_keypoints(smal, p, cameras, kp_reg, device):
    """Project SMAL keypoints (kp_reg @ verts) into every camera."""
    theta = torch.cat([p["global_orient"], p["body_pose"]], dim=1)
    trans0 = torch.zeros(1, 3, device=device)
    verts, _, _ = smal(beta=p["betas"], betas_limbs=p["betas_limbs"],
                       theta=theta, trans=trans0, get_skin=True)
    world = torch.exp(p["log_scale"]) * verts + p["trans"][:, None, :]   # [1,V,3]
    kp3d = torch.einsum("kv,bvc->bkc", kp_reg, world)                    # [1,24,3]
    n = cameras.R.shape[0]
    scr = cameras.transform_points_screen(kp3d.repeat(n, 1, 1))          # [n,24,3]
    return scr[..., :2]


def keypoint_loss(proj, target, weights, render_res):
    # confidence-weighted mean squared error, px normalized to [0,1] so it is
    # comparable in scale to the dice silhouette term.
    err = ((proj - target) / render_res).pow(2).sum(-1)                  # [N,24]
    return (weights * err).sum() / (weights.sum() + 1e-6)


@torch.no_grad()
def init_scale_from_area(p, smal, renderer, cameras, masks, device):
    """Set log_scale so the rendered area roughly matches the mask area (area ~ scale^2)."""
    sil = render_silhouette(smal, p, renderer, cameras, device)
    a_pred = (sil > 0.5).float().mean().clamp_min(1e-4)
    a_gt = (masks > 0.5).float().mean().clamp_min(1e-4)
    p["log_scale"].fill_(float(0.5 * torch.log(a_gt / a_pred)))


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


def optimize(smal, p, renderer, cameras, masks, device, args, opt_params, iters, lr,
             kp_target=None, kp_weights=None, kp_reg=None, w_kp=0.0):
    # Soft-silhouette gradients can spike once converged; guard with grad clipping,
    # NaN handling, param clamps, and keep-best so a late blowup can't poison the result.
    opt = torch.optim.Adam([p[k] for k in opt_params], lr=lr)
    best_loss = float("inf")
    best = {k: p[k].detach().clone() for k in p}
    render_res = masks.shape[-1]
    use_kp = w_kp > 0 and kp_target is not None
    for it in range(iters):
        opt.zero_grad()
        pred = render_silhouette(smal, p, renderer, cameras, device)
        loss = (silhouette_loss(pred, masks)
                + prior_loss(p, args.w_beta, args.w_limb, args.w_pose))
        if use_kp:
            proj = project_keypoints(smal, p, cameras, kp_reg, device)
            loss = loss + w_kp * keypoint_loss(proj, kp_target, kp_weights, render_res)
        if not torch.isfinite(loss):
            break  # numerical blowup -> stop, keep best
        loss.backward()
        for k in opt_params:
            if p[k].grad is not None:
                torch.nan_to_num_(p[k].grad)
        torch.nn.utils.clip_grad_norm_([p[k] for k in opt_params], 1.0)
        opt.step()
        with torch.no_grad():
            p["log_scale"].clamp_(-1.5, 1.5)  # keep dog at a sane size; no vanishing
            p["trans"].clamp_(-3.0, 3.0)
        l = float(loss)
        if l < best_loss:
            best_loss = l
            for k in p:
                best[k] = p[k].detach().clone()
    with torch.no_grad():
        for k in p:
            p[k].data.copy_(best[k])
    return best_loss


def load_barc_init(scene_dir, out_subdir, num_betas, device):
    """BARC init: median shape (padded to num_betas) + representative-view articulation.
    Only body_pose (joints 1..34) is frame-invariant and reused; global_orient (root) is
    in BARC's camera frame, so it's left to the fit to recover in the COLMAP world frame."""
    from pytorch3d.transforms import matrix_to_axis_angle
    d = json.load(open(os.path.join(scene_dir, out_subdir, "barc_init", "params.json")))
    keys = list(d)
    betas = np.median([d[k]["betas"] for k in keys], axis=0)
    limbs = np.median([d[k]["betas_limbs"] for k in keys], axis=0)
    rep = np.array(d[keys[len(keys) // 2]]["pose_rotmat"], dtype=np.float32)  # (35,3,3)
    theta = matrix_to_axis_angle(torch.from_numpy(rep)).numpy()              # (35,3)
    b = np.zeros(num_betas, dtype=np.float32)
    b[:len(betas)] = betas
    return (torch.tensor(b, device=device),
            torch.tensor(limbs, dtype=torch.float32, device=device),
            torch.tensor(theta[1:], dtype=torch.float32, device=device))     # body_pose (34,3)


def orientation_candidates(device):
    """Dense-ish SO(3) cover (yaw x pitch) for the COLMAP-frame global orientation."""
    from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_axis_angle
    yaws = torch.linspace(0, 2 * np.pi, 9)[:-1]      # 8
    pitches = torch.tensor([-1.0, 0.0, 1.0])         # ~ +-57 deg
    cands = []
    for y in yaws:
        for p in pitches:
            R = euler_angles_to_matrix(torch.tensor([float(p), float(y), 0.0]), "XYZ")
            cands.append(matrix_to_axis_angle(R))
    return torch.stack(cands).to(device)             # [24,3]


def fit_scene(scene_dir, smal, args, device):
    cam_json = os.path.join(scene_dir, args.out_subdir, "cameras.json")
    frames = json.load(open(cam_json))["frames"]
    render_hw = (args.render_res, args.render_res)

    cameras = build_cameras(frames, render_hw, device)
    masks = load_masks(scene_dir, frames, args.out_subdir, render_hw, device)
    renderer = make_renderer(cameras, render_hw, device)

    kp_reg = build_kp_regressor(smal, device)
    kp_tgt, kp_wt = load_keypoints(scene_dir, frames, args.out_subdir, render_hw,
                                   device, args.kp_conf_thr)
    if float(kp_wt.sum()) == 0.0:
        kp_tgt = None  # no usable keypoints -> silhouette-only

    barc = load_barc_init(scene_dir, args.out_subdir, smal.num_betas, device) if args.init_barc else None
    rigid = ["global_orient", "trans", "log_scale"]

    if barc is not None:
        # BARC gives shape + articulation; densely search the COLMAP-frame orientation
        # (short screen), then refine rigid placement, then UNLOCK shape+pose guided by
        # multi-view silhouette + 2D keypoints so stuck verts can move to the right place.
        def fresh(go):
            p = init_params(smal, device)
            with torch.no_grad():
                p["betas"].copy_(barc[0]); p["betas_limbs"].copy_(barc[1])
                p["body_pose"].copy_(barc[2]); p["global_orient"][0, 0].copy_(go)
            init_scale_from_area(p, smal, renderer, cameras, masks, device)
            return p
        best_p, best_loss = None, float("inf")
        for go in orientation_candidates(device):
            p = fresh(go)
            loss = optimize(smal, p, renderer, cameras, masks, device, args,
                            rigid, iters=args.screen_iters, lr=args.lr_rigid,
                            kp_target=kp_tgt, kp_weights=kp_wt, kp_reg=kp_reg, w_kp=args.w_kp)
            if args.screen_by_kp and kp_tgt is not None:
                # select orientation by keypoints alone: the silhouette dice is
                # nearly front-back symmetric for fluffy dogs and picks flips
                with torch.no_grad():
                    proj = project_keypoints(smal, p, cameras, kp_reg, device)
                    loss = float(keypoint_loss(proj, kp_tgt, kp_wt, args.render_res))
            if loss < best_loss:
                best_loss, best_p = loss, p
        p = best_p
        final = optimize(smal, p, renderer, cameras, masks, device, args,
                         rigid, iters=args.iters_rigid, lr=args.lr_rigid,
                         kp_target=kp_tgt, kp_weights=kp_wt, kp_reg=kp_reg, w_kp=args.w_kp)
        if args.unlock_iters > 0:
            final = optimize(smal, p, renderer, cameras, masks, device, args,
                             list(p.keys()), iters=args.unlock_iters, lr=args.lr_unlock,
                             kp_target=kp_tgt, kp_weights=kp_wt, kp_reg=kp_reg, w_kp=args.w_kp)
    else:
        best_p, best_loss = None, float("inf")
        for yaw in [0.0, 1.5708, 3.1416, 4.7124]:
            p = init_params(smal, device)
            with torch.no_grad():
                p["global_orient"][0, 0, 1] = yaw
            init_scale_from_area(p, smal, renderer, cameras, masks, device)
            loss = optimize(smal, p, renderer, cameras, masks, device, args,
                            rigid, iters=args.iters_rigid, lr=args.lr_rigid)
            if loss < best_loss:
                best_loss, best_p = loss, p
        p = best_p
        final = optimize(smal, p, renderer, cameras, masks, device, args,
                         list(p.keys()), iters=args.iters_full, lr=args.lr_full)

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
                   args.out_subdir, device, kp_tgt, kp_wt, kp_reg)
    print(f"[ok] {scene_dir}: final_loss={final:.4f}, scale={out['scale']:.3f}")
    return final


@torch.no_grad()
def save_debug(scene_dir, frames, smal, p, renderer, cameras, masks, out_subdir, device,
               kp_tgt=None, kp_wt=None, kp_reg=None):
    from PIL import ImageDraw
    pred = render_silhouette(smal, p, renderer, cameras, device).cpu().numpy()
    proj = (project_keypoints(smal, p, cameras, kp_reg, device).cpu().numpy()
            if kp_tgt is not None else None)
    dbg = os.path.join(scene_dir, out_subdir, "smal_debug")
    os.makedirs(dbg, exist_ok=True)
    h, w = pred.shape[1:]
    for i, fr in enumerate(frames):
        img = Image.open(os.path.join(scene_dir, fr["image_path"])).convert("RGB").resize((w, h))
        ov = np.array(img).astype(np.float32)
        ov[..., 0] = np.clip(ov[..., 0] + 120 * pred[i], 0, 255)  # red = SMAL silhouette
        im = Image.fromarray(ov.astype(np.uint8))
        if proj is not None:
            d = ImageDraw.Draw(im)
            tg = kp_tgt[i].cpu().numpy(); wt = kp_wt[i].cpu().numpy()
            for j in range(tg.shape[0]):
                if wt[j] > 0:  # yellow = BARC detection (target), green = SMAL projection
                    x, y = tg[j]; d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 0))
                x, y = proj[i, j]; d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 0))
        im.save(os.path.join(dbg, os.path.splitext(fr["name"])[0] + ".png"))


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
    # shape/pose priors are weak so the unlock stage can adapt geometry; keypoints +
    # multi-view silhouette are the data terms. Validated on 00139-ten (loss 0.43->0.19).
    ap.add_argument("--w_beta", type=float, default=0.1)
    ap.add_argument("--w_limb", type=float, default=0.1)
    ap.add_argument("--w_pose", type=float, default=0.05)
    ap.add_argument("--w_kp", type=float, default=3.0,
                    help="weight of the multi-view 2D keypoint reprojection loss")
    ap.add_argument("--kp_conf_thr", type=float, default=0.1,
                    help="drop BARC keypoints below this detection confidence")
    ap.add_argument("--unlock_iters", type=int, default=600,
                    help="final stage: unlock shape+pose (0 disables, keeps rigid-only fit)")
    ap.add_argument("--lr_unlock", type=float, default=0.005)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init_barc", action="store_true",
                    help="init shape+articulation from preprocess/barc_init; fit only rigid placement")
    ap.add_argument("--screen_by_kp", action="store_true",
                    help="pick the orientation candidate by keypoint loss only")
    ap.add_argument("--screen_iters", type=int, default=40,
                    help="short rigid iters per orientation candidate during BARC-init screening")
    ap.add_argument("--debug", action="store_true", help="save silhouette overlays")
    ap.add_argument("--oom_retries", type=int, default=4,
                    help="retry a scene this many times on CUDA OOM (shared GPU contention)")
    ap.add_argument("--oom_wait", type=float, default=45.0,
                    help="seconds to wait before an OOM retry (let co-tenant free memory)")
    args = ap.parse_args()

    device = args.device
    smal = load_smal(device)
    print(f"SMAL ready: num_betas={smal.num_betas}, "
          f"num_betas_logscale={smal.num_betas_logscale}")

    scenes = ([args.scene_dir] if args.scene_dir else
              [os.path.join(args.root, d) for d in sorted(os.listdir(args.root))
               if os.path.isdir(os.path.join(args.root, d))])
    for scene in scenes:
        for attempt in range(args.oom_retries + 1):
            try:
                fit_scene(scene, smal, args, device)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt < args.oom_retries:
                    torch.cuda.empty_cache()
                    print(f"[oom-retry {attempt+1}/{args.oom_retries}] {scene}: "
                          f"wait {args.oom_wait:.0f}s for GPU memory", flush=True)
                    time.sleep(args.oom_wait)
                else:
                    print(f"[skip] {scene}: {e}")
                    break
            except Exception as e:
                print(f"[skip] {scene}: {e}")
                break
    print(f"[done] {len(scenes)} scene(s)")


if __name__ == "__main__":
    main()
