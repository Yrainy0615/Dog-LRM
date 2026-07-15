#!/usr/bin/env python3
"""GS-only wind demo: fur dynamics with NO template mesh / skinning.

Everything is estimated from the gaussian cloud itself:
  - per-GS normal: PCA plane of 32-NN, oriented outward via the 512-NN centroid
  - coat coordinate rel: signed depth along the local normal, ranked within the
    64-NN shell (p20 = local root, p90 = local tip)
  - amplitude: local shell thickness (for a volumetric long coat this IS the
    local fur depth), normalized by its own p80
Wind machinery (gust + travelling ruffle + per-GS jitter, lift + slide + splat
bend) matches animate_fur_wind.py v6.

Usage: python animate_fur_wind_gsonly.py [ply] [updir xyz: default 0,0,1]
"""
import math, os, sys
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from dog_lrm.render import intrinsics, render_gaussians, load_ply
import animate_gs_coatdepth as base

dev = "cuda"
PLY = sys.argv[1] if len(sys.argv) > 1 else "train_data/gaussians_3dgs.ply"
OUT = "exps/coatdepth_demo"
UP = np.array([0.0, 0.0, 1.0], np.float32)         # this ply is z-up
FWD = np.array([1.0, 0.0, 0.0], np.float32)        # head towards +x
T, FPS = 72, 24


def lookat(eye, center, fx, W, H, up):
    eye = torch.tensor(eye, device=dev, dtype=torch.float32)
    center = torch.tensor(center, device=dev, dtype=torch.float32)
    fwd = torch.nn.functional.normalize(center - eye, dim=0)
    upw = torch.tensor(up, device=dev, dtype=torch.float32)
    right = torch.nn.functional.normalize(torch.cross(fwd, upw, dim=0), dim=0)
    upv = torch.cross(right, fwd, dim=0)
    c2w = torch.eye(4, device=dev)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, -upv, fwd, eye
    return c2w, intrinsics(fx, fx, W / 2, H / 2, dev)


def main():
    os.makedirs(OUT, exist_ok=True)
    gs = load_ply(PLY, dev)
    prune_op = float(os.environ.get("PRUNE_OP", "0"))
    if prune_op > 0:                            # drop ultra-sparse fringe splats
        keep = gs["opacities"] > prune_op
        gs = {k: v[keep] for k, v in gs.items()}
        print(f"pruned op<{prune_op}: kept {int(keep.sum())}/{len(keep)}")
    opg = float(os.environ.get("OP_GAMMA", "1.0"))
    if opg != 1.0:
        gs["opacities"] = gs["opacities"] ** opg
    mu0 = gs["means"]
    xyz = mu0.cpu().numpy().astype(np.float64)
    N = len(xyz)
    height = (xyz @ UP).max() - (xyz @ UP).min()
    ctr = xyz.mean(0)

    tree = cKDTree(xyz)
    idx32 = tree.query(xyz, k=32)[1]
    idx64 = tree.query(xyz, k=64)[1]
    idx512 = tree.query(xyz, k=512)[1]

    # PCA normals (smallest eigvec of 32-NN covariance), oriented outward
    nb = xyz[idx32]                                     # [N,32,3]
    d = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", d, d)
    _, vecs = np.linalg.eigh(cov)
    nrm = vecs[:, :, 0]
    outward = xyz - xyz[idx512].mean(1)
    nrm *= np.sign((nrm * outward).sum(-1, keepdims=True) + 1e-12)
    for _ in range(2):                                  # smooth + renormalize
        nrm = nrm[idx32].mean(1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True).clip(1e-8)

    # local coat coordinate: signed depth along own normal within the 64-NN shell
    ddep = np.einsum("nkj,nj->nk", xyz[idx64] - xyz[:, None], nrm)  # neighbor depths rel. to me
    p20 = np.percentile(ddep, 20, axis=1)
    p90 = np.percentile(ddep, 90, axis=1)
    t_loc = np.maximum(p90 - p20, 1e-4)
    rel = np.clip((0.0 - p20) / t_loc, 0.0, 1.0)        # my own depth is 0 by construction
    for _ in range(2):
        rel = rel[idx32].mean(1)
        t_loc = t_loc[idx32].mean(1)
    w = rel * rel
    amp_scale = np.clip(t_loc / np.percentile(t_loc, 80), 0.3, 1.2)

    # no skinning -> suppress rigid features geometrically: muzzle/eyes (cluster at
    # the head-front extreme along FWD) and paws (lowest band along UP)
    fx = xyz @ FWD
    face_c = xyz[fx > fx.max() - 0.08 * height].mean(0)
    dface = np.linalg.norm(xyz - face_c, axis=-1) / (0.25 * height)
    zz = xyz @ UP
    dfeet = (zz - zz.min()) / (0.10 * height)
    # color outliers = eyes/nose/tongue (fur is locally color-smooth)
    rgb = gs["rgb"].cpu().numpy().astype(np.float64)
    cdiff = np.linalg.norm(rgb - rgb[idx32].mean(1), axis=-1)
    cdiff = np.maximum(cdiff, cdiff[idx32].max(1))       # dilate to cover feature rims
    damp_c = 1.0 - np.clip((cdiff - 0.50) / 0.30, 0.0, 1.0)
    damp = np.clip(dface, 0.05, 1.0) ** 2 * np.clip(dfeet, 0.15, 1.0) * damp_c
    amp_scale = amp_scale * damp
    print(f"N={N} height={height:.3f}  t_loc mm: p20/50/80 "
          f"{np.percentile(t_loc*1000,[20,50,80]).round(1)}  w>0.3: {(w>0.3).mean():.3f}")

    w_t = torch.from_numpy((w * amp_scale).astype(np.float32)).to(dev)[:, None]
    nrm_t = torch.from_numpy(nrm.astype(np.float32)).to(dev)

    # wind setup (frame: z-up, head +x)
    AMP = 0.030 * height                                # tip displacement
    ROT_AMP = 0.35
    wind_np = FWD * 0.8 + np.array([0, 0.45, 0], np.float32) + UP * 0.15
    wind = torch.tensor(wind_np / np.linalg.norm(wind_np), device=dev)[None]
    jit = torch.sin(mu0 @ torch.tensor([127.1, 311.7, 74.7], device=dev)) * math.pi
    u1 = torch.sin(mu0 @ (torch.tensor([127.1, 311.7, 74.7], device=dev) * 2.7)) * 0.5 + 0.5
    kx = mu0 @ (torch.tensor([1.0, 0.5, 0.3], device=dev) * 2 * math.pi / (0.10 * height))
    bend_axis = torch.nn.functional.normalize(torch.linalg.cross(nrm_t, wind.expand_as(nrm_t)), dim=-1)
    # needle splats poking out when rotated -> damp bend by elongation
    s_sorted = gs["scales"].sort(dim=1).values
    rot_damp = (4.0 / (s_sorted[:, 2] / s_sorted[:, 1].clamp_min(1e-9))).clamp(0.25, 1.0)

    def wind_fields(t):
        gust = 0.55 + 0.45 * math.sin(2 * math.pi * 0.35 * t + 1.2)
        wave = torch.sin(2 * math.pi * 1.1 * t - kx + 0.3 * jit)
        ind = torch.sin(2 * math.pi * (1.6 + 1.2 * u1) * t + 7.0 * jit)
        return gust, wave, ind

    def wind_apply(t, wdir=None, bend=None, boost=1.0):
        wd = wind if wdir is None else wdir[None]
        ba = bend_axis if bend is None else bend
        gust, wave, ind = wind_fields(t)
        drive = 0.6 * wave + 0.4 * ind
        lift = nrm_t * (boost * AMP * 0.6 * gust * drive)[:, None]
        slide = wd * (boost * AMP * 0.7 * gust * drive)[:, None]
        mu = mu0 + w_t * (lift + slide)
        drive_r = 0.85 * wave + 0.15 * ind      # bend coherently: lone needles = spikes
        ang = 0.5 * boost * ROT_AMP * rot_damp * w_t[:, 0] * gust * drive_r
        qd = torch.cat([torch.cos(ang)[:, None], ba * torch.sin(ang)[:, None]], -1)
        return mu, base.quat_mul(qd, gs["quats"])

    if os.environ.get("HIRES"):                 # 2160p/1080p mp4, slow 360 orbit (z-up)
        import subprocess
        four_k = os.environ["HIRES"] == "4k"
        mode = os.environ.get("MODE", "dynamic")
        bgname = os.environ.get("BG", "white")
        bgc = torch.ones(3, device=dev) if bgname == "white" else torch.zeros(3, device=dev)
        W2, H2 = (3840, 2160) if four_k else (1920, 1080)
        FX2 = 4700.0 if four_k else 2350.0
        NF = int(os.environ.get("NF", "480")); BOOST = 1.35
        name = os.path.splitext(os.path.basename(PLY))[0]
        path = f"{OUT}/pom_{mode}_{bgname}_{'4k' if four_k else 'hires'}.mp4"
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W2}x{H2}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264",
             "-preset", "medium" if four_k else "slow", "-crf", "17",
             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        cx = ctr + UP * 0.02                    # look-at
        with torch.no_grad():
            for f in range(NF):
                th = 2 * math.pi * f / NF
                eye = cx + np.array([math.cos(th), math.sin(th), 0.0]) * 1.25 + UP * 0.20
                c2w, K = lookat(eye, cx, FX2, W2, H2, UP)
                if mode == "static":
                    mu, q = mu0, gs["quats"]
                else:
                    wdir = torch.nn.functional.normalize(0.5 * wind[0] + 0.65 * c2w[:3, 0], dim=0)
                    bend = torch.nn.functional.normalize(
                        torch.linalg.cross(nrm_t, wdir[None].expand_as(nrm_t)), dim=-1)
                    mu, q = wind_apply(f / FPS, wdir, bend, BOOST)
                rgb, _ = render_gaussians(mu, q, gs["scales"], gs["opacities"], gs["rgb"],
                                          c2w, K, W2, H2, bg=bgc)
                proc.stdin.write((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).tobytes())
        proc.stdin.close()
        proc.wait()
        print("saved", path)
        return

    # cameras: side full body + rear-3/4 close-up on the fluff
    L = height
    side_eye = ctr + np.array([0.1, -1, 0]) / np.linalg.norm([0.1, -1, 0]) * 2.6 * L + UP * 0.35 * L
    rear_eye = ctr + np.array([-1, -0.75, 0]) / np.linalg.norm([-1, -0.75, 0]) * 1.7 * L + UP * 0.55 * L
    views = [lookat(side_eye, ctr, 900.0, 640, 640, UP),
             lookat(rear_eye, ctr + UP * 0.1 * L, 1100.0, 640, 640, UP)]
    white = torch.ones(3, device=dev)

    frames = []
    with torch.no_grad():
        for f in range(T):
            mu, q = wind_apply(f / FPS)
            row = []
            for c2w, K in views:
                rgb, _ = render_gaussians(mu, q, gs["scales"], gs["opacities"], gs["rgb"],
                                          c2w, K, 640, 640, bg=white)
                row.append((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            frames.append(np.concatenate(row, 1))
    name = os.path.splitext(os.path.basename(PLY))[0]
    path = f"{OUT}/wind_gsonly_{name}.gif"
    Image.fromarray(frames[0]).save(path, save_all=True,
                                    append_images=[Image.fromarray(x) for x in frames[1:]],
                                    duration=1000 // FPS, loop=0)
    print("saved", path)


if __name__ == "__main__":
    main()
