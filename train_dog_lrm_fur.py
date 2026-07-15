#!/usr/bin/env python3
"""Dog-LRM training with explicit body/fur Gaussian branches.

This is a research interface for testing the body/fur split:

  * body Gaussians stay close to the SMAL surface and explain low-frequency shape.
  * fur Gaussians are anchored to the same subdivided SMAL vertices, but their
    offsets are expressed in the local normal/tangent frame.
  * fur length is learned, with a zero-biased sparsity prior, smoothness on the
    SMAL mesh, and per-sample presets for short/long coat dynamics.

The default scene list is the short-hair rinda sample plus the long-hair bear
sample, if both exist in the workspace.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.model import fourier_embed
from dog_lrm.render import intrinsics, render_gaussians, save_ply
from dog_lrm.smal_model import SMALModel, load_pseudo_gt


DEFAULT_RINDA = "test/00208-rinda/colmap"
DEFAULT_BEAR = "received_data_from_Pinstudio_20260424/unzipped/0423/00062-bear/colmap"

FUR_PRESETS = {
    "short": {
        "lmax": 0.035,
        "sigma": 0.008,
        "lat": 0.010,
        "stiffness": 95.0,
        "damping": 18.0,
        "wind": 0.10,
        "gravity": 0.01,
    },
    "long": {
        "lmax": 0.120,
        "sigma": 0.032,
        "lat": 0.025,
        "stiffness": 24.0,
        "damping": 5.0,
        "wind": 0.45,
        "gravity": 0.035,
    },
}


def scene_name(scene_dir):
    return os.path.basename(os.path.dirname(os.path.normpath(scene_dir)))


def default_scene_dirs():
    return [p for p in (DEFAULT_RINDA, DEFAULT_BEAR) if os.path.exists(p)]


def infer_preset(name):
    low = name.lower()
    if "bear" in low or "kuma" in low:
        return "long"
    return "short"


def parse_preset_overrides(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Bad --preset item '{item}', expected name=short|long")
        name, preset = item.split("=", 1)
        if preset not in FUR_PRESETS:
            raise ValueError(f"Unknown fur preset '{preset}'")
        out[name] = preset
    return out


def load_view(scene_dir, fr, scale_div, device):
    img = Image.open(os.path.join(scene_dir, fr["image_path"])).convert("RGB")
    W, H = fr["width"] // scale_div, fr["height"] // scale_div
    rgb = torch.from_numpy(np.asarray(img.resize((W, H))).astype(np.float32) / 255.).to(device)
    stem = os.path.splitext(fr["name"])[0]
    mask_path = os.path.join(scene_dir, "preprocess", "masks", stem + ".png")
    mask = Image.open(mask_path).convert("L")
    mask = torch.from_numpy(np.asarray(mask.resize((W, H))).astype(np.float32) / 255.).to(device)
    K = intrinsics(fr["fx"] / scale_div, fr["fy"] / scale_div,
                   fr["cx"] / scale_div, fr["cy"] / scale_div, device)
    return dict(rgb=rgb, mask=mask[..., None], K=K,
                c2w=torch.tensor(fr["c2w"], device=device).float(), W=W, H=H,
                name=fr["name"])


def load_scene(scene_dir, smal, scale_div, device, preset_override=None):
    frames = json.load(open(os.path.join(scene_dir, "preprocess", "cameras.json")))["frames"]
    views = [load_view(scene_dir, fr, scale_div, device) for fr in frames]
    gt = load_pseudo_gt(scene_dir, "preprocess", smal.num_betas, device)
    canonical = smal.canonical_verts(gt["betas"], gt["limbs"])[0]
    posed = smal.posed_verts(gt["betas"], gt["limbs"], gt["theta"], gt["trans"], gt["scale"])[0]
    inputs_all = torch.stack([
        F.interpolate(v["rgb"].permute(2, 0, 1)[None], size=(224, 224),
                      mode="bilinear", align_corners=False)[0] for v in views
    ])
    diag = (posed.max(dim=0).values - posed.min(dim=0).values).norm().clamp_min(1e-6)
    name = scene_name(scene_dir)
    preset = preset_override or infer_preset(name)
    return dict(name=name, scene_dir=scene_dir, views=views, canonical=canonical,
                posed=posed, inputs_all=inputs_all, gt=gt, body_diag=diag,
                preset=preset)


def subdivided_faces(faces, n_subdiv):
    faces = faces.long().cpu()
    V = int(faces.max()) + 1
    for _ in range(n_subdiv):
        edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
        edges = torch.sort(edges, dim=1).values
        edges = torch.unique(edges, dim=0)
        eidx = {tuple(e.tolist()): V + i for i, e in enumerate(edges)}

        def mid(a, b):
            return eidx[tuple(sorted((int(a), int(b))))]

        nf = []
        for f in faces:
            a, b, c = f.tolist()
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        faces = torch.tensor(nf, dtype=torch.long)
        V += edges.shape[0]
    return faces


def unique_edges(faces):
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def vertex_normals(verts, faces):
    """verts [B,V,3], faces [F,3] -> unit vertex normals [B,V,3]."""
    faces = faces.to(verts.device)
    v0, v1, v2 = verts[:, faces[:, 0]], verts[:, faces[:, 1]], verts[:, faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = torch.zeros_like(verts)
    for k in range(3):
        idx = faces[:, k][None, :, None].expand(verts.shape[0], -1, 3)
        normals.scatter_add_(1, idx, fn)
    return F.normalize(normals, dim=-1, eps=1e-6)


def orient_normals_outward(verts, normals):
    center = verts.mean(dim=1, keepdim=True)
    outward = ((verts - center) * normals).sum(dim=-1, keepdim=True)
    return torch.where(outward < 0, -normals, normals)


def tangent_frame(normals):
    up = torch.tensor([0.0, 0.0, 1.0], device=normals.device).view(1, 1, 3)
    alt = torch.tensor([0.0, 1.0, 0.0], device=normals.device).view(1, 1, 3)
    ref = torch.where((normals * up).sum(-1, keepdim=True).abs() > 0.92, alt, up)
    t1 = F.normalize(torch.cross(ref.expand_as(normals), normals, dim=-1), dim=-1, eps=1e-6)
    t2 = F.normalize(torch.cross(normals, t1, dim=-1), dim=-1, eps=1e-6)
    return t1, t2


def erode_mask(mask, radius):
    if radius <= 0:
        return mask
    x = mask.permute(2, 0, 1)[None]
    y = -F.max_pool2d(-x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[0].permute(1, 2, 0)


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    x = mask.permute(2, 0, 1)[None]
    y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[0].permute(1, 2, 0)


class FurDogLRM(nn.Module):
    def __init__(self, dim=384, n_layers=4, n_heads=6, n_freq=8,
                 dino_name="facebook/dinov2-large", body_offset_max=0.035,
                 body_scale=0.015, fur_scale=0.012, body_k=1, fur_k=1):
        super().__init__()
        from transformers import AutoModel
        self.dino = AutoModel.from_pretrained(dino_name)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        self.body_offset_max = body_offset_max
        self.body_scale = body_scale
        self.fur_scale = fur_scale
        self.body_k = body_k
        self.fur_k = fur_k

        self.img_proj = nn.Linear(self.dino.config.hidden_size, dim)
        pe = 3 + 3 * 2 * n_freq
        self.pt_proj = nn.Sequential(nn.Linear(pe, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerDecoderLayer(dim, n_heads, dim * 4, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.transformer = nn.TransformerDecoder(layer, n_layers)

        self.body_offset = nn.Linear(dim, body_k * 3)
        self.body_scale_head = nn.Linear(dim, body_k * 3)
        self.body_quat = nn.Linear(dim, body_k * 4)
        self.body_opacity = nn.Linear(dim, body_k)
        self.body_rgb = nn.Linear(dim, body_k * 3)

        self.fur_len = nn.Linear(dim, fur_k)
        self.fur_lat = nn.Linear(dim, fur_k * 2)
        self.fur_scale_head = nn.Linear(dim, fur_k * 3)
        self.fur_quat = nn.Linear(dim, fur_k * 4)
        self.fur_opacity = nn.Linear(dim, fur_k)
        self.fur_rgb = nn.Linear(dim, fur_k * 3)

        for h in (self.body_offset, self.body_scale_head, self.body_quat,
                  self.body_opacity, self.body_rgb, self.fur_lat,
                  self.fur_scale_head, self.fur_quat, self.fur_opacity, self.fur_rgb):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        nn.init.zeros_(self.fur_len.weight)
        nn.init.constant_(self.fur_len.bias, -4.0)
        nn.init.constant_(self.fur_opacity.bias, -1.5)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def encode_image(self, img, res=224):
        x = F.interpolate((img - self.mean) / self.std, size=(res, res),
                          mode="bilinear", align_corners=False)
        return self.dino(pixel_values=x).last_hidden_state[:, 1:]

    def forward(self, img, canonical_pts, posed_pts, posed_normals, fur_lmax,
                fur_latmax, fur_floor_frac=None, subdivide=None):
        img_tok = self.img_proj(self.encode_image(img))
        pt = self.pt_proj(fourier_embed(canonical_pts))
        x = self.transformer(pt, img_tok)
        if subdivide is not None:
            x = subdivide(x)
            posed_pts = subdivide(posed_pts)

        B, N, _ = x.shape
        bk, fk = self.body_k, self.fur_k
        q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=x.device)

        body_offset = torch.tanh(self.body_offset(x)).view(B, N, bk, 3) * self.body_offset_max
        body_means = posed_pts[:, :, None, :] + body_offset
        body_scales = torch.exp(self.body_scale_head(x).view(B, N, bk, 3).clamp(-6, 2)) * self.body_scale
        body_quats = F.normalize(self.body_quat(x).view(B, N, bk, 4) + q0, dim=-1)
        body_op = torch.sigmoid(self.body_opacity(x).view(B, N, bk) + 2.0)
        body_rgb = torch.sigmoid(self.body_rgb(x).view(B, N, bk, 3))

        normals = F.normalize(posed_normals, dim=-1, eps=1e-6)
        t1, t2 = tangent_frame(normals)
        if fur_floor_frac is None:
            fur_floor_frac = torch.zeros(B, device=x.device)
        fur_length = (F.softplus(self.fur_len(x)).view(B, N, fk)
                      + fur_floor_frac.view(B, 1, 1)) * fur_lmax.view(B, 1, 1)
        fur_lat = torch.tanh(self.fur_lat(x)).view(B, N, fk, 2) * fur_latmax.view(B, 1, 1, 1)
        fur_delta = (normals[:, :, None, :] * fur_length[..., None]
                     + t1[:, :, None, :] * fur_lat[..., 0:1]
                     + t2[:, :, None, :] * fur_lat[..., 1:2])
        fur_roots = posed_pts[:, :, None, :]
        fur_means = fur_roots + fur_delta
        fur_scales = torch.exp(self.fur_scale_head(x).view(B, N, fk, 3).clamp(-6, 2)) * self.fur_scale
        fur_quats = F.normalize(self.fur_quat(x).view(B, N, fk, 4) + q0, dim=-1)
        fur_op = torch.sigmoid(self.fur_opacity(x).view(B, N, fk))
        fur_rgb = torch.sigmoid(self.fur_rgb(x).view(B, N, fk, 3))

        body = dict(means=body_means.reshape(B, N * bk, 3),
                    scales=body_scales.reshape(B, N * bk, 3),
                    quats=body_quats.reshape(B, N * bk, 4),
                    opacities=body_op.reshape(B, N * bk),
                    rgb=body_rgb.reshape(B, N * bk, 3))
        fur = dict(means=fur_means.reshape(B, N * fk, 3),
                   scales=fur_scales.reshape(B, N * fk, 3),
                   quats=fur_quats.reshape(B, N * fk, 4),
                   opacities=fur_op.reshape(B, N * fk),
                   rgb=fur_rgb.reshape(B, N * fk, 3),
                   roots=fur_roots.expand(-1, -1, fk, -1).reshape(B, N * fk, 3),
                   delta=fur_delta.reshape(B, N * fk, 3),
                   length=fur_length.reshape(B, N * fk),
                   lateral=fur_lat.reshape(B, N * fk, 2))
        full = dict(means=torch.cat([body["means"], fur["means"]], dim=1),
                    scales=torch.cat([body["scales"], fur["scales"]], dim=1),
                    quats=torch.cat([body["quats"], fur["quats"]], dim=1),
                    opacities=torch.cat([body["opacities"], fur["opacities"]], dim=1),
                    rgb=torch.cat([body["rgb"], fur["rgb"]], dim=1))
        return dict(body=body, fur=fur, full=full)


def smooth_length_loss(length, edges, n_verts, fur_k):
    if edges.numel() == 0:
        return length.new_zeros(())
    B = length.shape[0]
    l = length.view(B, n_verts, fur_k)
    e = edges.to(length.device)
    return (l[:, e[:, 0]] - l[:, e[:, 1]]).abs().mean()


def render_branch(gs, view, bg):
    return render_gaussians(gs["means"], gs["quats"], gs["scales"], gs["opacities"],
                            gs["rgb"], view["c2w"], view["K"], view["W"], view["H"], bg=bg)


def save_static_outputs(out_dir, tag, scene, gs):
    prefix = os.path.join(out_dir, f"{tag}_{scene['name']}")
    save_ply(prefix + "_full.ply", gs["full"]["means"], gs["full"]["scales"],
             gs["full"]["quats"], gs["full"]["opacities"], gs["full"]["rgb"])
    save_ply(prefix + "_body.ply", gs["body"]["means"], gs["body"]["scales"],
             gs["body"]["quats"], gs["body"]["opacities"], gs["body"]["rgb"])
    save_ply(prefix + "_fur.ply", gs["fur"]["means"], gs["fur"]["scales"],
             gs["fur"]["quats"], gs["fur"]["opacities"], gs["fur"]["rgb"])


def save_fur_dynamics(out_dir, tag, scene, gs, view, fps=15, frames=48):
    preset = FUR_PRESETS[scene["preset"]]
    device = gs["fur"]["means"].device
    bg = torch.ones(3, device=device)
    wind_dir = F.normalize(torch.tensor([0.75, 0.15, 0.25], device=device), dim=0)
    down = torch.tensor([0.0, 0.0, -1.0], device=device)
    body_diag = scene["body_diag"].to(device)

    x = gs["fur"]["delta"].clone()
    v = torch.zeros_like(x)
    rest = x.clone()
    dt = 1.0 / fps
    stiffness = preset["stiffness"]
    damping = preset["damping"]
    wind_amp = preset["wind"] * 0.08 * body_diag
    grav_amp = preset["gravity"] * body_diag
    out_frames = []

    with torch.no_grad():
        for t in range(frames):
            phase = np.sin(2.0 * np.pi * t / max(frames - 1, 1))
            gust = wind_dir.view(1, 3) * (wind_amp * phase)
            gravity = down.view(1, 3) * grav_amp
            force = stiffness * (rest - x) - damping * v + gust + gravity
            v = v + dt * force
            x = x + dt * v
            fur_means = gs["fur"]["roots"] + x
            rgb, _ = render_gaussians(
                torch.cat([gs["body"]["means"], fur_means], dim=0),
                torch.cat([gs["body"]["quats"], gs["fur"]["quats"]], dim=0),
                torch.cat([gs["body"]["scales"], gs["fur"]["scales"]], dim=0),
                torch.cat([gs["body"]["opacities"], gs["fur"]["opacities"]], dim=0),
                torch.cat([gs["body"]["rgb"], gs["fur"]["rgb"]], dim=0),
                view["c2w"], view["K"], view["W"], view["H"], bg=bg)
            out_frames.append((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    import imageio
    path = os.path.join(out_dir, f"{tag}_{scene['name']}_{scene['preset']}_fur_dyn.mp4")
    imageio.mimsave(path, out_frames, fps=fps, quality=8)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dirs", nargs="+", default=None,
                    help="COLMAP scene dirs. Default: rinda + bear samples if present.")
    ap.add_argument("--root", default=None, help="Optional root containing */colmap scenes.")
    ap.add_argument("--preset", action="append", default=[],
                    help="Override fur preset per scene, e.g. 00208-rinda=short.")
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--views_per_iter", type=int, default=6)
    ap.add_argument("--scale_div", type=int, default=8)
    ap.add_argument("--vis_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=600)
    ap.add_argument("--lpips_weight", type=float, default=0.0)
    ap.add_argument("--lpips_start", type=int, default=300)
    ap.add_argument("--w_body_core", type=float, default=0.5)
    ap.add_argument("--w_body_boundary", type=float, default=0.0,
                    help="Penalize body-only alpha in the boundary band so fur explains long coats.")
    ap.add_argument("--w_fur_residual", type=float, default=0.4)
    ap.add_argument("--w_len", type=float, default=0.015)
    ap.add_argument("--w_smooth", type=float, default=0.08)
    ap.add_argument("--w_lateral", type=float, default=0.03)
    ap.add_argument("--w_fur_opacity", type=float, default=0.01)
    ap.add_argument("--core_erode", type=int, default=4)
    ap.add_argument("--body_dilate", type=int, default=2)
    ap.add_argument("--body_offset_max", type=float, default=0.035)
    ap.add_argument("--fur_len_floor_frac", type=float, default=0.0,
                    help="Minimum fur length as a fraction of per-scene Lmax. Useful for long-coat tests.")
    ap.add_argument("--body_k", type=int, default=1)
    ap.add_argument("--fur_k", type=int, default=1)
    ap.add_argument("--dino_name", default="facebook/dinov2-large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="exps/dog_lrm_fur")
    ap.add_argument("--anim_frames", type=int, default=48)
    ap.add_argument("--anim_fps", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    if args.root:
        scene_dirs = sorted(glob.glob(os.path.join(args.root, "*", "colmap")))
    else:
        scene_dirs = args.scene_dirs or default_scene_dirs()
    if not scene_dirs:
        raise RuntimeError("No scenes found. Pass --scene_dirs or add the default samples.")

    overrides = parse_preset_overrides(args.preset)
    smal = SMALModel(dev)
    faces_sub = subdivided_faces(smal.faces, 1).to(dev)
    edges_sub = unique_edges(faces_sub).to(dev)
    scenes = []
    for sd in scene_dirs:
        name = scene_name(sd)
        scenes.append(load_scene(sd, smal, args.scale_div, dev, overrides.get(name)))
    print("scenes: " + ", ".join(f"{s['name']}({s['preset']})" for s in scenes), flush=True)

    canonical = torch.stack([s["canonical"] for s in scenes])
    posed = torch.stack([s["posed"] for s in scenes])
    posed_sub = smal.subdivide(posed)
    normals_sub = orient_normals_outward(posed_sub, vertex_normals(posed_sub, faces_sub))
    body_diag = torch.stack([s["body_diag"] for s in scenes]).to(dev)
    fur_lmax = torch.tensor([FUR_PRESETS[s["preset"]]["lmax"] for s in scenes],
                            device=dev) * body_diag
    fur_sigma = torch.tensor([FUR_PRESETS[s["preset"]]["sigma"] for s in scenes],
                             device=dev) * body_diag
    fur_latmax = torch.tensor([FUR_PRESETS[s["preset"]]["lat"] for s in scenes],
                              device=dev) * body_diag
    fur_floor_frac = torch.full((len(scenes),), args.fur_len_floor_frac, device=dev)

    model = FurDogLRM(dino_name=args.dino_name, body_offset_max=args.body_offset_max,
                      body_k=args.body_k, fur_k=args.fur_k).to(dev)
    print(f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M",
          flush=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    lpips_fn = None
    if args.lpips_weight > 0:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net="alex").to(dev)
        for p in lpips_fn.parameters():
            p.requires_grad = False

    white = torch.ones(3, device=dev)
    last_gs = None
    for it in range(args.iters):
        ref = [int(np.random.randint(len(s["views"]))) for s in scenes]
        inputs = torch.stack([scenes[b]["inputs_all"][ref[b]] for b in range(len(scenes))])
        gs = model(inputs, canonical, posed, normals_sub, fur_lmax, fur_latmax,
                   fur_floor_frac=fur_floor_frac,
                   subdivide=smal.subdivide)
        last_gs = gs

        loss_rgb = loss_mask = loss_body = loss_body_boundary = loss_fur = loss_perc = 0.0
        n = 0
        for b, scene in enumerate(scenes):
            choices = [j for j in range(len(scene["views"])) if j != ref[b]]
            if args.views_per_iter > 0 and len(choices) > args.views_per_iter:
                choices = list(np.random.choice(choices, args.views_per_iter, replace=False))
            for j in choices:
                v = scene["views"][j]
                rgb_full, alpha_full = render_branch({k: gs["full"][k][b] for k in
                                                      ("means", "quats", "scales", "opacities", "rgb")},
                                                     v, white)
                rgb_body, alpha_body = render_branch({k: gs["body"][k][b] for k in
                                                      ("means", "quats", "scales", "opacities", "rgb")},
                                                     v, white)
                mask = v["mask"]
                loss_rgb = loss_rgb + F.l1_loss(rgb_full * mask, v["rgb"] * mask)
                loss_mask = loss_mask + F.l1_loss(alpha_full, mask)

                core = erode_mask(mask, args.core_erode)
                boundary = (mask - core).clamp_min(0.0)
                loss_body = loss_body + F.l1_loss(rgb_body * core, v["rgb"] * core)
                loss_body = loss_body + F.l1_loss(alpha_body * core, mask * core)
                loss_body_boundary = loss_body_boundary + (alpha_body * boundary).mean()

                fur_target = (mask - dilate_mask(alpha_body.detach(), args.body_dilate)).clamp_min(0.0)
                _, alpha_fur = render_branch({k: gs["fur"][k][b] for k in
                                              ("means", "quats", "scales", "opacities", "rgb")},
                                             v, None)
                loss_fur = loss_fur + F.l1_loss(alpha_fur * boundary, fur_target * boundary)

                if lpips_fn is not None and it >= args.lpips_start:
                    gt_w = v["rgb"] * mask + (1 - mask) * white
                    r = F.interpolate(rgb_full.permute(2, 0, 1)[None] * 2 - 1, 256,
                                      mode="bilinear", align_corners=False)
                    g = F.interpolate(gt_w.permute(2, 0, 1)[None] * 2 - 1, 256,
                                      mode="bilinear", align_corners=False)
                    loss_perc = loss_perc + lpips_fn(r, g).mean()
                n += 1

        len_prior = (gs["fur"]["length"] / fur_sigma.view(-1, 1).clamp_min(1e-6)).mean()
        lat_prior = (gs["fur"]["lateral"].abs() / fur_latmax.view(-1, 1, 1).clamp_min(1e-6)).mean()
        smooth = smooth_length_loss(gs["fur"]["length"], edges_sub, posed_sub.shape[1], args.fur_k)
        op_budget = gs["fur"]["opacities"].mean()
        loss = ((loss_rgb + loss_mask + args.w_body_core * loss_body
                 + args.w_body_boundary * loss_body_boundary
                 + args.w_fur_residual * loss_fur
                 + args.lpips_weight * loss_perc) / max(n, 1)
                + args.w_len * len_prior
                + args.w_smooth * smooth
                + args.w_lateral * lat_prior
                + args.w_fur_opacity * op_budget)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if it % 50 == 0:
            lp = float(loss_perc) / max(n, 1) if lpips_fn is not None and it >= args.lpips_start else 0.0
            print(f"it{it:5d} loss={float(loss):.4f} rgb={float(loss_rgb)/max(n,1):.4f} "
                  f"mask={float(loss_mask)/max(n,1):.4f} body={float(loss_body)/max(n,1):.4f} "
                  f"bodybd={float(loss_body_boundary)/max(n,1):.4f} "
                  f"fur={float(loss_fur)/max(n,1):.4f} len={float(len_prior):.4f} "
                  f"smooth={float(smooth):.4f} lpips={lp:.4f}", flush=True)

        if it % args.vis_every == 0 or it == args.iters - 1:
            with torch.no_grad():
                tiles = []
                for b, scene in enumerate(scenes):
                    v = scene["views"][len(scene["views"]) // 2]
                    full_b = {k: gs["full"][k][b] for k in ("means", "quats", "scales", "opacities", "rgb")}
                    rgb, _ = render_branch(full_b, v, white)
                    pair = np.concatenate([(rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                                           (v["rgb"].cpu().numpy() * 255).astype(np.uint8)], axis=1)
                    tiles.append(pair)
                hmin = min(t.shape[0] for t in tiles)
                Image.fromarray(np.concatenate([t[:hmin] for t in tiles], axis=1)).save(
                    os.path.join(args.out, f"it{it:05d}.png"))

        if args.save_every and it > 0 and it % args.save_every == 0:
            torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("dino.")},
                       os.path.join(args.out, "model.pt"))

    ref = [len(s["views"]) // 2 for s in scenes]
    inputs = torch.stack([scenes[b]["inputs_all"][ref[b]] for b in range(len(scenes))])
    model.eval()
    with torch.no_grad():
        last_gs = model(inputs, canonical, posed, normals_sub, fur_lmax, fur_latmax,
                        fur_floor_frac=fur_floor_frac,
                        subdivide=smal.subdivide)

    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("dino.")},
               os.path.join(args.out, "model.pt"))
    for b, scene in enumerate(scenes):
        gs_b = {
            branch: {k: v[b].detach() for k, v in last_gs[branch].items()}
            for branch in ("body", "fur", "full")
        }
        save_static_outputs(args.out, "final", scene, gs_b)
        view = scene["views"][len(scene["views"]) // 2]
        mp4 = save_fur_dynamics(args.out, "final", scene, gs_b, view,
                                fps=args.anim_fps, frames=args.anim_frames)
        print(f"saved {scene['name']} static ply + fur dynamics: {mp4}", flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
