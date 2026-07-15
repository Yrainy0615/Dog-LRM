#!/usr/bin/env python3
"""Single-case STRAND fur trainer (builds on the de-fur 'coat' design).

Fixes "moves but doesn't look like fur" by replacing the isotropic fur blobs
with actual strands:

  * each subdivided SMAL vertex grows one strand = a polyline of K Gaussians,
  * growth direction = surface normal (to climb out of the de-furred shell)
    + a learned tangential flow + a gravity droop  -> coherent fur direction,
  * each Gaussian is anisotropic: elongated along the strand, thin across,
  * opacity ramps opaque(root) -> soft(tip); color tied to the body albedo.

Body is the de-furred (inset) opaque inner core, same as train_fur_coat.py.
Dynamics reuse render_fur_dynamics.sway_dynamics: each strand point sways by
amp_frac * |offset-from-root|, so roots stay put and tips lead (cantilever).
Defaults to the long-hair bear.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from dog_lrm.model import fourier_embed
from dog_lrm.render import render_gaussians, save_ply
from dog_lrm.smal_model import SMALModel
import train_dog_lrm_fur as T
from render_fur_dynamics import sway_dynamics

DEFAULT_BEAR = "received_data_from_Pinstudio_20260424/unzipped/0423/00062-bear/colmap"


def quat_align_z(d):
    """Unit quaternion (wxyz) rotating local +z onto unit direction d [...,3].
    Half-vector construction (no acos -> stable gradients even when d || z)."""
    w = 1.0 + d[..., 2:3]                                     # 1 + cos = 1 + d.z
    xyz = torch.stack([-d[..., 1], d[..., 0], torch.zeros_like(d[..., 0])], dim=-1)  # cross(z,d)
    q = F.normalize(torch.cat([w, xyz], dim=-1), dim=-1, eps=1e-8)
    near = (w < 1e-6)                                         # d ~ -z (anti-parallel)
    q_flip = torch.zeros_like(q); q_flip[..., 1] = 1.0       # 180 about x
    return torch.where(near, q_flip, q)


class StrandFurLRM(nn.Module):
    def __init__(self, dino_name="facebook/dinov2-large", body_offset_max=0.035,
                 body_scale=0.015, n_pts=6, normal_weight=1.0, dir_strength=0.8,
                 droop=0.35, curl=0.5, radius_frac=0.004, elong=0.7,
                 strands_per_vertex=10, jitter_frac=0.012, body_op_bias=-1.0,
                 fur_op_floor=0.1, fur_op_fixed=0.0):
        super().__init__()
        self.bb = T.FurDogLRM(dino_name=dino_name, body_offset_max=body_offset_max,
                              body_scale=body_scale)
        dim = self.bb.img_proj.out_features
        self.n_pts = n_pts
        self.normal_weight = normal_weight
        self.dir_strength = dir_strength
        self.droop = droop
        self.curl = curl
        self.radius_frac = radius_frac
        self.elong = elong
        self.strands_per_vertex = strands_per_vertex
        self.jitter_frac = jitter_frac
        self.body_op_bias = body_op_bias
        self.fur_op_floor = fur_op_floor
        self.fur_op_fixed = fur_op_fixed
        # sunflower disk: S strands per vertex jittered in the tangent plane to form a dense coat
        i = torch.arange(strands_per_vertex).float()
        r = torch.sqrt((i + 0.5) / strands_per_vertex)
        th = i * 2.399963229728653                                   # golden angle
        self.register_buffer("jitter", torch.stack([r * torch.cos(th), r * torch.sin(th)], -1))  # [S,2]
        g = torch.Generator().manual_seed(7)
        # fixed per-strand brightness variation: gives visible strand texture that can't drift
        self.register_buffer("strand_tone", 1.0 + 0.18 * (2 * torch.rand(strands_per_vertex, generator=g) - 1))

        self.s_len = nn.Linear(dim, 1)
        self.s_dir = nn.Linear(dim, 2)        # tangential flow (t1, t2)
        self.s_op = nn.Linear(dim, 1)
        self.s_rad = nn.Linear(dim, 1)
        self.s_shade = nn.Linear(dim, 1)      # root->tip shading on the body albedo
        for h in (self.s_dir, self.s_rad, self.s_shade):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        nn.init.zeros_(self.s_len.weight); nn.init.constant_(self.s_len.bias, -2.0)  # start near the length floor/prior
        nn.init.zeros_(self.s_op.weight); nn.init.constant_(self.s_op.bias, 2.0)  # opaque start

    @property
    def dino(self):
        return self.bb.dino

    def feats(self, img, canonical_pts, subdivide):
        img_tok = self.bb.img_proj(self.bb.encode_image(img))
        pt = self.bb.pt_proj(fourier_embed(canonical_pts))
        x = self.bb.transformer(pt, img_tok)
        return subdivide(x)

    def body_branch(self, x, posed_pts):
        bb = self.bb
        B, N, _ = x.shape
        q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=x.device)
        offset = torch.tanh(bb.body_offset(x)).view(B, N, 3) * bb.body_offset_max
        means = posed_pts + offset
        scales = torch.exp(bb.body_scale_head(x).view(B, N, 3).clamp(-6, 2)) * bb.body_scale
        quats = F.normalize(bb.body_quat(x).view(B, N, 4) + q0, dim=-1)
        bias = getattr(self, "body_bias_field", None)        # per-vertex [1,N]: opaque face, hidden coat
        bias = self.body_op_bias if bias is None else bias
        op = torch.sigmoid(bb.body_opacity(x).view(B, N) + bias)
        rgb = torch.sigmoid(bb.body_rgb(x).view(B, N, 3))
        return dict(means=means, scales=scales, quats=quats, opacities=op, rgb=rgb)

    def forward(self, img, canonical_pts, posed_pts, posed_normals, fur_lmax,
                body_diag, subdivide, fur_floor_frac=0.5):
        x = self.feats(img, canonical_pts, subdivide)
        posed_pts = subdivide(posed_pts)
        B, N, _ = x.shape
        K = self.n_pts
        body = self.body_branch(x, posed_pts)
        ar = getattr(self, "albedo_res", None)   # optional free per-vertex albedo (test-time opt)
        if ar is not None:
            body["rgb"] = (body["rgb"] + ar).clamp(0, 1)
        body_rgb = body["rgb"]                                       # shared coat albedo

        normals = F.normalize(posed_normals, dim=-1, eps=1e-6)
        t1, t2 = T.tangent_frame(normals)
        flow = torch.tanh(self.s_dir(x)) * self.dir_strength         # [B,N,2]
        g = torch.tensor([0.0, 0.0, -1.0], device=x.device).view(1, 1, 3)
        gt = g - (g * normals).sum(-1, keepdim=True) * normals       # gravity in tangent plane
        gt = F.normalize(gt, dim=-1, eps=1e-6)
        direction = F.normalize(self.normal_weight * normals
                                + flow[..., 0:1] * t1 + flow[..., 1:2] * t2
                                + self.droop * gt, dim=-1)           # [B,N,3] growth dir

        # clamp softplus so a gradient spike can't blow up strand length -> giant Gaussians / OOM
        # fur_lmax: scalar per batch [B] or per-vertex field [B,N] (VLM body-part prior)
        lm = fur_lmax.view(B, 1, 1) if fur_lmax.numel() == B else fur_lmax.view(B, N, 1)
        length = (F.softplus(self.s_len(x)).clamp(max=3.0).view(B, N, 1) + fur_floor_frac) * lm
        S, M = self.strands_per_vertex, N * self.strands_per_vertex * K
        jit = self.jitter.to(x.device)                               # [S,2]
        jr = self.jitter_frac * body_diag.view(B, 1, 1, 1)
        if getattr(self, "root_follow_offset", False):
            # root on the corrected body surface (absorbs SMAL misfit), not raw SMAL verts
            posed_pts = posed_pts + torch.tanh(self.bb.body_offset(x)).view(B, N, 3) * self.bb.body_offset_max
        # S jittered roots per vertex (tangent-plane sunflower) -> a dense coat
        roots_s = (posed_pts[:, :, None, :]
                   + jr * (jit[None, None, :, 0:1] * t1[:, :, None, :]
                           + jit[None, None, :, 1:2] * t2[:, :, None, :]))             # [B,N,S,3]
        ts = torch.linspace(0, 1, K, device=x.device).view(1, 1, 1, K, 1)
        dirS = direction[:, :, None, None, :]
        lenS = length[:, :, None, None, :]
        gtS = gt[:, :, None, None, :]
        pts = (roots_s[:, :, :, None, :] + dirS * (ts * lenS)
               + gtS * (self.curl * ts ** 2 * lenS))                                   # [B,N,S,K,3]

        seg = F.normalize(pts[..., 1:, :] - pts[..., :-1, :], dim=-1, eps=1e-6)
        tangent = torch.cat([seg[..., :1, :], seg], dim=3)                             # [B,N,S,K,3]
        quats = quat_align_z(tangent).reshape(B, M, 4)
        seg_len = (length / max(K - 1, 1)).clamp_min(1e-5)                             # [B,N,1]
        rad = (torch.exp(self.s_rad(x).view(B, N, 1).clamp(-0.7, 0.7)) * self.radius_frac
               * body_diag.view(B, 1, 1))                # [B,N,1]; ±0.7 caps radius at 2x (tile-memory safety)
        rad_e = rad.view(B, N, 1, 1, 1).expand(B, N, S, K, 1)
        long_e = (seg_len * self.elong).view(B, N, 1, 1, 1).expand(B, N, S, K, 1)
        scales = torch.cat([rad_e, rad_e, long_e], dim=-1).reshape(B, M, 3)            # thin,thin,long

        if self.fur_op_fixed > 0:
            # fur is the only appearance layer -> opacity needn't be learned; fixing it
            # removes the collapse direction entirely
            op = torch.full((B, N, 1), self.fur_op_fixed, device=x.device)
        else:
            # opacity floor: fur can never go fully transparent -> can't collapse to "redundant"
            op = self.fur_op_floor + (1.0 - self.fur_op_floor) * torch.sigmoid(self.s_op(x).view(B, N, 1))
        op_ramp = (1.0 - 0.4 * ts.view(1, 1, 1, K))
        op_k = (op.view(B, N, 1, 1) * op_ramp).expand(B, N, S, K).reshape(B, M)
        shade = torch.tanh(self.s_shade(x).view(B, N, 1)) * 0.1
        tone = self.strand_tone.to(x.device).view(1, 1, S, 1, 1)
        rgb_e = (body_rgb.view(B, N, 1, 1, 3) * tone
                 * (1.0 + shade.view(B, N, 1, 1, 1) * (ts - 0.5))).clamp(0, 1)
        rgb_k = rgb_e.expand(B, N, S, K, 3).reshape(B, M, 3)

        roots_out = roots_s[:, :, :, None, :].expand(B, N, S, K, 3).reshape(B, M, 3)
        means = pts.reshape(B, M, 3)
        fur = dict(means=means, quats=quats, scales=scales, opacities=op_k, rgb=rgb_k,
                   roots=roots_out, delta=means - roots_out,
                   tangent=tangent.reshape(B, M, 3),
                   length=length.view(B, N), direction=direction)
        return dict(body=body, fur=fur)


def gabor_orientation(gray, n_theta=18, ksize=15, sig_a=4.0, sig_b=1.6, lam=5.0):
    """gray [H,W] in [0,1] -> (cos2,sin2,conf) [H,W]. phi=strand direction (Gabor long
    axis along the strand, oscillation across it; argmax_phi response = strand dir)."""
    img = gray[None, None]
    ax = torch.arange(ksize, device=gray.device) - ksize // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    thetas = torch.linspace(0, np.pi, n_theta + 1, device=gray.device)[:-1]
    resp = []
    for ph in thetas:
        a = xx * torch.cos(ph) + yy * torch.sin(ph)
        b = -xx * torch.sin(ph) + yy * torch.cos(ph)
        gb = torch.exp(-(a ** 2 / (2 * sig_a ** 2) + b ** 2 / (2 * sig_b ** 2))) * torch.cos(2 * np.pi * b / lam)
        gb = gb - gb.mean()
        resp.append(F.conv2d(img, gb[None, None], padding=ksize // 2)[0, 0].abs())
    w = torch.stack(resp)                                        # [nθ,H,W]
    cos2 = (w * torch.cos(2 * thetas).view(-1, 1, 1)).sum(0)
    sin2 = (w * torch.sin(2 * thetas).view(-1, 1, 1)).sum(0)
    mag = torch.sqrt(cos2 ** 2 + sin2 ** 2)
    conf = mag / (w.sum(0) + 1e-6)
    return cos2 / (mag + 1e-6), sin2 / (mag + 1e-6), conf


def project_dir(points, tangents, c2w, K, eps=1e-3):
    """Screen-space 2D direction of each 3D tangent at each point. Returns d2 [M,2] (pixels)."""
    w2c = torch.inverse(c2w)
    R, tr = w2c[:3, :3], w2c[:3, 3]
    def proj(p):
        pc = p @ R.T + tr
        uv = pc[:, :2] / pc[:, 2:3].clamp_min(1e-6)
        return torch.stack([K[0, 0] * uv[:, 0] + K[0, 2], K[1, 1] * uv[:, 1] + K[1, 2]], -1)
    return proj(points + eps * tangents) - proj(points)          # [M,2]


def orientation_loss(fur, view, gab, white_dummy=None):
    """Render strand orientation (double-angle, reliability-weighted) and compare to the
    cached Gabor orientation of the GT view. gab = (cos2,sin2,conf,K,W,H,mask)."""
    gc, gs2, conf, Ko, Wo, Ho, m = gab
    d2 = project_dir(fur["means"], fur["tangent"], view["c2w"], Ko)
    dx, dy = d2[:, 0:1], d2[:, 1:2]
    r2 = (dx * dx + dy * dy).clamp_min(1e-4)                      # clamp denom -> bounded grad
    rel = r2.sqrt().clamp_max(50.0)                              # reliability = projected length
    cos2 = (dx * dx - dy * dy) / r2                              # double-angle, no atan2
    sin2 = (2 * dx * dy) / r2
    color = torch.cat([rel * cos2, rel * sin2, rel], dim=-1)     # [M,3]
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=fur["means"].device).expand(fur["means"].shape[0], 4)
    img, _ = render_gaussians(fur["means"], fur["quats"], fur["scales"], fur["opacities"],
                              color, view["c2w"], Ko, Wo, Ho, bg=None)            # [Ho,Wo,3]
    c0, c1, cw = img[..., 0], img[..., 1], img[..., 2].clamp_min(1e-4)
    align = (c0 * gc + c1 * gs2) / cw                            # [Ho,Wo], normalized by reliability
    wgt = (m * conf)
    return (wgt * (1.0 - align)).sum() / wgt.sum().clamp_min(1e-6)


def measure_fur_length(views, posed_sub, normals_sub, inset, lmax_cap, device):
    """Geometric fur-length supervision: at silhouette-grazing vertices, march from the
    projected bald root outward (along the projected normal) until the GT mask ends; the
    march distance (converted to world units) is the measured fur depth. Median across
    views. Returns (target_len [N], valid [N]) — independent of any pixel loss."""
    V = posed_sub[0]                                                # [N,3]
    Nrm = normals_sub[0]
    roots = V - inset[0] * Nrm                                      # bald roots
    per_view = []
    n_steps = 64
    for v in views:
        c2w = v["c2w"]; K = v["K"]
        w2c = torch.inverse(c2w)
        cam = c2w[:3, 3]
        pc = roots @ w2c[:3, :3].T + w2c[:3, 3]                     # cam coords of roots
        z = pc[:, 2].clamp_min(1e-6)
        uv = torch.stack([K[0, 0] * pc[:, 0] / z + K[0, 2],
                          K[1, 1] * pc[:, 1] / z + K[1, 2]], -1)    # [N,2] px
        pc2 = (roots + 0.01 * Nrm) @ w2c[:3, :3].T + w2c[:3, 3]
        z2 = pc2[:, 2].clamp_min(1e-6)
        uv2 = torch.stack([K[0, 0] * pc2[:, 0] / z2 + K[0, 2],
                           K[1, 1] * pc2[:, 1] / z2 + K[1, 2]], -1)
        d2 = F.normalize(uv2 - uv, dim=-1, eps=1e-6)                # outward 2D dir
        vdir = F.normalize(V - cam, dim=-1)
        grazing = ((Nrm * vdir).sum(-1).abs() < 0.35) & (pc[:, 2] > 0)
        m = v["mask"][..., 0]                                       # [H,W]
        H, W = m.shape
        tmax_px = (lmax_cap * 1.5) * K[0, 0] / z                    # cap march at 1.5x lmax
        ts = torch.linspace(0, 1, n_steps, device=device).view(1, -1)
        pts = uv[:, None, :] + (ts * tmax_px[:, None])[..., None] * d2[:, None, :]  # [N,S,2]
        gx = (pts[..., 0] / (W - 1) * 2 - 1).clamp(-1, 1)
        gy = (pts[..., 1] / (H - 1) * 2 - 1).clamp(-1, 1)
        grid = torch.stack([gx, gy], -1)[None]                      # [1,N,S,2]
        samp = F.grid_sample(m[None, None], grid, align_corners=True)[0, 0]  # [N,S]
        inside = samp > 0.5
        root_in = inside[:, 0]
        exit_idx = (inside.float().cumprod(dim=1).sum(dim=1) - 1).clamp_min(0)  # last consecutive-in step
        meas_px = exit_idx / (n_steps - 1) * tmax_px
        meas_world = meas_px * z / K[0, 0]
        valid = grazing & root_in & (exit_idx < n_steps - 1)        # discard never-exited (occlusion)
        out = torch.full((V.shape[0],), float("nan"), device=device)
        out[valid] = meas_world[valid]
        per_view.append(out)
    stack = torch.stack(per_view)                                   # [views,N]
    cnt = (~stack.isnan()).sum(0)
    # occlusion only ever inflates the march distance -> min over views is the robust estimate
    target = stack.nan_to_num(float("inf")).min(0).values
    valid = (cnt >= 3) & torch.isfinite(target)
    return target, valid


def composite(body, fur, b):
    return dict(
        means=torch.cat([body["means"][b], fur["means"][b]], 0),
        quats=torch.cat([body["quats"][b], fur["quats"][b]], 0),
        scales=torch.cat([body["scales"][b], fur["scales"][b]], 0),
        opacities=torch.cat([body["opacities"][b], fur["opacities"][b]], 0),
        rgb=torch.cat([body["rgb"][b], fur["rgb"][b]], 0))


def render(full, view, bg):
    return render_gaussians(full["means"], full["quats"], full["scales"], full["opacities"],
                            full["rgb"], view["c2w"], view["K"], view["W"], view["H"], bg=bg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", default=DEFAULT_BEAR)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_final", type=float, default=0.0, help="if >0, cosine-decay lr to this value")
    ap.add_argument("--save_every", type=int, default=0, help="if >0, save model.pt every N iters")
    ap.add_argument("--views_per_iter", type=int, default=8)
    ap.add_argument("--scale_div", type=int, default=8)
    ap.add_argument("--vis_every", type=int, default=200)
    ap.add_argument("--n_pts", type=int, default=6)
    ap.add_argument("--fur_lmax_mult", type=float, default=4.0,
                    help="max fur length = inset * this; >1 opens per-vertex length to grow past the shell")
    ap.add_argument("--radius_frac", type=float, default=0.008,
                    help="strand radius as fraction of body_diag; bigger = denser coat (fur covers body)")
    ap.add_argument("--strands_per_vertex", type=int, default=10, help="jittered strands per vertex (density)")
    ap.add_argument("--jitter_frac", type=float, default=0.012, help="root jitter radius as fraction of body_diag")
    ap.add_argument("--body_op_bias", type=float, default=-1.0, help="body opacity logit bias (low = faint occluder)")
    ap.add_argument("--fur_op_floor", type=float, default=0.1, help="min fur opacity (prevents collapse)")
    ap.add_argument("--fur_op_fixed", type=float, default=0.0, help="if >0, fur opacity is this constant (not learned)")
    ap.add_argument("--free_albedo", action="store_true",
                    help="add a free per-vertex albedo residual (test-time opt; upper-bound diagnostic)")
    ap.add_argument("--root_follow_offset", action="store_true",
                    help="fur roots follow the learned body offset (absorbs SMAL misfit)")
    ap.add_argument("--body_offset_max", type=float, default=0.035)
    ap.add_argument("--face_mode", action="store_true",
                    help="suppress strands on skull/muzzle and make the body visible there (crisp face)")
    ap.add_argument("--w_face_crop", type=float, default=5.0, help="extra L1 weight on the projected face box")
    ap.add_argument("--w_len_geo", type=float, default=0.0,
                    help="geometric length supervision: L1 to silhouette-marched fur depth")
    ap.add_argument("--body_inset", type=float, default=0.10)
    ap.add_argument("--vlm_prior", default=None,
                    help="json with joint_lengths_cm + dog_bbox_diag_cm; makes inset/lmax per-vertex fields")
    ap.add_argument("--fur_len_floor_frac", type=float, default=0.3,
                    help="min length floor (× lmax). Keep floor*lmax_mult ≳ 1 so fur still covers the bald shell.")
    ap.add_argument("--normal_weight", type=float, default=1.0)
    ap.add_argument("--dir_strength", type=float, default=0.8)
    ap.add_argument("--droop", type=float, default=0.35)
    ap.add_argument("--curl", type=float, default=0.5)
    ap.add_argument("--w_len", type=float, default=0.01)
    ap.add_argument("--w_smooth", type=float, default=0.08)
    ap.add_argument("--w_dir_smooth", type=float, default=0.05, help="growth-direction smoothness on the mesh")
    ap.add_argument("--w_rgb_smooth", type=float, default=0.0,
                    help="coat albedo smoothness on mesh edges (kills drifting colors of occluded strands)")
    ap.add_argument("--lpips_weight", type=float, default=0.0)
    ap.add_argument("--lpips_start", type=int, default=300)
    ap.add_argument("--w_outside", type=float, default=0.0, help="penalize fur alpha outside the silhouette (anti-wispy)")
    ap.add_argument("--w_orient", type=float, default=0.3, help="Gabor orientation loss weight")
    ap.add_argument("--orient_res", type=int, default=320, help="long-side res for the orientation term")
    ap.add_argument("--orient_views", type=int, default=2, help="views per iter that get the orientation loss")
    ap.add_argument("--dino_name", default="facebook/dinov2-large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="exps/dog_lrm_fur_strand")
    ap.add_argument("--anim_frames", type=int, default=60)
    ap.add_argument("--anim_fps", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    smal = SMALModel(dev)
    faces_sub = T.subdivided_faces(smal.faces, 1).to(dev)
    edges_sub = T.unique_edges(faces_sub).to(dev)
    scene = T.load_scene(args.scene_dir, smal, args.scale_div, dev, args.preset)
    print(f"scene: {scene['name']} ({scene['preset']})", flush=True)

    # cache per-view Gabor orientation (cos2,sin2,conf) + scaled intrinsics at a low res
    gab_cache = []
    if args.w_orient > 0:
        for v in scene["views"]:
            s = args.orient_res / max(v["W"], v["H"])
            Wo, Ho = max(int(round(v["W"] * s)), 1), max(int(round(v["H"] * s)), 1)
            gray = F.interpolate(v["rgb"].mean(-1)[None, None], size=(Ho, Wo),
                                 mode="bilinear", align_corners=False)[0, 0]
            mo = F.interpolate(v["mask"].permute(2, 0, 1)[None], size=(Ho, Wo),
                               mode="bilinear", align_corners=False)[0, 0]
            gc, gs2, conf = gabor_orientation(gray)
            Ko = v["K"].clone(); Ko[0] *= s; Ko[1] *= s
            gab_cache.append((gc, gs2, conf * (mo > 0.5), Ko, Wo, Ho, (mo > 0.5).float()))
        print(f"cached Gabor orientation for {len(gab_cache)} views @ ~{args.orient_res}px", flush=True)

    canonical = scene["canonical"][None]
    posed = scene["posed"][None]
    posed_sub = smal.subdivide(posed)
    normals_sub = T.orient_normals_outward(posed_sub, T.vertex_normals(posed_sub, faces_sub))
    body_diag = scene["body_diag"].to(dev).view(1)
    if args.vlm_prior:
        import json as _json
        import scipy.sparse as _sp
        import pickle as _pkl
        prior = _json.load(open(args.vlm_prior))
        d = _pkl.load(open("third_party/barc_release/data/smal_data/my_smpl_SMBLD_nbj_v3.pkl", "rb"),
                      encoding="latin1")
        Wsk = d["weights"]
        Wsk = Wsk.toarray() if _sp.issparse(Wsk) else np.asarray(Wsk)        # [3889,35]
        Lj = np.full(Wsk.shape[1], float(prior.get("default_cm", 4.0)))
        for j, cm in prior["joint_lengths_cm"].items():
            Lj[int(j)] = float(cm)
        Lv_cm = torch.tensor(Wsk @ Lj, dtype=torch.float32, device=dev)      # smooth per-vertex cm
        units_per_cm = (body_diag / float(prior["dog_bbox_diag_cm"])).item() # scale alignment
        Lv = smal.subdivide(Lv_cm.view(1, -1, 1))[0, :, 0] * units_per_cm    # [Nsub] scene units
        inset = Lv.view(1, -1, 1)                                            # per-vertex de-fur depth
        fur_lmax = (Lv * args.fur_lmax_mult).view(1, -1)                     # per-vertex max length
        print(f"VLM prior: units/cm={units_per_cm:.4f} len_cm range "
              f"[{float(Lv_cm.min()):.1f},{float(Lv_cm.max()):.1f}] -> units "
              f"[{float(Lv.min()):.3f},{float(Lv.max()):.3f}]", flush=True)
    else:
        inset = args.body_inset * body_diag
        fur_lmax = (inset * args.fur_lmax_mult).view(1)

    w_face = None
    if args.face_mode:
        assert args.vlm_prior, "--face_mode requires --vlm_prior (reuses the skinning weights)"
        # skull(16)+muzzle(32) weights -> strand suppression + opaque body there (ears 33/34 keep fur)
        wf = torch.tensor(Wsk[:, 16] + Wsk[:, 32], dtype=torch.float32, device=dev).clamp(0, 1)
        w_face = smal.subdivide(wf.view(1, -1, 1))[0, :, 0]              # [Nsub]
        suppress = (1.0 - 0.9 * w_face).clamp_min(0.1)
        fur_lmax = fur_lmax * suppress.view(1, -1)
        inset = inset * suppress.view(1, 1, -1).transpose(1, 2)          # face barely de-furred
        print(f"face_mode: {int((w_face>0.5).sum())} face verts suppressed", flush=True)

    len_geo_target = len_geo_valid = None
    if args.w_len_geo > 0:
        len_geo_target, len_geo_valid = measure_fur_length(
            scene["views"], posed_sub, normals_sub, inset, float(fur_lmax.max()), dev)
        cover = float(len_geo_valid.float().mean())
        print(f"len_geo: measured {int(len_geo_valid.sum())}/{len_geo_valid.numel()} verts "
              f"({cover*100:.0f}%), median={float(len_geo_target[len_geo_valid].median()):.4f}", flush=True)

    face_boxes = None
    if args.face_mode:
        face_boxes = []
        fv = posed_sub[0][w_face > 0.5]                                  # face verts world
        for v in scene["views"]:
            w2c = torch.inverse(v["c2w"]); K = v["K"]
            pc = fv @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2].clamp_min(1e-6)
            u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).clamp(0, v["W"] - 1)
            vv = (K[1, 1] * pc[:, 1] / z + K[1, 2]).clamp(0, v["H"] - 1)
            pad = 8
            face_boxes.append((int(vv.min()) - pad, int(vv.max()) + pad,
                               int(u.min()) - pad, int(u.max()) + pad))

    fur_sigma = torch.tensor([T.FUR_PRESETS[scene["preset"]]["sigma"]], device=dev) * body_diag
    shift = inset * normals_sub                                  # [1, N, 3] per vertex
    ppv = args.strands_per_vertex * args.n_pts                   # points per vertex (S*K)
    fur_shift = shift[:, :, None, :].expand(-1, -1, ppv, -1).reshape(
        1, normals_sub.shape[1] * ppv, 3)                        # [1, N*S*K, 3] per strand point
    floor = args.fur_len_floor_frac

    model = StrandFurLRM(dino_name=args.dino_name, n_pts=args.n_pts,
                         normal_weight=args.normal_weight, dir_strength=args.dir_strength,
                         droop=args.droop, curl=args.curl, radius_frac=args.radius_frac,
                         strands_per_vertex=args.strands_per_vertex, jitter_frac=args.jitter_frac,
                         body_op_bias=args.body_op_bias, fur_op_floor=args.fur_op_floor,
                         fur_op_fixed=args.fur_op_fixed,
                         body_offset_max=args.body_offset_max).to(dev)
    print(f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M "
          f"| fur gaussians: {posed_sub.shape[1]*args.strands_per_vertex*args.n_pts}", flush=True)
    if args.free_albedo:
        model.albedo_res = nn.Parameter(torch.zeros(1, posed_sub.shape[1], 3, device=dev))
    model.root_follow_offset = args.root_follow_offset
    if w_face is not None:
        # opaque body on the face (crisp eyes/nose), hidden under the coat elsewhere
        model.body_bias_field = (args.body_op_bias * (1 - w_face) + 3.0 * w_face).view(1, -1)

    lpips_fn = None
    if args.lpips_weight > 0:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net="alex").to(dev)
        for p in lpips_fn.parameters():
            p.requires_grad = False

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = None
    if args.lr_final > 0:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.lr_final)
    e = edges_sub

    def run(inp):
        gs = model(inp, canonical, posed, normals_sub, fur_lmax, body_diag,
                   subdivide=smal.subdivide, fur_floor_frac=floor)
        gs["body"]["means"] = gs["body"]["means"] - shift
        gs["fur"]["means"] = gs["fur"]["means"] - fur_shift
        gs["fur"]["roots"] = gs["fur"]["roots"] - fur_shift
        return gs

    white = torch.ones(3, device=dev)
    for it in range(args.iters):
        ref = int(np.random.randint(len(scene["views"])))
        inputs = scene["inputs_all"][ref][None]
        gs = run(inputs)
        choices = [j for j in range(len(scene["views"])) if j != ref]
        if args.views_per_iter > 0 and len(choices) > args.views_per_iter:
            choices = list(np.random.choice(choices, args.views_per_iter, replace=False))
        loss_rgb = loss_mask = loss_out = loss_perc = loss_face = 0.0
        for j in choices:
            v = scene["views"][j]
            rgb, alpha = render(composite(gs["body"], gs["fur"], 0), v, white)
            m = v["mask"]
            # normalize by mask area so the empty background doesn't dilute appearance gradients
            loss_rgb = loss_rgb + ((rgb - v["rgb"]).abs() * m).sum() / (3.0 * m.sum().clamp_min(1.0))
            loss_mask = loss_mask + F.l1_loss(alpha, m)
            loss_out = loss_out + (alpha * (1.0 - m)).mean()   # anti-overshoot: no fur past the silhouette
            if face_boxes is not None:
                y0, y1, x0, x1 = face_boxes[j]
                y0 = max(y0, 0); x0 = max(x0, 0)
                if y1 > y0 + 4 and x1 > x0 + 4:
                    loss_face = loss_face + F.l1_loss(rgb[y0:y1, x0:x1] * m[y0:y1, x0:x1],
                                                      v["rgb"][y0:y1, x0:x1] * m[y0:y1, x0:x1])
            if lpips_fn is not None and it >= args.lpips_start:
                gt_w = v["rgb"] * m + (1 - m)
                r = F.interpolate(rgb.permute(2, 0, 1)[None] * 2 - 1, 256,
                                  mode="bilinear", align_corners=False)
                g2 = F.interpolate(gt_w.permute(2, 0, 1)[None] * 2 - 1, 256,
                                   mode="bilinear", align_corners=False)
                loss_perc = loss_perc + lpips_fn(r, g2).mean()
        n = max(len(choices), 1)

        loss_orient = torch.zeros((), device=dev)
        if args.w_orient > 0:
            fur0 = {k: gs["fur"][k][0] for k in ("means", "tangent", "quats", "scales", "opacities")}
            no = min(args.orient_views, len(choices))
            for j in choices[:no]:
                loss_orient = loss_orient + orientation_loss(fur0, scene["views"][j], gab_cache[j])
            loss_orient = loss_orient / max(no, 1)

        L = gs["fur"]["length"]                                       # [B,N]
        len_prior = (L / fur_sigma.view(-1, 1).clamp_min(1e-6)).mean()
        smooth = (L[:, e[:, 0]] - L[:, e[:, 1]]).abs().mean()
        d = gs["fur"]["direction"]                                    # [B,N,3]
        dir_smooth = (1.0 - (d[:, e[:, 0]] * d[:, e[:, 1]]).sum(-1)).mean()
        c = gs["body"]["rgb"]                                         # shared coat albedo [B,N,3]
        rgb_smooth = (c[:, e[:, 0]] - c[:, e[:, 1]]).abs().mean()
        loss_lgeo = torch.zeros((), device=dev)
        if len_geo_target is not None:
            loss_lgeo = (gs["fur"]["length"][0][len_geo_valid]
                         - len_geo_target[len_geo_valid]).abs().mean()
        loss = ((loss_rgb + loss_mask + args.w_outside * loss_out
                 + args.lpips_weight * loss_perc + args.w_face_crop * loss_face) / n
                + args.w_len * len_prior + args.w_smooth * smooth + args.w_dir_smooth * dir_smooth
                + args.w_rgb_smooth * rgb_smooth
                + args.w_len_geo * loss_lgeo
                + args.w_orient * loss_orient)

        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if torch.isfinite(gnorm):
            opt.step()                                           # skip non-finite grad steps
        else:
            print(f"it{it:5d} skipped (non-finite grad)", flush=True)
        if sched is not None:
            sched.step()
        if args.save_every and it > 0 and it % args.save_every == 0:
            torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("bb.dino.")},
                       os.path.join(args.out, "model.pt"))

        if it % 50 == 0:
            Ld = gs["fur"]["length"] / body_diag.view(-1, 1)     # length as fraction of body_diag
            print(f"it{it:5d} loss={float(loss):.4f} rgb={float(loss_rgb)/n:.4f} "
                  f"mask={float(loss_mask)/n:.4f} orient={float(loss_orient):.4f} "
                  f"furop={float(gs['fur']['opacities'].mean()):.3f} "
                  f"len µ={float(Ld.mean()):.3f}/σ={float(Ld.std()):.3f} dirsm={float(dir_smooth):.3f}", flush=True)
        if it % args.vis_every == 0 or it == args.iters - 1:
            with torch.no_grad():
                v = scene["views"][len(scene["views"]) // 2]
                rgb, _ = render(composite(gs["body"], gs["fur"], 0), v, white)
                pair = np.concatenate([(rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                                       (v["rgb"].cpu().numpy() * 255).astype(np.uint8)], axis=1)
                Image.fromarray(pair).save(os.path.join(args.out, f"it{it:05d}.png"))

    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("bb.dino.")},
               os.path.join(args.out, "model.pt"))
    model.eval()
    with torch.no_grad():
        gs = run(inputs)
        v = scene["views"][len(scene["views"]) // 2]
        rgb, _ = render(composite(gs["body"], gs["fur"], 0), v, white)
        pair = np.concatenate([(rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                               (v["rgb"].cpu().numpy() * 255).astype(np.uint8)], axis=1)
        Image.fromarray(pair).save(os.path.join(args.out, "final_render_vs_gt.png"))
        body = {k: gs["body"][k][0].detach() for k in ("means", "quats", "scales", "opacities", "rgb")}
        fur = {k: gs["fur"][k][0].detach() for k in
               ("means", "quats", "scales", "opacities", "rgb", "roots", "delta")}
        save_ply(os.path.join(args.out, "final_full.ply"),
                 torch.cat([body["means"], fur["means"]]), torch.cat([body["scales"], fur["scales"]]),
                 torch.cat([body["quats"], fur["quats"]]), torch.cat([body["opacities"], fur["opacities"]]),
                 torch.cat([body["rgb"], fur["rgb"]]))
        import imageio
        frames = sway_dynamics(body, fur, v, amp_frac=0.5, frames=args.anim_frames, fps=args.anim_fps)
        path = os.path.join(args.out, f"strand_{scene['name']}_{scene['preset']}.mp4")
        imageio.mimsave(path, frames, fps=args.anim_fps, quality=8)
        arr = np.stack([f.astype(np.float32) for f in frames])
        dmax = np.abs(arr - arr[0]).mean(axis=(1, 2, 3)).max()
        print(f"furop final={float(gs['fur']['opacities'].mean()):.3f} | dyn max|frame-frame0|={dmax:.3f} | {path}",
              flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
