#!/usr/bin/env python3
"""Per-dog fur optimization = TEACHER for the feed-forward student.

Init: grow_strands groom (torch port of preprocess/blender_fur_dataset.py) on the dog's
posed D-SMAL, with its own L_geo/w_face/w_ear fields and vlm curl_class style prior.
Optimize ONLY what multi-view photometry can resolve (user decision):
  - APPEARANCE: per-root strand albedo + body skin colour (both init by projecting GT views)
  - LENGTH:     per-root d_logL residual
  - DIRECTION:  per-root comb residual (azimuth d_az around normal + loft d_loft) + global droop
Curl / clump / frizz stay frozen at the style prior. Residuals kNN-smoothed + L2-regularised
so the prior stays dominant (photometric L1 alone cannot be trusted for fur -- oracle finding).

  PATH=$ENV/bin:$PATH TORCH_EXTENSIONS_DIR=.torch_ext_lhm python optimize_fur_dog.py \
      --dog 00085-kotori --iters 800 --out exps/fur_teacher
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from PIL import Image
from dog_lrm.render import render_gaussians, intrinsics
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from train_fur_strand import quat_align_z
from train_strand_predictor import vnormals, tbn_frames

dev = "cuda"
DSMAL_ROOT = "received_data_from_Pinstudio_20260424/InterPet2026/dsmal_dataset"
CLASS2STYLE = {"short_smooth": "short", "double_coat": "short", "long_straight": "long",
               "wavy": "wavy", "wire": "wavy", "curly": "curly"}
STYLES = {"short": (0.04, 0.0, 0.0), "long": (0.12, 0.0, 0.0),
          "curly": (0.085, 0.55, 3.0), "wavy": (0.10, 0.25, 1.6)}


def load_scene(root, dog):
    pre = os.path.join(root, dog, "colmap", "preprocess")
    cams = json.load(open(os.path.join(pre, "cameras.json")))["frames"]
    imgs, masks, Ks, c2ws = [], [], [], []
    for fr in cams:
        stem = os.path.splitext(os.path.basename(fr["name"]))[0]
        fp = os.path.join(pre, "cache_s4", stem + ".jpg")
        mp = os.path.join(pre, "cache_s4", stem + ".png")
        if not (os.path.exists(fp) and os.path.exists(mp)):
            continue
        img = torch.from_numpy(np.asarray(Image.open(fp), np.float32) / 255.)
        m = torch.from_numpy((np.asarray(Image.open(mp).convert("L"), np.float32) > 127).astype(np.float32))
        imgs.append(img); masks.append(m)
        Ks.append(intrinsics(fr["fx"], fr["fy"], fr["cx"], fr["cy"], "cpu"))
        c2ws.append(torch.tensor(fr["c2w"], dtype=torch.float32))
    K0 = torch.stack(Ks); c2w = torch.stack(c2ws)
    H, W = imgs[0].shape[:2]
    return imgs, masks, K0, c2w, H, W


def scale_K(K, H, W, fullH, fullW):
    K = K.clone()
    K[:, 0] *= W / fullW
    K[:, 1] *= H / fullH
    return K


class Groom(torch.nn.Module):
    """differentiable regrow: fixed prior fields + learnable length/direction/appearance"""

    def __init__(self, verts, faces, lgeo, wface, wear, style, K=10, n=80000, seed=0, nofur=None):
        super().__init__()
        g = torch.Generator(device=dev).manual_seed(seed)
        vn = vnormals(verts, faces)
        length, kamp, kfreq = style
        tri = verts[faces]
        area = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1) / 2
        lengthw = (lgeo / lgeo.max().clamp(min=1e-6)) * (1 - 0.85 * wface.clamp(0, 1)) * (1 - 0.5 * wear.clamp(0, 1))
        nz = lengthw[lengthw > 1e-4]
        lo = torch.quantile(nz, 0.08) if nz.numel() else torch.tensor(1e-6)
        dens = (lengthw / lo.clamp(min=1e-6)).clamp(0, 1)
        dens = dens * dens * (3 - 2 * dens); dens[lengthw < 1e-4] = 0
        if nofur is not None:
            dens = dens * (1 - nofur.clamp(0, 1))                     # NO strands on face/ears/paws (user prior):
        # their detail is carried by the body skin, strands only smear it
        fd = dens[faces].mean(1) * area
        fi = torch.multinomial(fd.clamp(min=1e-12), n, replacement=True, generator=g)
        r = torch.rand(n, 2, device=dev, generator=g); su = r[:, 0].sqrt()
        b = torch.stack([1 - su, su * (1 - r[:, 1]), su * r[:, 1]], 1)
        self.fi, self.bary = fi, b                                    # (face,bary) binding: roots track
        self.faces = faces                                            # the DEFORMED surface (soft body)
        self.roots = (tri[fi] * b[..., None]).sum(1)
        nrm = F.normalize((vn[faces[fi]] * b[..., None]).sum(1), dim=-1)
        lw = (lengthw[faces[fi]] * b).sum(1)
        ref = torch.quantile(lengthw[lengthw > 0.05], 0.70)
        diag = float((verts.max(0).values - verts.min(0).values).norm())
        self.diag = diag
        L0 = length * (lw / ref.clamp(min=1e-6)).clamp(0.08, 1.6) * (0.85 + 0.3 * torch.rand(n, device=dev, generator=g))
        self.L0 = L0 * diag / 2.154                                    # style lengths defined at diag 2.154
        flow = F.normalize(torch.tensor([-1.0, 0.0, -0.6], device=dev), dim=0)
        t = flow[None] - (nrm @ flow)[:, None] * nrm
        bad = t.norm(dim=1) < 1e-6
        t[bad] = torch.tensor([-1.0, 0.0, 0.0], device=dev)
        self.t0 = F.normalize(t, dim=-1)
        self.n = nrm
        self.b0 = torch.cross(nrm, self.t0, dim=-1)
        self.az0 = 0.15 * torch.randn(n, device=dev, generator=g)
        self.tmix0 = (0.65 + 0.08 * torch.randn(n, device=dev, generator=g)).clamp(0.45, 0.85)
        self.K = K
        self.kamp, self.kfreq = kamp, kfreq
        self.ph = 2 * math.pi * torch.rand(n, 1, device=dev, generator=g)
        gsel = torch.randperm(n, device=dev, generator=g)[: n // 15]
        self.gid = gsel[torch.cdist(self.roots, self.roots[gsel]).argmin(1)]
        self.cf = (0.35 + 0.08 * torch.randn(n, device=dev, generator=g)).clamp(0.1, 0.6)
        ph1, ph2 = 2 * math.pi * torch.rand(2, n, 1, device=dev, generator=g)
        k = torch.arange(K, device=dev, dtype=torch.float32)[None]
        self.wob1, self.wob2 = torch.sin(k * 1.7 + ph1), torch.cos(k * 2.3 + ph2)
        # learnable (LENGTH + DIRECTION + APPEARANCE; everything else frozen prior)
        tone_clump = (1 + 0.12 * torch.randn(n, 1, device=dev, generator=g)).clamp(0.7, 1.3)
        tone_strand = (1 + 0.05 * torch.randn(n, 1, device=dev, generator=g)).clamp(0.85, 1.15)
        self.tone = tone_clump[self.gid] * tone_strand                # lock-scale light/dark, soft per-hair
        self.d_logL = torch.nn.Parameter(torch.zeros(n))
        self.d_az = torch.nn.Parameter(torch.zeros(n))
        self.d_loft = torch.nn.Parameter(torch.zeros(n))
        self.droop_g = torch.nn.Parameter(torch.tensor(0.45))
        self.albedo = torch.nn.Parameter(torch.full((n, 3), 0.5))
        knn = torch.empty(n, 8, dtype=torch.long, device=dev)
        for i in range(0, n, 8192):                                   # chunked: full cdist would be 25GB
            d = torch.cdist(self.roots[i:i + 8192], self.roots)
            knn[i:i + 8192] = d.topk(9, largest=False).indices[:, 1:]
        self.knn = knn

    def set_surface(self, verts, vn):
        """re-bind roots/frames to a deformed surface (keeps az0/tmix0/L0/clump/knn fixed)"""
        f = self.faces[self.fi]
        self.roots = (verts[f] * self.bary[..., None]).sum(1)
        nrm = F.normalize((vn[f] * self.bary[..., None]).sum(1), dim=-1)
        self.n = nrm
        flow = F.normalize(torch.tensor([-1.0, 0.0, -0.6], device=verts.device), dim=0)
        t = flow[None] - (nrm @ flow)[:, None] * nrm
        bad = t.norm(dim=1) < 1e-6
        t[bad] = torch.tensor([-1.0, 0.0, 0.0], device=verts.device)
        self.t0 = F.normalize(t, dim=-1)
        self.b0 = torch.cross(nrm, self.t0, dim=-1)

    def strands(self):
        n, K = self.L0.shape[0], self.K
        az = self.az0 + self.d_az.clamp(-0.8, 0.8)
        t = self.t0 * az.cos()[:, None] + self.b0 * az.sin()[:, None]
        tmix = (self.tmix0 + self.d_loft.clamp(-0.3, 0.3)).clamp(0.3, 0.95)[:, None]
        d = F.normalize(tmix * t + (1 - tmix) * self.n, dim=-1)
        L = self.L0 * self.d_logL.clamp(-0.45, 0.45).exp()            # tighter: lone 2x-length strands = spikes
        down = torch.tensor([0., 0., -1.], device=dev)
        frac = (torch.arange(1, K, device=dev) / (K - 1))[None, :, None]
        droop = (self.droop_g.clamp(0.1, 1.0) * (L / (0.08 * self.diag / 2.154)).clamp(0.6, 2.4))[:, None, None]
        dirs = F.normalize(d[:, None] + droop * frac ** 2 * down, dim=-1)
        seg = (L / (K - 1))[:, None, None]
        pts = torch.cat([self.roots[:, None], self.roots[:, None] + torch.cumsum(dirs * seg, 1)], 1)
        if self.kamp > 0:
            e1 = F.normalize(torch.cross(d, down.expand_as(d), dim=-1), dim=-1)
            e2 = torch.cross(d, e1, dim=-1)
            kk = torch.arange(K, device=dev, dtype=torch.float32)[None]
            th = 2 * math.pi * self.kfreq * kk / (K - 1) + self.ph
            amp = (self.kamp * L * 0.35)[:, None] * (kk / (K - 1))
            pts = pts + (th.cos() * amp)[..., None] * e1[:, None] + (th.sin() * amp)[..., None] * e2[:, None]
        w = self.cf[:, None, None] * (torch.arange(K, device=dev)[None, :, None] / (K - 1)) ** 1.2
        pts = pts + w * (pts[self.gid] - pts)
        fz = (0.06 * L).clamp(max=0.006 * self.diag / 2.154)[:, None]
        e1 = F.normalize(torch.cross(d, down.expand_as(d), dim=-1), dim=-1)
        e2 = torch.cross(d, e1, dim=-1)
        pts = pts + (self.wob1 * fz)[..., None] * e1[:, None] + (self.wob2 * fz)[..., None] * e2[:, None]
        pts = torch.cat([self.roots[:, None], pts[:, 1:]], 1)          # keep roots on surface
        return pts

    def gaussians(self, radius_frac=None):
        if radius_frac is None:
            radius_frac = 0.0007 if self.kamp > 0.3 else 0.0005       # soft: fatter + translucent (v11 recipe)
        pts = self.strands()
        p0, p1 = pts[:, :-1].reshape(-1, 3), pts[:, 1:].reshape(-1, 3)
        mid = (p0 + p1) / 2
        seg = p1 - p0
        sl = seg.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        quat = quat_align_z(seg / sl)
        r = radius_frac * self.diag
        scales = torch.cat([torch.full_like(sl, r), torch.full_like(sl, r), sl * 0.6], -1)
        rgb = (self.albedo.clamp(0, 1) * self.tone).clamp(0, 1)
        rgb = rgb[:, None].expand(-1, self.K - 1, -1).reshape(-1, 3)
        return mid, quat, scales, rgb


def rot6d_to_mat(x):
    """[...,6] -> [...,3,3] Gram-Schmidt (Zhou et al.)"""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)


class SmalRig(torch.nn.Module):
    """D-SMAL parametric body: refine pose/shape/offsets FROM the dog's stored fit.
    Init reproduces dsmal_anchors['posed'] exactly (same params + scene_norm), so the
    optimization starts image-aligned; the refined (pose, betas, trans, scale, vert_off)
    are themselves ff-training targets (user: shape&pose supervision for the student)."""

    def __init__(self, dog, scene_root):
        super().__init__()
        cwd = os.getcwd()
        os.chdir(DSMAL_ROOT)
        sys.path.insert(0, os.path.join(os.getcwd(), "dsmal_code"))
        try:
            import _compat_shim  # noqa: F401
            from smal_pytorch.smal_model.smal_torch_new import SMAL
            from configs.SMAL_configs import SMAL_MODEL_CONFIG
            self.smal = SMAL(smal_model_type="39dogs_norm_newv3", template_name="neutral",
                             logscale_part_list=SMAL_MODEL_CONFIG["39dogs_norm_newv3"]["logscale_part_list"]).to(dev)
        finally:
            os.chdir(cwd)
        d = np.load(os.path.join(DSMAL_ROOT, "params", f"{dog}.npz"))
        t = lambda k: torch.tensor(d[f"offset_{k}"], device=dev).float()
        for k in ("betas", "betas_limbs", "trans", "log_scale", "vert_off_compact", "pose"):
            self.register_buffer(k + "0", t(k))
        for k in ("betas", "betas_limbs", "trans", "log_scale", "vert_off_compact"):
            setattr(self, "d_" + k, torch.nn.Parameter(torch.zeros_like(getattr(self, k + "0"))))
        id6 = torch.tensor([1., 0., 0., 0., 1., 0.], device=dev).expand(1, 35, 6).clone()
        self.d_rot = torch.nn.Parameter(id6)                           # delta rotation, identity init:
        # pose = rot6d(d_rot) @ pose0 -> init reproduces the cached fit EXACTLY
        sn = json.load(open(os.path.join(scene_root, dog, "colmap/preprocess/scene_norm.json")))
        self.register_buffer("nc", torch.tensor(sn["center"], device=dev).float())
        self.ns = float(sn["scale"])

    def verts(self):
        pose = rot6d_to_mat(self.d_rot) @ self.pose0                   # [1,35,3,3]
        v = self.smal(beta=self.betas0 + self.d_betas,
                      betas_limbs=self.betas_limbs0 + self.d_betas_limbs, pose=pose,
                      trans=self.trans0 + self.d_trans,
                      vert_off_compact=self.vert_off_compact0 + self.d_vert_off_compact,
                      get_skin=True, uniform_scale=(self.log_scale0 + self.d_log_scale).exp())[0][0]
        return (v - self.nc) * self.ns

    def reg(self):
        id6 = torch.tensor([1., 0., 0., 0., 1., 0.], device=self.d_rot.device)
        return (2.0 * (self.d_rot - id6).pow(2).mean()
                + 1.0 * self.d_betas.pow(2).mean() + 1.0 * self.d_betas_limbs.pow(2).mean()
                + 10.0 * self.d_trans.pow(2).mean() + 10.0 * self.d_log_scale.pow(2).mean()
                + 5.0 * self.d_vert_off_compact.pow(2).mean())


def project_color(pts, imgs, masks, Ks, c2ws, k=12):
    """init colours by averaging GT pixels over k evenly-spaced views (no occlusion test)"""
    acc = torch.zeros(pts.shape[0], 3, device=dev); wsum = torch.zeros(pts.shape[0], 1, device=dev)
    ids = np.linspace(0, len(imgs) - 1, k).astype(int)
    for i in ids:
        img, m = imgs[i].to(dev), masks[i].to(dev)
        H, W = img.shape[:2]
        w2c = torch.inverse(c2ws[i].to(dev))
        cam = (w2c[:3, :3] @ pts.T + w2c[:3, 3:4]).T
        z = cam[:, 2].clamp(min=1e-4)
        uv = (Ks[i].to(dev) @ (cam / z[:, None]).T).T[:, :2]
        gu = torch.stack([uv[:, 0] / W * 2 - 1, uv[:, 1] / H * 2 - 1], -1)
        inb = (gu.abs() < 0.99).all(-1) & (z > 0)
        col = F.grid_sample(img.permute(2, 0, 1)[None], gu[None, :, None], align_corners=False)[0, :, :, 0].T
        mm = F.grid_sample(m[None, None], gu[None, :, None], align_corners=False)[0, 0, :, 0]
        w = (inb.float() * mm)[:, None]
        acc += col * w; wsum += w
    return (acc / wsum.clamp(min=1e-6)).clamp(0.05, 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--dog", default="00085-kotori")
    ap.add_argument("--priors", default="exps/vlm_priors")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--views_per_step", type=int, default=0, help="views accumulated per optimizer step; 0 = ALL train views (global consensus gradient)")
    ap.add_argument("--beauty", type=int, default=0, help="appearance-refine steps AFTER geometry (freeze geo, per-segment colour, LPIPS+struct)")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--fur_op", type=float, default=0.65)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--w_sil", type=float, default=0.5)
    ap.add_argument("--w_smooth", type=float, default=1.0)
    ap.add_argument("--w_reg", type=float, default=0.1)
    ap.add_argument("--soft_body", action="store_true", help="learnable low-freq shape offset (fit != template)")
    ap.add_argument("--free_body", action="store_true", help="GaussianAvatars-style body offsets (needs long schedule)")
    ap.add_argument("--opt_smal", action="store_true", help="refine D-SMAL params (pose/betas/scale/vert_off) instead of free offsets")
    ap.add_argument("--warm", type=int, default=30, help="sil-only warm-up FULL-BATCH steps")
    ap.add_argument("--off_cap", type=float, default=0.04, help="offset cap in diag units")
    ap.add_argument("--w_lap", type=float, default=20.0)
    ap.add_argument("--out", default="exps/fur_teacher")
    args = ap.parse_args()
    out = os.path.join(args.out, args.dog); os.makedirs(out, exist_ok=True)

    imgs, masks, K0, c2ws, H, W = load_scene(args.root, args.dog)
    # cameras.json intrinsics are for FULL res; cache_s4 is /4
    full = np.asarray(Image.open(glob.glob(os.path.join(args.root, args.dog, "colmap/preprocess/masks/*"))[0]))
    Ks = scale_K(K0, H, W, full.shape[0], full.shape[1]).to(dev)
    c2ws = c2ws.to(dev)
    print(f"[opt] {args.dog}: {len(imgs)} views @ {W}x{H}", flush=True)

    pre = os.path.join(args.root, args.dog, "colmap", "preprocess")
    da = np.load(os.path.join(pre, "dsmal_anchors.npz"))
    fa = np.load(os.path.join(pre, "fur_anchors.npz"))
    posed = torch.from_numpy(da["posed"].astype(np.float32)).to(dev)
    faces0 = torch.from_numpy(da["faces"].astype(np.int64))
    M = build_subdiv(faces0, 1, "cpu")
    verts = torch.sparse.mm(M, posed.cpu()).to(dev)
    faces = subdivided_faces(faces0, 1).to(dev)
    M2 = build_subdiv(faces0, 2, "cpu")                                # body skin at subdiv-2 (62k):
    bverts0 = torch.sparse.mm(M2, posed.cpu()).to(dev)                 # performance-ceiling run, no economy
    lgeo = torch.from_numpy(fa["L_geo"].astype(np.float32)).to(dev)
    wface = torch.from_numpy(fa["w_face"].astype(np.float32)).to(dev)
    wear = torch.from_numpy(fa["w_ear"].astype(np.float32)).to(dev)

    meta = json.load(open(os.path.join(args.priors, f"{args.dog}.json")))
    style = STYLES[CLASS2STYLE.get(meta.get("curl_class", "short_smooth"), "short")]
    print(f"[opt] curl_class={meta.get('curl_class')} -> style {style}", flush=True)

    paw = torch.from_numpy(np.load("synth_fur/paw_mask.npy").astype(np.float32)).to(dev)
    nofur = ((wface > 0.25) | (wear > 0.25) | (paw > 0.5)).float()
    groom = Groom(verts, faces, lgeo, wface, wear, style, K=args.K, n=args.n, nofur=nofur).to(dev)

    # soft body: scalar offset along base-vertex normal, subdiv-propagated; Laplacian + cap keep it low-freq
    posed_gpu = posed
    faces0_gpu = faces0.to(dev)
    n_base = vnormals(posed_gpu, faces0_gpu)
    M1c, M2c = M.to(dev), M2.to(dev)
    off = torch.nn.Parameter(torch.zeros(posed.shape[0], device=dev))
    e01 = torch.cat([faces0_gpu[:, [0, 1]], faces0_gpu[:, [1, 2]], faces0_gpu[:, [2, 0]]])   # edges for Laplacian

    rig = SmalRig(args.dog, args.root).to(dev) if args.opt_smal else None
    pbase = {}

    def surface():
        if args.opt_smal:
            p = rig.verts()
            pbase["p"] = p
        elif args.soft_body:
            p = posed_gpu + (off.tanh() * args.off_cap * groom.diag)[:, None] * n_base
        else:
            return verts, bverts0
        return torch.sparse.mm(M1c, p), torch.sparse.mm(M2c, p)
    with torch.no_grad():
        groom.albedo.copy_(project_color(groom.roots, imgs, masks, Ks, c2ws))
    diag = groom.diag

    # body skin gaussians on subdiv-2 verts (colour learnable, init projected)
    # GaussianAvatars-style: learnable LOCAL offset + log-scale per gaussian -- the mesh is only
    # the animation scaffold; the splats move off-surface to align with pixels (SMAL residuals
    # are absorbed here, not fought in the rig).
    body_rgb = torch.nn.Parameter(project_color(bverts0, imgs, masks, Ks, c2ws))
    vq = torch.zeros(bverts0.shape[0], 4, device=dev); vq[:, 0] = 1
    body_doff = torch.nn.Parameter(torch.zeros(bverts0.shape[0], 3, device=dev))
    body_logs = torch.nn.Parameter(torch.zeros(bverts0.shape[0], 1, device=dev))
    M2f = subdivided_faces(faces0, 2).to(dev)

    opt = torch.optim.Adam([
        dict(params=[groom.d_logL, groom.d_az, groom.d_loft], lr=args.lr),
        dict(params=[groom.droop_g], lr=args.lr * 0.5),
        dict(params=[off] + (list(rig.parameters()) if rig is not None else []), lr=args.lr * 0.3),
        dict(params=[groom.albedo, body_rgb], lr=args.lr),
        dict(params=[body_doff, body_logs], lr=(args.lr * 0.5) if args.free_body else 0.0)])

    train_ids = [i for i in range(len(imgs)) if i % 8 != 4]
    test_ids = [i for i in range(len(imgs)) if i % 8 == 4]
    white = torch.ones(3, device=dev)
    fur_op = args.fur_op

    def lap_loss():
        return args.w_lap * (off[e01[:, 0]] - off[e01[:, 1]]).pow(2).mean() + 0.01 * off.pow(2).mean()

    def def_smooth():
        dv = pbase["p"] - posed_gpu                                    # added deformation must be LOW-FREQ
        return 300.0 * (dv[e01[:, 0]] - dv[e01[:, 1]]).pow(2).mean()

    def render(vi, bv):
        mid, quat, scales, rgb = groom.gaussians()
        bvn = vnormals(bv, M2f)
        btbn = tbn_frames(bv, bvn)                                     # local frame: offsets ride animation
        bmeans = bv + torch.einsum("nij,nj->ni", btbn, body_doff.clamp(-0.02, 0.02) * diag)
        bscl = (0.0025 * diag) * body_logs.clamp(-1.2, 1.2).exp().expand(-1, 3)
        means = torch.cat([mid, bmeans])
        quats = torch.cat([quat, vq])
        scl = torch.cat([scales, bscl])
        cols = torch.cat([rgb, body_rgb.clamp(0, 1)])
        ops = torch.cat([torch.full((mid.shape[0],), fur_op, device=dev),
                         torch.ones(bv.shape[0], device=dev)])
        return render_gaussians(means, quats, scl, ops, cols, c2ws[vi], Ks[vi], W, H, bg=white)

    def rebind():
        v1, bv = surface()
        if args.soft_body or args.opt_smal:
            groom.set_surface(v1, vnormals(v1, faces))
        return bv

    if args.soft_body or args.opt_smal:                                # warm-up: SILHOUETTE-only shape fit
        wp = list(rig.parameters()) if args.opt_smal else [off]        # (contours can't bake fur into geometry)
        opt_off = torch.optim.Adam(wp, lr=2e-3 if args.opt_smal else 2e-2)
        Mw = len(train_ids)
        for wit in range(args.warm):
            opt_off.zero_grad()
            ws = 0.0
            for vi in train_ids:                                       # ALL views: consensus shape fit
                bv = rebind()
                _, alpha = render(vi, bv)
                wl = F.mse_loss(alpha[..., 0], masks[vi].to(dev)) / Mw
                wl.backward()
                ws += float(wl)
            rebind()                                               # fresh graph for the reg terms
            reg = (0.01 * rig.reg() + def_smooth() if args.opt_smal else lap_loss())
            reg.backward()
            opt_off.step()
            if wit % 10 == 0:
                print(f"warm{wit:4d} sil={ws:.4f}", flush=True)

    M = len(train_ids) if args.views_per_step == 0 else args.views_per_step
    for it in range(args.iters + 1):
        opt.zero_grad()
        vids = train_ids if args.views_per_step == 0 else \
            [train_ids[j] for j in np.random.choice(len(train_ids), M, replace=False)]
        photo = 0.0
        for vi in vids:                                                # accumulate: multi-view CONSENSUS
            bv = rebind()                                              # gradient (kills per-view overfit)
            rgb, alpha = render(vi, bv)
            img, m = imgs[vi].to(dev), masks[vi].to(dev)
            gt = img * m[..., None] + (1 - m[..., None])
            li = ((rgb - gt).abs().mean() + args.w_sil * F.mse_loss(alpha[..., 0], m)) / M
            li.backward()
            photo += float(li)
        bv = rebind()
        loss = torch.zeros((), device=dev)
        if args.soft_body:
            loss = loss + lap_loss()
        if args.opt_smal:
            loss = loss + 0.01 * rig.reg() + def_smooth()
        loss = loss + 0.5 * body_doff.pow(2).mean() + 0.05 * body_logs.pow(2).mean()
        nb = groom.knn
        for p, wgt in [(groom.d_logL, 3.0), (groom.d_az, 1.0), (groom.d_loft, 1.0)]:
            loss = loss + args.w_smooth * wgt * (p[:, None] - p[nb]).pow(2).mean()
        loss = loss + args.w_smooth * 2.0 * (groom.albedo[:, None] - groom.albedo[nb]).pow(2).mean()
        loss = loss + args.w_reg * (groom.d_logL.pow(2).mean() + groom.d_az.pow(2).mean() + groom.d_loft.pow(2).mean())
        loss.backward()
        loss = loss + photo                                            # for logging only
        opt.step()
        if it % 100 == 0:
            with torch.no_grad():
                tl = 0.
                for ti in test_ids[:6]:
                    r, _ = render(ti, bv)
                    gtt = imgs[ti].to(dev) * masks[ti].to(dev)[..., None] + (1 - masks[ti].to(dev)[..., None])
                    tl += float((r - gtt).abs().mean())
            print(f"it{it:4d} loss={float(loss):.4f} heldout_L1={tl/6:.4f} droop={float(groom.droop_g):.2f}", flush=True)

    if args.beauty > 0:
        # ---- BEAUTY PASS (v11 recipe): geometry FROZEN, appearance free but structure-constrained
        for p in list(groom.parameters()) + ([off] + (list(rig.parameters()) if rig is not None else [])):
            p.requires_grad_(False)
        groom.albedo.requires_grad_(True)
        seg_dalb = torch.nn.Parameter(torch.zeros(args.n, args.K - 1, 3, device=dev))
        import lpips as lpips_lib
        lp = lpips_lib.LPIPS(net="vgg").to(dev).eval()
        bopt = torch.optim.Adam([dict(params=[seg_dalb], lr=5e-3),
                                 dict(params=[groom.albedo, body_rgb], lr=2e-3)])
        bv = rebind()
        mid, quat, scales, _ = groom.gaussians()                       # geometry fixed once
        mid, quat, scales = mid.detach(), quat.detach(), scales.detach()
        bvn = vnormals(bv, M2f); btbn = tbn_frames(bv, bvn)
        bmeans = (bv + torch.einsum("nij,nj->ni", btbn, body_doff.clamp(-0.02, 0.02) * diag)).detach()
        bscl = ((0.0025 * diag) * body_logs.clamp(-1.2, 1.2).exp().expand(-1, 3)).detach()
        ops_all = torch.cat([torch.full((mid.shape[0],), fur_op, device=dev), torch.ones(bmeans.shape[0], device=dev)])
        nb = groom.knn
        for it in range(args.beauty + 1):
            vi = train_ids[np.random.randint(len(train_ids))]
            base_rgb = (groom.albedo.clamp(0, 1) * groom.tone).clamp(0, 1)
            rgbseg = (base_rgb[:, None] + seg_dalb).clamp(0, 1).reshape(-1, 3)
            cols = torch.cat([rgbseg, body_rgb.clamp(0, 1)])
            rgb, alpha = render_gaussians(torch.cat([mid, bmeans]), torch.cat([quat, vq]),
                                          torch.cat([scales, bscl]), ops_all, cols,
                                          c2ws[vi], Ks[vi], W, H, bg=white)
            img, m = imgs[vi].to(dev), masks[vi].to(dev)
            gt = img * m[..., None] + (1 - m[..., None])
            loss = (rgb - gt).abs().mean()
            loss = loss + 0.4 * lp(rgb.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).mean()
            salb = base_rgb + seg_dalb.mean(1)                         # struct: strand-mean colour agrees w/ knn
            loss = loss + 1.0 * (salb[:, None] - salb[nb]).pow(2).mean()
            loss = loss + 0.05 * seg_dalb.pow(2).mean()
            bopt.zero_grad(); loss.backward(); bopt.step()
            if it % 100 == 0:
                with torch.no_grad():
                    tl = 0.
                    for ti in test_ids[:4]:
                        r, _ = render_gaussians(torch.cat([mid, bmeans]), torch.cat([quat, vq]),
                                                torch.cat([scales, bscl]), ops_all, cols,
                                                c2ws[ti], Ks[ti], W, H, bg=white)
                        gtt = imgs[ti].to(dev) * masks[ti].to(dev)[..., None] + (1 - masks[ti].to(dev)[..., None])
                        tl += float((r[0] if isinstance(r, tuple) else r - gtt).abs().mean()) if False else float((r - gtt).abs().mean())
                print(f"beauty{it:5d} loss={float(loss):.4f} heldout_L1={tl/4:.4f}", flush=True)

    with torch.no_grad():
        strands = groom.strands()
        rigp = {k: v.detach().cpu() for k, v in rig.named_parameters()} if rig is not None else {}
        if args.beauty > 0:
            rigp["seg_dalb"] = seg_dalb.detach().cpu()
        rigp["body_doff"] = body_doff.detach().cpu(); rigp["body_logs"] = body_logs.detach().cpu()
        torch.save(dict(strands=strands.cpu(), roots=groom.roots.cpu(), vert_offset=off.detach().cpu(), rig=rigp, albedo=groom.albedo.clamp(0, 1).cpu(),
                        body_rgb=body_rgb.clamp(0, 1).cpu(), bverts=bv.detach().cpu(), verts=verts.cpu(), faces=faces.cpu(),
                        d_logL=groom.d_logL.cpu(), d_az=groom.d_az.cpu(), d_loft=groom.d_loft.cpu(),
                        droop_g=groom.droop_g.cpu(), style=style, args=vars(args)),
                   os.path.join(out, "teacher.pt"))
        import torchvision
        # black-bg 3D orbit (viewer-style) of the final asset
        mid, quat, scales, rgb = groom.gaussians()
        if args.beauty > 0:
            base_rgb = (groom.albedo.clamp(0, 1) * groom.tone).clamp(0, 1)
            rgb = (base_rgb[:, None] + seg_dalb).clamp(0, 1).reshape(-1, 3)
        bvn2 = vnormals(bv, M2f); btbn2 = tbn_frames(bv, bvn2)
        bmeans2 = bv + torch.einsum("nij,nj->ni", btbn2, body_doff.clamp(-0.02, 0.02) * diag)
        bscl2 = (0.0025 * diag) * body_logs.clamp(-1.2, 1.2).exp().expand(-1, 3)
        Ms = torch.cat([mid, bmeans2]); Qs = torch.cat([quat, vq])
        Ss = torch.cat([scales, bscl2]); Cs = torch.cat([rgb, body_rgb.clamp(0, 1)])
        Os = torch.cat([torch.full((mid.shape[0],), fur_op, device=dev), torch.ones(bmeans2.shape[0], device=dev)])
        import math as _m
        ctr2 = (Ms.max(0).values + Ms.min(0).values) / 2
        rad2 = float((Ms.max(0).values - Ms.min(0).values).norm()) * 1.15
        upw = -c2ws[0][:3, 1]
        Ko = torch.tensor([[900., 0, 400], [0, 900., 400], [0, 0, 1]], device=dev)
        blk = torch.zeros(3, device=dev)
        orbs = []
        for oi in range(6):
            az = 2 * _m.pi * oi / 6
            base_fwd = F.normalize(torch.linalg.cross(upw, torch.tensor([1., 0., 0.], device=dev)), dim=0)
            side = torch.linalg.cross(upw, base_fwd)
            dirv = F.normalize(_m.cos(az) * base_fwd + _m.sin(az) * side - 0.2 * upw, dim=0)
            eye = ctr2 - rad2 * dirv
            fwd = F.normalize(ctr2 - eye, dim=0)
            right = F.normalize(torch.linalg.cross(fwd, upw), dim=0)
            upv = torch.linalg.cross(right, fwd)
            c2wo = torch.eye(4, device=dev)
            c2wo[:3, 0], c2wo[:3, 1], c2wo[:3, 2], c2wo[:3, 3] = right, -upv, fwd, eye
            ro, _ = render_gaussians(Ms, Qs, Ss, Os, Cs, c2wo, Ko, 800, 800, bg=blk)
            orbs.append(ro.clamp(0, 1).permute(2, 0, 1))
        torchvision.utils.save_image(orbs, os.path.join(out, "orbit.png"), nrow=3)
        panels = []
        for ti in test_ids[:4]:
            r, _ = render(ti, bv)
            gtt = imgs[ti].to(dev) * masks[ti].to(dev)[..., None] + (1 - masks[ti].to(dev)[..., None])
            panels.append(torch.cat([gtt, r], 1).permute(2, 0, 1))
        torchvision.utils.save_image(panels, os.path.join(out, "cmp.png"), nrow=1)
    print(f"[opt] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
