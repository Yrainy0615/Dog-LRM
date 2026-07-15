#!/usr/bin/env python3
"""Stage-2 Phase A — single-dog strand fur on a FROZEN Stage-1 body, big framework.

Pipeline (FUR_STAGE2_PLAN.md):
  Stage-1 (frozen) -> body Gaussians (subdiv-1) + per-anchor coat COLOR
  fur_anchors.npz  -> roots (barycentric on subdiv-1 faces) + per-vertex TBN/L/w_face
  color prior      -> root base color = barycentric-interp(Stage1.rgb)         (sec.1)
  strands          -> v6 thin-sliver geometry, per-root LEARNABLE params       (sec.2)
  region           -> nofur on face+paws -> strand opacity gated to 0          (sec.3)
  render body (+) strands -> L1+LPIPS+mask vs multi-view GT (per-scene optimise)

Deferred (next): curl/droop/gamma, undercoat darkening, L_rest, penetration, coat CE.
"""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.abspath("."))
from dog_lrm.model import DogLRM
from dog_lrm.render import intrinsics, render_gaussians, save_ply
from dog_lrm.smal_model import SMALModel, load_pseudo_gt, subdivided_faces
from train_dog_lrm_ddp import _load_rgb_mask
from train_fur_strand import quat_align_z


def vertex_normals(v, f):
    """Area-weighted vertex normals. v[V,3], f[F,3] -> [V,3]."""
    fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])   # [F,3] (||~2A)
    vn = torch.zeros_like(v)
    vn.index_add_(0, f.reshape(-1), fn.repeat_interleave(3, 0))
    return F.normalize(vn, dim=-1)


def interp(field, vidx, bary):
    """Barycentric interp of a per-vertex field to roots. field[V,...], vidx[N,3], bary[N,3]."""
    g = field[vidx]                                                      # [N,3,...]
    while bary.dim() < g.dim():
        bary = bary.unsqueeze(-1)
    return (g * bary).sum(1)


def sample_roots(verts, faces, N):
    """Area-weighted barycentric root sampling on a mesh -> (face_id[N], bary[N,3])."""
    v = verts[faces]                                                     # [F,3,3]
    area = 0.5 * torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1).norm(dim=1)
    fid = torch.multinomial(area, N, replacement=True)
    e = -torch.log(torch.rand(N, 3, device=verts.device).clamp_min(1e-8))  # uniform on the simplex
    return fid, e / e.sum(1, keepdim=True)


class Strands(nn.Module):
    """Per-root LEARNABLE strand params -> Kp-1 thin-sliver Gaussians per strand."""
    def __init__(self, root_pos, root_n, root_L, nofur, albedo0, diag, gravity,
                 Kp=8, radius_frac=0.0032, offset_shell=0.2, len_scale=1.0, droop_bias=0.5,
                 op_init=1.25, curl_amp=0.0, curl_freq=2.0):
        super().__init__()
        N = root_pos.shape[0]
        self.Kp, self.rf, self.shell, self.diag = Kp, radius_frac, offset_shell, float(diag)
        self.len_scale, self.droop_bias = len_scale, droop_bias
        self.curl_amp, self.curl_freq = curl_amp, curl_freq            # v6 helical curl (fur clumps)
        root_b = F.normalize(torch.cross(root_n, gravity.expand_as(root_n), dim=-1), dim=-1)  # surface tangent
        root_phase = torch.rand(N, device=root_pos.device) * 6.2831853  # per-root curl phase
        for k, v in dict(root_pos=root_pos, root_n=root_n, root_L=root_L, nofur=nofur,
                         albedo0=albedo0, gravity=gravity, root_b=root_b, root_phase=root_phase).items():
            self.register_buffer(k, v)
        self.register_buffer("sway_vec", torch.zeros(N, 3))           # animation drive (wind sway), 0 at train
        self.dir_delta = nn.Parameter(torch.zeros(N, 3))
        self.log_len = nn.Parameter(torch.zeros(N))                      # len_mult=0.6+0.8*sig(0)=1
        self.op_logit = nn.Parameter(torch.full((N,), float(op_init)))  # op=0.9*sig(op_init)
        self.log_radius = nn.Parameter(torch.zeros(N))
        self.d_albedo = nn.Parameter(torch.zeros(N, 3))                  # color residual on the prior
        self.droop_logit = nn.Parameter(torch.zeros(N))                  # droop=sig(0+bias) ~0.62
        self.gamma_logit = nn.Parameter(torch.zeros(N))                  # gamma=1.5 (rise-then-flop)

    def forward(self):
        N, Kp, S = self.root_pos.shape[0], self.Kp, self.Kp - 1
        d0 = F.normalize(self.root_n + torch.tanh(self.dir_delta) * 0.3 + self.sway_vec, dim=-1)  # +wind sway
        L = (self.root_L * self.len_scale * (0.6 + 0.8 * torch.sigmoid(self.log_len)))[:, None]  # [N,1]
        r = self.rf * self.diag * (3.0 ** torch.tanh(self.log_radius))
        op = 0.9 * torch.sigmoid(self.op_logit) * (1.0 - self.nofur)                # killed on face/paw
        albedo = (self.albedo0 + torch.tanh(self.d_albedo) * 0.1).clamp(0, 1)       # prior + residual
        droop = torch.sigmoid(self.droop_logit + self.droop_bias)[:, None]         # [N,1]
        gamma = (1.5 * (3.0 ** torch.tanh(self.gamma_logit)))[:, None]             # [N,1] in [.5,4.5]
        g = self.gravity.view(1, 3)
        u2 = F.normalize(torch.cross(d0, self.root_b, dim=-1), dim=-1)             # curl plane axis
        # v6 grooming: strand rises along d0 at the root, flops toward gravity at the tip
        origin = self.root_pos + (self.shell * L) * self.root_n                    # offset-shell lift
        pts = [origin]; pcur = origin
        for k in range(1, Kp):
            s = k / (Kp - 1)
            beta = droop * (s ** gamma)                                            # [N,1] 0@root -> droop@tip
            dk = F.normalize((1 - beta) * d0 + beta * g, dim=-1)
            pcur = pcur + dk * (L / (Kp - 1))
            if self.curl_amp > 0:                                                  # v6 helical curl -> clumps
                th = (2 * 3.1415927 * self.curl_freq * s) + self.root_phase        # [N]
                curl = self.curl_amp * L * s * (th.sin()[:, None] * self.root_b
                                                + th.cos()[:, None] * u2 - u2)
                pts.append(pcur + curl)
            else:
                pts.append(pcur)
        pts = torch.stack(pts, 1)                                                  # [N,Kp,3]
        mid = 0.5 * (pts[:, 1:] + pts[:, :-1])
        seg = pts[:, 1:] - pts[:, :-1]
        slen = seg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sdir = seg / slen
        rxy = r.view(N, 1, 1).expand(N, S, 1)
        scales = torch.cat([rxy.expand(N, S, 2), slen * 0.5], dim=-1)               # thin prolate sliver
        quats = quat_align_z(sdir.reshape(N * S, 3)).reshape(N, S, 4)
        ops = op[:, None].expand(N, S)
        rgb = albedo[:, None, :].expand(N, S, 3)
        return dict(means=mid.reshape(N * S, 3), scales=scales.reshape(N * S, 3),
                    quats=quats.reshape(N * S, 4), opacities=ops.reshape(N * S),
                    rgb=rgb.reshape(N * S, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", default="00010-hanabi")
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--stage1_ckpt", default="exps/dog_lrm_stage1_lpips/model.pt")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--n_root", type=int, default=40000, help="subsample roots (<=40000)")
    ap.add_argument("--Kp", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--scale_div", type=int, default=8)
    ap.add_argument("--lpips_weight", type=float, default=0.1)
    ap.add_argument("--nofur_thr", type=float, default=0.3)
    ap.add_argument("--len_frac", type=float, default=0.045, help="strand length as fraction of body diag")
    ap.add_argument("--len_scale", type=float, default=1.0, help="extra global length factor")
    ap.add_argument("--vis_every", type=int, default=250)
    ap.add_argument("--out", default="exps/fur_stage2_dbg")
    args = ap.parse_args()
    dev = "cuda"; s = args.scale_div
    os.makedirs(args.out, exist_ok=True)
    scene = os.path.join(args.root, args.dog, "colmap")

    # --- Stage-1 frozen body (subdiv-1 to match fur_anchors) ---
    smal = SMALModel(dev, n_subdiv=2)
    model = DogLRM(gaussians_per_point=1).to(dev).eval()
    model.load_state_dict(torch.load(args.stage1_ckpt, map_location=dev), strict=False)
    for p in model.parameters():
        p.requires_grad = False
    gt_fit = load_pseudo_gt(scene, "preprocess", smal.num_betas, dev)
    canon = smal.canonical_verts(gt_fit["betas"], gt_fit["limbs"])
    posed = smal.posed_verts(gt_fit["betas"], gt_fit["limbs"], gt_fit["theta"],
                             gt_fit["trans"], gt_fit["scale"])
    frames = json.load(open(os.path.join(scene, "preprocess", "cameras.json")))["frames"]
    rgb_r, _, _, _ = _load_rgb_mask(scene, frames[0], s)                # reference view -> body
    ref = F.interpolate(torch.from_numpy(rgb_r).permute(2, 0, 1)[None].to(dev),
                        (224, 224), mode="bilinear", align_corners=False)
    with torch.no_grad():
        body = model(ref, canon, posed, subdivide=smal.subdivide)       # frozen undercoat + color
    body = {k: v[0].detach() for k, v in body.items()}

    # --- self-consistent roots + masks on MY Stage-1 mesh (D-SMAL skinning) ---
    faces1 = subdivided_faces(smal.faces, 2).to(dev)                    # [F2,3] -> 62k verts (match Stage-1)
    posed1 = smal.subdivide(posed)[0]                                   # [15550,3] my posed surface
    vn = vertex_normals(posed1, faces1)
    W = smal.subdivide(smal.smal.weights[None].float())[0]              # [15550,35] subdivided skinning
    face_w = W[:, [16, 32]].sum(1)                                      # skull+muzzle (dsmal_region_masks)
    paw_w = W[:, [10, 14, 20, 24]].sum(1)                               # 4 paws (SMAL_configs:90-93)
    diag = float((posed1.max(0).values - posed1.min(0).values).norm())
    grav = F.normalize(posed1[paw_w > 0.4].mean(0) - posed1.mean(0), dim=0)   # down = body->paws
    fid, bary = sample_roots(posed1, faces1, args.n_root)               # area-weighted on my mesh
    vidx = faces1[fid]
    root_pos = interp(posed1, vidx, bary)
    root_n = F.normalize(interp(vn, vidx, bary), dim=-1)
    albedo0 = interp(body["rgb"], vidx, bary)                          # Stage-1 colour prior
    fr_ = interp(face_w, vidx, bary); pw_ = interp(paw_w, vidx, bary)
    nofur = ((fr_ > args.nofur_thr) | (pw_ > 0.4)).float()
    root_L = torch.full((args.n_root,), args.len_frac * diag, device=dev)
    print(f"{args.dog}: {args.n_root} roots | nofur {int(nofur.sum())} "
          f"(face {int((fr_>args.nofur_thr).sum())} paw {int((pw_>0.4).sum())}) | "
          f"len {args.len_frac*diag:.3f} diag {diag:.2f} | body {body['means'].shape[0]} g", flush=True)

    strands = Strands(root_pos, root_n, root_L, nofur, albedo0, diag, grav,
                      Kp=args.Kp, len_scale=args.len_scale).to(dev)
    opt = torch.optim.Adam(strands.parameters(), lr=args.lr)
    import lpips as L
    lpips_fn = L.LPIPS(net="alex").to(dev)
    for p in lpips_fn.parameters():
        p.requires_grad = False
    white = torch.ones(3, device=dev)

    # cache GT views
    views = []
    for fr in frames:
        rgb, mask, W, H = _load_rgb_mask(scene, fr, s)
        views.append(dict(rgb=torch.from_numpy(rgb).to(dev), mask=torch.from_numpy(mask).to(dev),
                          K=intrinsics(fr["fx"]/s, fr["fy"]/s, fr["cx"]/s, fr["cy"]/s, dev),
                          c2w=torch.tensor(fr["c2w"], device=dev).float(), W=W, H=H))

    def render(v):
        st = strands()
        means = torch.cat([body["means"], st["means"]])
        quats = torch.cat([body["quats"], st["quats"]])
        scales = torch.cat([body["scales"], st["scales"]])
        ops = torch.cat([body["opacities"], st["opacities"]])
        rgb = torch.cat([body["rgb"], st["rgb"]])
        return render_gaussians(means, quats, scales, ops, rgb, v["c2w"], v["K"], v["W"], v["H"], bg=white)

    for it in range(args.iters):
        v = views[np.random.randint(len(views))]
        rgb, alpha = render(v)
        loss_rgb = F.l1_loss(rgb * v["mask"], v["rgb"] * v["mask"])
        loss_mask = F.l1_loss(alpha, v["mask"])
        gt_w = v["rgb"] * v["mask"] + (1 - v["mask"]) * white
        r = F.interpolate(rgb.permute(2, 0, 1)[None] * 2 - 1, 256, mode="bilinear", align_corners=False)
        g = F.interpolate(gt_w.permute(2, 0, 1)[None] * 2 - 1, 256, mode="bilinear", align_corners=False)
        loss = loss_rgb + loss_mask + args.lpips_weight * lpips_fn(r, g).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(strands.parameters(), 1.0)
        opt.step()
        if it % args.vis_every == 0 or it == args.iters - 1:
            print(f"it{it:4d} loss={float(loss):.4f} rgb={float(loss_rgb):.4f} mask={float(loss_mask):.4f}",
                  flush=True)
            with torch.no_grad():
                vv = views[len(views) // 2]
                rr, _ = render(vv)
                pair = np.concatenate([(rr.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                                       (vv["rgb"].cpu().numpy() * 255).astype(np.uint8)], axis=1)
                Image.fromarray(pair).save(os.path.join(args.out, f"it{it:04d}.png"))

    with torch.no_grad():
        st = strands()
    save_ply(os.path.join(args.out, f"{args.dog}_fur.ply"),
             torch.cat([body["means"], st["means"]]), torch.cat([body["scales"], st["scales"]]),
             torch.cat([body["quats"], st["quats"]]), torch.cat([body["opacities"], st["opacities"]]),
             torch.cat([body["rgb"], st["rgb"]]))
    torch.save({k: v for k, v in strands.state_dict().items()}, os.path.join(args.out, "strands.pt"))
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
