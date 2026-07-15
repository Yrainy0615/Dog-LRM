#!/usr/bin/env python3
"""Sample the token-DiT fur diffusion on held-out grooms with 1/2/4 views; evaluate + visualize.

Metrics: per-family strand-point error (xdiag) vs the train-mean baseline, per view-count
(the single-vs-sparse-view axis). Viz: strand polylines projected over the reference image
+ dense-decoded strands as an OBJ polyline export (Blender/meshlab QA).

  python sample_fur_diff.py --ckpt exps/fur_diff/diff.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from PIL import Image
from train_strand_predictor import vnormals, tbn_frames, project
from train_fur_diff import DiT, Codec, build_geo, splat_depth, anchor_visibility, fps

dev = "cuda"


@torch.no_grad()
def ddim(net, cond, C, T, steps=50):
    feat, geo, glob = cond
    betas = torch.linspace(1e-4, 0.02, T, device=dev)
    ac = torch.cumprod(1 - betas, 0)
    z = torch.randn(1, feat.shape[0], C, device=dev)
    ts = torch.linspace(T - 1, 0, steps, device=dev).long()
    for i in range(steps):
        t = ts[i][None]
        a = ac[t][:, None, None]
        eps = net(z, t, feat[None], geo[None], glob[None])
        z0 = (z - (1 - a).sqrt() * eps) / a.sqrt()
        if i < steps - 1:
            a2 = ac[ts[i + 1]]
            z = a2.sqrt() * z0 + (1 - a2).sqrt() * eps
        else:
            z = z0
    return z[0]



@torch.no_grad()
def predict(net, cond, C, A):
    """single forward for --regress ckpts, DDIM otherwise"""
    if A.get("regress"):
        feat, geo, gl = cond
        z = torch.zeros(1, feat.shape[0], C, device=dev)
        return net(z, torch.zeros(1, device=dev, dtype=torch.long), feat[None], geo[None], gl[None])[0]
    return ddim(net, cond, C, A["T"])


def strand_stats(pts):
    """phase-invariant per-strand stats: length, mean direction, curliness (arc/chord)"""
    seg = torch.diff(pts, dim=1).norm(dim=-1)
    L = seg.sum(1)
    chord = (pts[:, -1] - pts[:, 0]).norm(dim=-1)
    d = torch.nn.functional.normalize(pts[:, -1] - pts[:, 0], dim=-1)
    return L, d, L / chord.clamp(min=1e-8)


def stats_err(pred_pts, gt_pts, diag=1.0):
    """returns (len_err_xdiag, dir_err_deg, curl_err) — pointwise-phase independent"""
    Lp, dp, cp = strand_stats(pred_pts)
    Lg, dg, cg = strand_stats(gt_pts)
    len_err = float((Lp - Lg).abs().mean() / diag)
    dir_err = float(torch.rad2deg(torch.acos((dp * dg).sum(-1).clamp(-1, 1))).mean())
    curl_err = float((cp - cg).abs().mean())
    return len_err, dir_err, curl_err


def dense_roots(V, Fc, lgeo_v, n, seed=0):
    """area x density weighted root sampling on faces -> pos, normal, tbn, z-interp weights"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Nrm = vnormals(V, Fc)
    tri = V[Fc]
    area = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1) / 2
    dens = lgeo_v[Fc].mean(1)
    w = (area * (dens > 1e-4)).cpu()
    fi = torch.multinomial(w.clamp(min=1e-12), n, replacement=True, generator=g).to(dev)
    r = torch.rand(n, 2, generator=g).to(dev)
    su = r[:, 0].sqrt()
    b = torch.stack([1 - su, su * (1 - r[:, 1]), su * r[:, 1]], -1)
    pos = (V[Fc[fi]] * b[..., None]).sum(1)
    nrm = F.normalize((Nrm[Fc[fi]] * b[..., None]).sum(1), dim=-1)
    return pos, nrm, tbn_frames(pos, nrm)


def interp_latent(zA, RA, pos, k=4):
    d = torch.cdist(pos, RA)
    dk, ik = d.topk(k, largest=False)
    w = 1.0 / dk.clamp(min=1e-6)
    w = w / w.sum(1, keepdim=True)
    return (zA[ik] * w[..., None]).sum(1)


def export_obj(path, pts):                                            # pts [N,K,3] polylines
    with open(path, "w") as f:
        for s in pts:
            for p in s:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        K = pts.shape[1]
        for i in range(pts.shape[0]):
            base = i * K + 1
            f.write("l " + " ".join(str(base + j) for j in range(K)) + "\n")


def overlay(img, K, c2w, strands, path, step=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.imshow(img)
    sel = strands[::step]
    uv = project(sel.reshape(-1, 3), K, c2w)[0].view(sel.shape[0], sel.shape[1], 2).cpu()
    for s in uv:
        ax.plot(s[:, 0], s[:, 1], lw=0.3, c="cyan", alpha=0.5)
    ax.set_axis_off(); fig.tight_layout(pad=0)
    fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exps/fur_diff_v2/diff.pt")
    ap.add_argument("--data", default="synth_fur/dataset")
    ap.add_argument("--template", default="synth_fur/blender_input.npz")
    ap.add_argument("--codec", default="synth_fur/strand_codec.npz")
    ap.add_argument("--view_counts", default="1,2,4")
    ap.add_argument("--dense_n", type=int, default=150000)
    ap.add_argument("--out", default="exps/fur_diff")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=dev)
    A = ck["args"]
    tz = np.load(args.template)
    V = torch.from_numpy(tz["verts"].astype(np.float32)).to(dev)
    Fc = torch.from_numpy(tz["faces"].astype(np.int64)).to(dev)
    Nrm = vnormals(V, Fc)
    TBN = tbn_frames(V, Nrm)
    diag = float((V.max(0).values - V.min(0).values).norm())
    lgeo_v = torch.from_numpy(tz["L_geo"].astype(np.float32)).to(dev)
    wear_v = torch.from_numpy(tz["w_ear"].astype(np.float32)).to(dev)
    aidx = ck["aidx"].to(dev)
    R, Rn, Rtbn = V[aidx], Nrm[aidx], TBN[aidx]
    lgeo, wear = lgeo_v[aidx] / max(float(lgeo_v.max()), 1e-6), wear_v[aidx]
    codec = Codec(args.codec)
    C = codec.C

    net = DiT(zdim=C, geo=ck["net"]["cgeo.weight"].shape[1], d=A["d"], depth=A["depth"], heads=max(A["d"] // 64, 4)).to(dev)
    net.load_state_dict(ck["net"]); net.eval()

    from transformers import AutoModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dino = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
    nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    nsd = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    res = A["res"]; ph = res // 14

    @torch.no_grad()
    def dfeat(img):
        x = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float()
        x = (F.interpolate(x, (res, res), mode="bilinear", align_corners=False) - nm) / nsd
        o = dino(pixel_values=x).last_hidden_state
        return o[:, 1:].transpose(1, 2).reshape(1, -1, ph, ph), o[0, 0]

    # train-mean baseline z + held-out grooms
    hc = A["heldout_color"]
    mean_z, cnt = 0, 0
    evl = []
    for gp in sorted(glob.glob(os.path.join(args.data, "*.npz"))):
        z = np.load(gp, allow_pickle=True)
        name = os.path.basename(gp)[:-4]
        strands = torch.from_numpy(z["strands"]).to(dev)
        roots = torch.from_numpy(z["roots"]).to(dev)
        nn_ = torch.cdist(R, roots).argmin(1)
        loc = torch.einsum("rij,rkj->rki", Rtbn.transpose(1, 2), strands[nn_] - R[:, None]) / diag
        z0 = codec.encode(loc)
        if int(name.split("_")[1]) == hc:
            evl.append((name, str(z["style"]), z0, z))
        else:
            mean_z = mean_z + z0; cnt += 1
    mean_z = mean_z / cnt
    print(f"[sample] heldout {len(evl)} grooms | baseline from {cnt}", flush=True)

    by = {}
    for name, style, z0gt, z in evl:
        Ks = torch.from_numpy(z["Ks"]).to(dev)
        c2ws = torch.from_numpy(z["c2ws"]).to(dev)
        gt_pts = codec.decode(z0gt)
        for k in [int(x) for x in args.view_counts.split(",")]:
            vids = list(range(0, 8, 8 // k))[:k]
            feats, viss, clss = [], [], []
            for vi in vids:
                img = np.array(Image.open(os.path.join(args.data, str(z["imgs"][vi]))).convert("RGB")) / 255.
                ri = img.shape[0]
                fm, cls = dfeat(img)
                uv, _ = project(R, Ks[vi], c2ws[vi])
                gu = (uv / ri * 2 - 1)[None, :, None, :]
                feats.append(F.grid_sample(fm, gu, align_corners=False)[0, :, :, 0].T)
                dmap = splat_depth(V, Ks[vi], c2ws[vi], ri, diag)
                viss.append(anchor_visibility(R, Ks[vi], c2ws[vi], dmap, ri, diag))
                clss.append(cls)
            v = torch.stack(viss).float(); f = torch.stack(feats)
            w = v[:, :, None]
            feat = (f * w).sum(0) / w.sum(0).clamp(min=1)
            visfrac = v.max(0).values
            feat = feat * visfrac[:, None]
            geo = build_geo(R, Rn, diag, visfrac)
            zs = predict(net, (feat, geo, torch.stack(clss).mean(0)), C, A)
            se = stats_err(codec.decode(zs), gt_pts)
            err = (codec.decode(zs) - gt_pts).norm(dim=-1).mean()
            base = (codec.decode(mean_z.to(dev)) - gt_pts).norm(dim=-1).mean()
            occ_err = (codec.decode(zs)[visfrac < 0.5] - gt_pts[visfrac < 0.5]).norm(dim=-1).mean() if (visfrac < 0.5).any() else err * 0
            by.setdefault((style, k), []).append((float(err), float(base), float(occ_err)) + se)
            if k == max(int(x) for x in args.view_counts.split(",")) or k == 1:
                # viz: dense decode + overlay on view 0
                pos, nrm, ptbn = dense_roots(V, Fc, lgeo_v, args.dense_n, seed=0)
                zd = interp_latent(zs, R, pos)
                loc = codec.decode(zd)
                world = pos[:, None] + torch.einsum("rij,rkj->rki", ptbn, loc * diag)
                img0 = np.array(Image.open(os.path.join(args.data, str(z["imgs"][0]))).convert("RGB")) / 255.
                overlay(img0, Ks[0], c2ws[0], world[::40], os.path.join(args.out, f"{name}_k{k}_overlay.png"))
                export_obj(os.path.join(args.out, f"{name}_k{k}_dense.obj"), world[::15].cpu().numpy())
        print(f"[sample] {name} done", flush=True)

    print("[sample] per (style,#views): pointwise | phase-invariant len/dir/curl", flush=True)
    for (st, k), vals in sorted(by.items()):
        v = np.array(vals)
        print(f"  {st:6s} k={k}: diff {v[:,0].mean():.4f} base {v[:,1].mean():.4f} occl {v[:,2].mean():.4f}"
              f" | len {v[:,3].mean():.4f} dir {v[:,4].mean():5.1f}deg curl {v[:,5].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
