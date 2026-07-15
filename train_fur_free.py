#!/usr/bin/env python3
"""FREE-Gaussian fur: seed the cloud from v6-flow strands, then FREELY optimize every attribute
(xyz / quat / scale / opacity / rgb) driven by pixels, while a 3D strand-FLOW structural loss keeps
the free Gaussians combed:
  - align  : each Gaussian's LONG axis -> combed tangent flow d0 (mod-pi)
  - aniso  : keep Gaussians prolate (sliver, reads as a strand) not round blobs
  - coh    : neighbour long-axes agree -> smooth flow, not messy
Body = frozen Stage-1 skin under the fur. kotori, ALL views train, gsplat render.
Rationale: strand geometry init is unreliable and can't align to real pixels under hard binding;
free the geometry (pixel-driven) but softly guide ORIENTATION with the robust analytic flow field.
"""
import argparse, json, os, sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, ".")
from dog_lrm.render import intrinsics, render_gaussians, save_ply
from train_fur_v6flow import FurV6Flow
from train_fur_final import prep, nn_argmin, visual_hull_keep
from train_dog_lrm_ddp import _load_rgb_mask

RK = ["means", "quats", "scales", "opacities", "rgb"]


def quat_to_rotmat(q):
    """q wxyz [M,4] (assumed normalized) -> rotation matrices [M,3,3] (columns = rotated basis)."""
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    return R


def long_axis(quats, log_scales):
    """World-frame direction of each Gaussian's LONGEST covariance axis [M,3] (unit)."""
    R = quat_to_rotmat(F.normalize(quats, dim=-1))
    a = log_scales.argmax(dim=1)                                # [M] index of the largest scale
    axis = R[torch.arange(R.shape[0], device=R.device), :, a]  # that column
    return F.normalize(axis, dim=-1, eps=1e-8)


def knn_idx(pos, k):
    """[M,k] indices of k nearest neighbours (excluding self) on a point set (cKDTree, cpu)."""
    from scipy.spatial import cKDTree
    P = pos.detach().cpu().numpy()
    _, idx = cKDTree(P).query(P, k=k + 1)
    return torch.from_numpy(idx[:, 1:]).long().to(pos.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", default="00085-kotori")
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--scale_div", type=int, default=4)
    # --- seed (v6-flow strand) ---
    ap.add_argument("--Kp", type=int, default=6)
    ap.add_argument("--radius_frac", type=float, default=0.0009)
    ap.add_argument("--fur_op", type=float, default=0.6)
    ap.add_argument("--tmix", type=float, default=0.88)
    ap.add_argument("--off", type=float, default=0.06)
    ap.add_argument("--len_scale", type=float, default=0.65)
    ap.add_argument("--comb_iters", type=int, default=6, help="comb the seed tangent flow -> smooth d0 target")
    ap.add_argument("--face_keep_thr", type=float, default=0.3)
    # --- free-gaussian lrs ---
    ap.add_argument("--lr_pos_frac", type=float, default=1.6e-4, help="means lr = this * diag (3DGS convention)")
    ap.add_argument("--lr_rot", type=float, default=1e-3)
    ap.add_argument("--lr_scale", type=float, default=5e-3)
    ap.add_argument("--lr_op", type=float, default=5e-2)
    ap.add_argument("--lr_col", type=float, default=2.5e-3)
    # --- photometric ---
    ap.add_argument("--lpips_w", type=float, default=0.15)
    ap.add_argument("--op_keep", type=float, default=0.05)
    ap.add_argument("--w_sil", type=float, default=0.5, help="fur alpha must not spill past GT mask")
    # --- structural (3D flow) ---
    ap.add_argument("--w_align", type=float, default=1.0, help="long-axis -> flow d0 (mod-pi)")
    ap.add_argument("--w_aniso", type=float, default=0.05, help="keep prolate (mid/max scale small)")
    ap.add_argument("--w_coh", type=float, default=0.3, help="neighbour long-axis agreement")
    ap.add_argument("--struct_k", type=int, default=8)
    ap.add_argument("--anneal_frac", type=float, default=0.4, help="structural weights ramp down over first frac of iters")
    ap.add_argument("--anneal_floor", type=float, default=0.2, help="structural weight floor after annealing")
    ap.add_argument("--requery", type=int, default=200, help="re-query flow target + rebuild kNN every N iters")
    ap.add_argument("--w_orient2d", type=float, default=0.0, help="(off; weak Gabor) 2D image-orientation loss")
    ap.add_argument("--w_col_coh", type=float, default=0.1, help="neighbour COLOUR smoothness -> kill per-gaussian speckle")
    # --- prune (clean floaters; no grow in v1.1) ---
    ap.add_argument("--prune_every", type=int, default=400, help="prune low-opacity/oversized gaussians every N iters (0=off)")
    ap.add_argument("--prune_op", type=float, default=0.05, help="drop gaussians with opacity below this")
    ap.add_argument("--prune_scale", type=float, default=0.05, help="drop gaussians with max scale > this * diag (floaters)")
    ap.add_argument("--hard_sil", type=int, default=1, help="final visual-hull clip: drop fur outside GT mask in most views")
    ap.add_argument("--sil_thr", type=float, default=0.5)
    ap.add_argument("--out", default="exps/fur_free")
    args = ap.parse_args()
    dev = "cuda"; s = args.scale_div; os.makedirs(args.out, exist_ok=True); white = torch.ones(3, device=dev)

    # ---- strand seed (reuse the whole v6-flow prep + groom) ----
    scene, frames, body, A = prep(args.dog, args.root, dev, s, comb_iters=args.comb_iters,
                                  face_keep_thr=args.face_keep_thr)
    seed = FurV6Flow(A["roots"], A["t"], A["b"], A["n"], A["L"] * args.len_scale, A["nofur"], A["ear"],
                     A["phase"], A["albedo"], A["diag"], A["g"], A["curl_id"], Kp=args.Kp,
                     radius_frac=args.radius_frac, fur_op=args.fur_op, tmix=args.tmix, off=args.off).to(dev)
    with torch.no_grad():
        S0 = seed()                                            # per-segment strand gaussians = the seed
    diag = A["diag"]
    # ---- free parameters (raw / pre-activation) in a dict store so prune can rebuild them ----
    P = dict(means=nn.Parameter(S0["means"].clone()),
             quats=nn.Parameter(F.normalize(S0["quats"], dim=-1).clone()),
             log_scales=nn.Parameter(S0["scales"].clamp_min(1e-8).log().clone()),
             logit_op=nn.Parameter(torch.logit(S0["opacities"].clamp(1e-4, 1 - 1e-4)).clone()),
             colors=nn.Parameter(S0["rgb"].clone()))
    LRS = dict(means=args.lr_pos_frac * diag, quats=args.lr_rot, log_scales=args.lr_scale,
               logit_op=args.lr_op, colors=args.lr_col)
    def build_opt():
        return torch.optim.Adam([{"params": [P[k]], "lr": LRS[k]} for k in P])
    opt = build_opt(); M = P["means"].shape[0]
    # ---- static surface flow field (roots, combed d0) for orientation supervision ----
    flow_pts = A["roots"].detach()                             # [Nr,3]
    flow_dir = F.normalize(seed.d0.detach(), dim=-1)           # [Nr,3] combed tmix*t+(1-tmix)*n
    dflow_tgt = flow_dir[nn_argmin(P["means"].detach(), flow_pts)]  # [M,3]
    nbr = knn_idx(P["means"].detach(), args.struct_k)          # [M,k]
    print(f"[free] {args.dog} seed gaussians M={M} diag={diag:.3f} body={body['means'].shape[0]}", flush=True)

    import lpips as Lp
    lpf = Lp.LPIPS(net="alex").to(dev)
    for p in lpf.parameters(): p.requires_grad = False
    views = []
    for fr in frames:
        rgb, mask, Wd, Hd = _load_rgb_mask(scene, fr, s)
        views.append(dict(rgb=torch.from_numpy(rgb).to(dev), mask=torch.from_numpy(mask).to(dev),
                          K=intrinsics(fr["fx"] / s, fr["fy"] / s, fr["cx"] / s, fr["cy"] / s, dev),
                          c2w=torch.tensor(fr["c2w"], device=dev).float(), W=Wd, H=Hd))

    def fur_gs():
        return dict(means=P["means"], quats=F.normalize(P["quats"], dim=-1), scales=P["log_scales"].exp(),
                    opacities=P["logit_op"].sigmoid(), rgb=P["colors"].clamp(0, 1))

    def comp(S, v):
        g = {k: torch.cat([body[k], S[k]]) for k in RK}
        return render_gaussians(g["means"], g["quats"], g["scales"], g["opacities"], g["rgb"],
                                v["c2w"], v["K"], v["W"], v["H"], bg=white)

    def lp(a, b_):
        a = F.interpolate(a.permute(2, 0, 1)[None] * 2 - 1, 256, mode="bilinear", align_corners=False)
        b_ = F.interpolate(b_.permute(2, 0, 1)[None] * 2 - 1, 256, mode="bilinear", align_corners=False)
        return lpf(a, b_).mean()

    def struct_loss():
        axis = long_axis(P["quats"], P["log_scales"])          # [M,3]
        l_align = (1 - (axis * dflow_tgt).sum(-1).abs()).mean()          # mod-pi alignment to flow
        sc = P["log_scales"].exp(); ss, _ = sc.sort(dim=1)
        l_aniso = (ss[:, 1] / ss[:, 2].clamp_min(1e-8)).mean()          # 2nd-largest / largest -> prolate
        nbr_axis = F.normalize(axis[nbr].mean(1), dim=-1, eps=1e-8)
        l_coh = (1 - (axis * nbr_axis).sum(-1).abs()).mean()           # neighbour agreement
        l_col = ((P["colors"] - P["colors"][nbr].mean(1)) ** 2).sum(-1).mean()   # neighbour colour smoothness -> kill speckle
        return l_align, l_aniso, l_coh, l_col

    def prune():
        nonlocal opt, dflow_tgt, nbr, M
        with torch.no_grad():
            op = P["logit_op"].sigmoid(); smax = P["log_scales"].exp().max(1).values
            keep = (op > args.prune_op) & (smax < args.prune_scale * diag)
        if bool(keep.all()): return
        for k in list(P):
            P[k] = nn.Parameter(P[k].detach()[keep].clone())
        opt = build_opt()                                      # momentum reset (only a handful of times)
        dflow_tgt = flow_dir[nn_argmin(P["means"].detach(), flow_pts)]
        nbr = knn_idx(P["means"].detach(), args.struct_k); M = P["means"].shape[0]
        print(f"[free]   prune -> {M} gaussians", flush=True)

    for it in range(args.iters):
        if it > 0 and args.prune_every > 0 and it % args.prune_every == 0 and it < 0.85 * args.iters:
            prune()
        if it > 0 and it % args.requery == 0:                  # positions drift -> refresh flow target + kNN
            with torch.no_grad():
                dflow_tgt = flow_dir[nn_argmin(P["means"].detach(), flow_pts)]
                nbr = knn_idx(P["means"].detach(), args.struct_k)
        frac = min(it / max(args.anneal_frac * args.iters, 1), 1.0)
        aw = 1.0 - (1.0 - args.anneal_floor) * frac            # structural weight anneal (strong early)
        v = views[np.random.randint(len(views))]
        S = fur_gs(); rgb, alpha = comp(S, v)
        gtw = v["rgb"] * v["mask"] + (1 - v["mask"]) * white
        l_align, l_aniso, l_coh, l_col = struct_loss()
        loss = (F.l1_loss(rgb * v["mask"], v["rgb"] * v["mask"]) + F.l1_loss(alpha, v["mask"])
                + args.lpips_w * lp(rgb, gtw) + args.op_keep * (1 - S["opacities"]).mean()
                + args.w_sil * F.relu(alpha - v["mask"]).mean() + args.w_col_coh * l_col
                + aw * (args.w_align * l_align + args.w_aniso * l_aniso + args.w_coh * l_coh))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), 1.0)
        opt.step()
        if it % 300 == 0:
            print(f"it{it:4d} loss={float(loss):.4f} align={float(l_align):.3f} coh={float(l_coh):.3f} "
                  f"aniso={float(l_aniso):.3f} col={float(l_col):.4f} aw={aw:.2f} op={float(S['opacities'].mean()):.3f} M={M}", flush=True)

    # ---- report (all-view train -> train-view fit metrics + intrinsic flow coherence) ----
    with torch.no_grad():
        S = fur_gs(); l1s, pss, lps = [], [], []
        for v in views:
            rgb, _ = comp(S, v); gtw = v["rgb"] * v["mask"] + (1 - v["mask"]) * white
            r = rgb.clamp(0, 1); l1s.append(float((r - gtw).abs().mean()))
            pss.append(float(-10 * math.log10(((r - gtw) ** 2).mean().clamp_min(1e-10))))
            lps.append(float(lp(r, gtw)))
        axis = long_axis(P["quats"], P["log_scales"])
        coh = float((axis * F.normalize(axis[nbr].mean(1), dim=-1, eps=1e-8)).sum(-1).abs().mean())
    print(f"[free] {args.dog} train-view L1={np.mean(l1s):.4f} PSNR={np.mean(pss):.2f} "
          f"LPIPS={np.mean(lps):.4f} | flow-coherence={coh:.4f} (higher=more combed)", flush=True)

    # ---- save: ply + decomp png (body / fur / composite / GT) on the widest view ----
    with torch.no_grad():
        S = fur_gs()
        if args.hard_sil:                                      # visual-hull clip: fur must not spill past the GT mask
            keep = visual_hull_keep(S["means"], views, args.sil_thr)
            S = {k: vv[keep] for k, vv in S.items()}
            print(f"[free] hard_sil: kept {int(keep.sum())}/{keep.numel()} fur gaussians inside mask", flush=True)
        save_ply(os.path.join(args.out, f"{args.dog}.ply"),
                 torch.cat([body["means"], S["means"]]), torch.cat([body["scales"], S["scales"]]),
                 torch.cat([body["quats"], S["quats"]]), torch.cat([body["opacities"], S["opacities"]]),
                 torch.cat([body["rgb"], S["rgb"]]))
        torch.save({k: P[k].detach() for k in P} | {"body": {k: body[k] for k in RK}, "diag": diag},
                   os.path.join(args.out, f"{args.dog}_free.pt"))
        best = None
        for fr in frames:
            _, mk, _, _ = _load_rgb_mask(scene, fr, s); yy, xx = np.where(mk[:, :, 0] > 0.5)
            if len(xx) < 50: continue
            ar = (xx.max() - xx.min()) / max(yy.max() - yy.min(), 1)
            if best is None or ar > best[0]: best = (ar, fr)
        fr = best[1]; v = dict(K=intrinsics(fr["fx"] / s, fr["fy"] / s, fr["cx"] / s, fr["cy"] / s, dev),
                               c2w=torch.tensor(fr["c2w"], device=dev).float(), W=fr["width"] // s, H=fr["height"] // s)
        rgbg, mask, _, _ = _load_rgb_mask(scene, fr, s)
        R = lambda gs: render_gaussians(gs["means"], gs["quats"], gs["scales"], gs["opacities"], gs["rgb"],
                                        v["c2w"], v["K"], v["W"], v["H"], bg=white)[0].clamp(0, 1).cpu().numpy()
        s1 = R({k: body[k] for k in RK}); fo = R(S)
        cp = comp(S, v)[0].clamp(0, 1).cpu().numpy()
        m = mask[:, :, 0] > 0.5; ys, xs = np.where(m)
        cr = lambda a: a[max(ys.min() - 10, 0):ys.max() + 10, max(xs.min() - 10, 0):xs.max() + 10]
        h = min(cr(x).shape[0] for x in [s1, fo, cp, rgbg])
        Image.fromarray((np.concatenate([cr(x)[:h] for x in [s1, fo, cp, rgbg]], 1) * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"{args.dog}_decomp.png"))
    print(f"[free] saved {args.dog} free.pt/ply/decomp -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
