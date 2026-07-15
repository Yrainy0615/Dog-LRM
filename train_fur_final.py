#!/usr/bin/env python3
"""FINAL skin->fur: v6-flow strands (geometry FROZEN -> no grain) + body RECEDES to a dark
undercoat where fur covers (validated coupling needs an explicit recession prior). Fur carries
the coat, body = undercoat, composite ~ GT, and it's simulatable: when fur sways the undercoat
shows through the parting (physically correct). One process: train -> decomp -> ply -> sway.
"""
import argparse, json, os, sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, ".")
from dog_lrm.model import DogLRM
from dog_lrm.render import intrinsics, render_gaussians, save_ply
from dog_lrm.smal_model import SMALModel, load_pseudo_gt, subdivided_faces
from train_dog_lrm_ddp import _load_rgb_mask
from train_fur_v6flow import FurV6Flow

RK = ["means", "quats", "scales", "opacities", "rgb"]


def nn_argmin(A, B, chunk=4096):
    """For each row of A, index of nearest row in B (chunked -> bounded memory, robust to GPU contention)."""
    out = []
    for i in range(0, A.shape[0], chunk):
        out.append(torch.cdist(A[i:i+chunk], B).argmin(1))
    return torch.cat(out)


def visual_hull_keep(means, views, thr=0.5):
    """Keep a fur gaussian only if it projects INSIDE the GT mask in >= thr of the views it is visible
    in -> hard guarantee the fur silhouette does not spill past the mask (visual-hull clip)."""
    inm = torch.zeros(means.shape[0], device=means.device); cnt = torch.zeros_like(inm)
    for v in views:
        w2c = torch.inverse(v["c2w"]); cam = (w2c[:3, :3] @ means.T + w2c[:3, 3:4]).T; z = cam[:, 2]
        uv = (v["K"] @ (cam / z[:, None].clamp(min=1e-4)).T).T[:, :2]; u, vy = uv[:, 0], uv[:, 1]
        front = z > 1e-4; inb = front & (u >= 0) & (u < v["W"]) & (vy >= 0) & (vy < v["H"])
        grid = torch.stack([u / v["W"] * 2 - 1, vy / v["H"] * 2 - 1], -1)[None, :, None, :]
        m = F.grid_sample(v["mask"].permute(2, 0, 1)[None].float(), grid, align_corners=False)[0, 0, :, 0]
        inm += (inb & (m > 0.5)).float(); cnt += inb.float()
    return (inm / cnt.clamp(min=1)) >= thr


def comb_tangent(roots, t, n, k=16, iters=6):
    """Comb the strand tangent field: iterated neighbour-averaging on the surface (project to tangent
    plane, renormalize) -> locally coherent FLOW so fur looks combed/smooth, not messy (杂乱/不顺)."""
    from scipy.spatial import cKDTree
    R = roots.detach().cpu().numpy()
    _, idx = cKDTree(R).query(R, k=k)                              # [Nr,k] nearest roots
    idx = torch.from_numpy(idx).long().to(roots.device)
    tt = F.normalize(t, dim=-1)
    for _ in range(iters):
        tt = tt[idx].mean(1)                                      # average neighbour directions
        tt = tt - (tt * n).sum(-1, keepdim=True) * n              # keep tangent to surface
        tt = F.normalize(tt, dim=-1)
    return tt


def body_root_knn(body_means, roots, k=8, chunk=1024):
    """For each body anchor, k nearest fur roots + distances (chunked to bound memory)."""
    idxs, ds = [], []
    for i in range(0, body_means.shape[0], chunk):
        d = torch.cdist(body_means[i:i+chunk], roots)            # [c, Nr]
        dk, ik = torch.topk(d, min(k, roots.shape[0]), largest=False, dim=1)
        idxs.append(ik); ds.append(dk)
    return torch.cat(idxs), torch.cat(ds)


class PatchD(nn.Module):
    """PatchGAN discriminator on fur crops (v9): adversarial texture loss through the differentiable
    renderer -> sharp high-freq fur (fixes the soft/'粗糙' look that L1+LPIPS alone gives)."""
    def __init__(self, ch=64):
        super().__init__()
        nrm = lambda o: nn.GroupNorm(8, o)
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nrm(ch * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nrm(ch * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 4, 1, 4, 1, 1))

    def forward(self, x):
        return self.net(x * 2 - 1)


def fur_patches(render_hwc, gt_hwc, mask_hw, sz, n):
    """n square crops centered on fur pixels; same box for fake(render)/real(GT), masked to the dog."""
    H, W = render_hwc.shape[:2]; m = mask_hw[:, :, 0] > 0.5; ys, xs = torch.where(m)
    if len(ys) < 50: return None, None
    rm = render_hwc * mask_hw; gm = gt_hwc * mask_hw; fakes, reals = [], []
    for _ in range(n):
        i = int(torch.randint(len(ys), (1,))); cy, cx = int(ys[i]), int(xs[i])
        y0 = min(max(cy - sz // 2, 0), max(H - sz, 0)); x0 = min(max(cx - sz // 2, 0), max(W - sz, 0))
        f = rm[y0:y0 + sz, x0:x0 + sz].permute(2, 0, 1)[None]; r = gm[y0:y0 + sz, x0:x0 + sz].permute(2, 0, 1)[None]
        if f.shape[-2:] != (sz, sz):
            f = F.interpolate(f, (sz, sz), mode="bilinear", align_corners=False); r = F.interpolate(r, (sz, sz), mode="bilinear", align_corners=False)
        fakes.append(f); reals.append(r)
    return torch.cat(fakes), torch.cat(reals)


def prep(dog, root, dev, s, uniform_n=0, len_floor=0.03, shrink=0.0, comb_iters=0, face_keep_thr=0.3, face_collar=0.0, head_clear=0.0, head_r=0.12, head_body=0.0):
    scene = os.path.join(root, dog, "colmap")
    smal = SMALModel(dev, n_subdiv=2)
    model = DogLRM(gaussians_per_point=1).to(dev).eval()
    model.load_state_dict(torch.load("/tmp/stage1_final.pt", map_location=dev), strict=False)
    for p in model.parameters(): p.requires_grad = False
    gt = load_pseudo_gt(scene, "preprocess", smal.num_betas, dev)
    canon = smal.canonical_verts(gt["betas"], gt["limbs"]); posed = smal.posed_verts(gt["betas"], gt["limbs"], gt["theta"], gt["trans"], gt["scale"])
    frames = json.load(open(os.path.join(scene, "preprocess", "cameras.json")))["frames"]
    rr, _, _, _ = _load_rgb_mask(scene, frames[0], s)
    ref = F.interpolate(torch.from_numpy(rr).permute(2, 0, 1)[None].to(dev), (224, 224), mode="bilinear", align_corners=False)
    with torch.no_grad():
        bf = {k: v[0].detach() for k, v in model(ref, canon, posed, subdivide=smal.subdivide).items()}
    body = {k: bf[k] for k in RK}
    fa = np.load(os.path.join(scene, "preprocess", "fur_anchors.npz")); t = lambda k: torch.from_numpy(fa[k]).to(dev).float()
    dfaces = torch.from_numpy(np.load(os.path.join(scene, "preprocess", "dsmal_anchors.npz"))["faces"]).long().to(dev)
    faces_sub = subdivided_faces(dfaces, 1).to(dev)
    from dog_lrm.smal_model import build_subdiv                      # template paw/pad mask (SMAL skinning joints) -> per-anchor, no strand on paws
    _W = smal.smal.weights; Wsk = (_W if torch.is_tensor(_W) else torch.tensor(_W)).to(dev).float()
    paw_base = (Wsk[:, [10, 14, 20, 24]].sum(1) > 0.4).float()       # [Nbase] paws + pads
    paw_anchor = torch.sparse.mm(build_subdiv(dfaces, 1, dev), paw_base[:, None])[:, 0]   # [15550] aligned with w_face
    diag = float(fa["diag"]); g = t("gravity"); curl_id = int(fa["curl_id"])
    if uniform_n > 0:                                               # DENSE uniform roots (cover everything except face)
        V15 = t("roots"); area = 0.5 * torch.cross(V15[faces_sub[:, 1]]-V15[faces_sub[:, 0]],
                                                   V15[faces_sub[:, 2]]-V15[faces_sub[:, 0]], dim=-1).norm(dim=-1)
        fid = torch.multinomial(area, uniform_n, replacement=True)
        uu = torch.rand(uniform_n, 2, device=dev); fl = uu.sum(1) > 1; uu[fl] = 1-uu[fl]
        ba = torch.stack([1-uu[:, 0]-uu[:, 1], uu[:, 0], uu[:, 1]], 1); idx = faces_sub[fid]
        itp = lambda V: (V[idx]*ba).sum(1) if V.dim() == 1 else (V[idx]*ba[..., None]).sum(1)
        roots = itp(V15); tt = F.normalize(itp(t("t")), dim=-1); n = F.normalize(itp(t("n")), dim=-1)
        L = itp(t("L")).clamp(min=len_floor*diag)                   # floor -> fur even in low-density areas
        w_face = itp(t("w_face")); w_ear = itp(t("w_ear")); w_paw = itp(paw_anchor); phase = torch.rand(uniform_n, device=dev)*6.283
        keep = w_face < face_keep_thr                               # drop the face/head (bigger thr-region -> more bald)
        roots, tt, n, L, w_face, w_ear, w_paw, phase = roots[keep], tt[keep], n[keep], L[keep], w_face[keep], w_ear[keep], w_paw[keep], phase[keep]
        print(f"[prep] uniform roots: {int(keep.sum())}/{uniform_n} (face dropped)", flush=True)
    else:
        rf = torch.from_numpy(fa["root_face"]).long().to(dev); ba = t("root_bary"); idx = faces_sub[rf]
        itp = lambda V: (V[idx]*ba).sum(1) if V.dim() == 1 else (V[idx]*ba[..., None]).sum(1)
        roots = itp(t("roots")); tt = F.normalize(itp(t("t")), dim=-1); n = F.normalize(itp(t("n")), dim=-1)
        L = itp(t("L")); w_face = itp(t("w_face")); w_ear = itp(t("w_ear")); w_paw = itp(paw_anchor); phase = t("root_phase")
    if comb_iters > 0:                                              # comb the tangent flow -> smooth, not messy
        tt = comb_tangent(roots, tt, n, iters=comb_iters)
    b = F.normalize(torch.cross(n, tt, dim=-1), dim=-1)
    nofur = ((w_face > min(0.4, face_keep_thr)) | (w_ear > 0.3) | (w_paw > 0.3)).float()   # NO strand on face / ears / paws+pads
    if face_collar > 0:                                             # short collar near the face -> fur can't drape over / occlude the head
        face_vic = (w_face / max(face_keep_thr, 1e-6)).clamp(0, 1)  # ~1 at the bald boundary, 0 away from face
        L = L * (1 - face_collar * face_vic)
    if head_clear > 0:                                              # head clearance along the HEAD-TAIL AXIS (covers neck/jaw, not just the face sphere)
        R0 = t("roots"); wf0 = t("w_face")
        fc = (R0 * wf0[:, None]).sum(0) / wf0.sum().clamp(min=1); bc = R0.mean(0)
        axis = F.normalize(fc - bc, dim=0); hpj = ((fc - bc) * axis).sum()     # head end along the axis
        proj = ((roots - bc) * axis).sum(-1)                       # per-root position along head-tail axis
        hp = ((proj - (hpj - head_r * diag)) / (head_r * diag)).clamp(0, 1)    # 1 at head end, ramps to 0 head_r*diag back (covers neck/jaw)
        L = L * (1 - head_clear * hp)
        if head_body > 0:                                          # Stage-1 baked long coat into the BODY at the head/neck (fur is dropped there)
            projb = ((body["means"] - bc) * axis).sum(-1)
            hpb = ((projb - (hpj - head_r * diag)) / (head_r * diag)).clamp(0, 1)
            sc = body["scales"].clone(); mxv, mxi = sc.max(1)      # ANISOTROPIC: shrink only the LONGEST axis (the baked long-fur direction), keep the other two for surface coverage (no holes)
            sc.scatter_(1, mxi[:, None], (mxv * (1 - head_body * hpb))[:, None])
            body["scales"] = sc
            print(f"[prep] head_body: shortened long axis of {int((hpb>0.3).sum())} head/neck body gaussians", flush=True)
    albedo = body["rgb"][nn_argmin(roots, body["means"])].clamp(1e-3, 1-1e-3)
    # dense cover: a body anchor recedes if its NEAREST fur root is fur (not a face/nofur root)
    bnb = nn_argmin(body["means"], roots)
    cover = (nofur[bnb] < 0.5).float()
    if shrink > 0:                                                  # NeuralFur furless-body: shrink geom inward, fur fills the shell
        part = torch.where(nofur > 0.5, torch.tensor(0., device=dev),       # face: no shrink (stays Stage-1 skin)
                           torch.where(w_ear > 0.3, torch.tensor(0.4, device=dev), torch.tensor(1.0, device=dev)))  # ears thin, body full
        t_r = (shrink * diag) * part                               # per-root inward thickness
        roots = roots - n * t_r[:, None]                           # roots descend to the furless surface
        L = L + t_r                                                # extend length so tips still reach the original silhouette (L1 preserved)
        body["means"] = body["means"] - n[bnb] * (shrink * diag * cover)[:, None]  # body recedes inward only where fur covers
    return scene, frames, body, dict(roots=roots, t=tt, b=b, n=n, L=L, nofur=nofur, ear=w_ear, phase=phase,
                                     albedo=albedo, diag=diag, g=g, curl_id=curl_id, cover=cover)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", default="00031-itsuki")
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--scale_div", type=int, default=4)
    ap.add_argument("--lpips_w", type=float, default=0.15)
    ap.add_argument("--fur_op", type=float, default=0.6)
    ap.add_argument("--op_keep", type=float, default=0.05)
    ap.add_argument("--w_recede", type=float, default=0.6, help="strength of body->undercoat recession in fur regions")
    ap.add_argument("--uc_floor", type=float, default=0.3, help="darkest undercoat brightness (tinted, not black)")
    ap.add_argument("--cov_gate", type=int, default=1, help="1: undercoat darkens PROPORTIONAL to actual local fur opacity (no dark holes where fur is thin); 0: old binary region recession")
    ap.add_argument("--strand_only", type=int, default=0, help="NO undercoat route: drop body in fur region, strand carries the WHOLE appearance; body kept only on the FACE (no strand on face)")
    ap.add_argument("--freeze_root", type=int, default=0, help="freeze d_root only (roots stay exactly on the sampled surface point); direction/length/colour still learnable")
    ap.add_argument("--skin_opt", type=int, default=0, help="keep body as SKIN under the fur and optimize its colour together with the strands (body_rgb learnable, no darkening)")
    ap.add_argument("--skin_cov", type=float, default=0.0, help="skin_opt: fade body OPACITY where fur covers (fur-dense -> skin transparent so fur dominates; fur-sparse -> skin shows). 0=off, ~0.8")
    ap.add_argument("--cov_k", type=int, default=8, help="k nearest fur roots used to estimate per-body-anchor fur coverage")
    ap.add_argument("--radius_frac", type=float, default=0.0009)
    ap.add_argument("--tmix", type=float, default=0.88, help="tangent-vs-normal: HIGH = combed flat / hug skin (low = sticks out)")
    ap.add_argument("--off", type=float, default=0.06, help="offset-shell lift off surface (low = hug skin)")
    ap.add_argument("--len_scale", type=float, default=0.65, help="global fur-length scale (lower = shorter)")
    ap.add_argument("--Kp", type=int, default=6)
    ap.add_argument("--op_floor", type=float, default=0.15, help="min fur opacity (prevents curly collapse)")
    ap.add_argument("--w_sil", type=float, default=0.5, help="silhouette tightness: penalize fur alpha beyond GT mask")
    ap.add_argument("--hard_sil", type=int, default=0, help="1: visual-hull clip at export — drop fur gaussians outside the GT mask in most views (hard guarantee fur<=mask)")
    ap.add_argument("--sil_thr", type=float, default=0.5, help="hard_sil: keep a fur gaussian if inside-mask fraction over visible views >= this")
    ap.add_argument("--v2_params", default="", help="npz from v2_to_furparams.py: use v2 patch-diffusion measured (curl_amp,freq,droop) instead of VLM curl_id; per-region (multi-patch) npz interpolates per-root")
    ap.add_argument("--w_adv", type=float, default=0.0, help="(1) PatchGAN adversarial weight on fur crops -> sharper texture (fixes 粗糙); ~0.02-0.05")
    ap.add_argument("--adv_start", type=int, default=400, help="iter to start adversarial")
    ap.add_argument("--adv_patch", type=int, default=140, help="fur crop size (px)")
    ap.add_argument("--adv_n", type=int, default=3, help="fur crops per iter")
    ap.add_argument("--lr_d", type=float, default=2e-4, help="discriminator lr")
    ap.add_argument("--clump", type=float, default=0.0, help="(2) clump strands toward lock centers (curly fur); 0=off, 0.3-0.5 curly")
    ap.add_argument("--clump_n", type=int, default=400, help="(2) number of clump/lock centers")
    ap.add_argument("--layer2", type=float, default=0.0, help="(4) 2-layer coat: fraction of roots that are SHORT dense undercoat (rest = long guard hair); 0=off, ~0.5")
    ap.add_argument("--under_len", type=float, default=0.4, help="(4) undercoat length factor vs guard hair")
    ap.add_argument("--no_curl", type=int, default=0, help="drop curl -> soft straight strands (combed into clumps); curl_amp/freq=0")
    ap.add_argument("--droop_val", type=float, default=0.45, help="droop (gravity hang) when --no_curl")
    ap.add_argument("--freeze_geo", type=int, default=0, help="freeze strand GEOMETRY (d_root/d_dir/d_logL); optimize only appearance (op/colour/tone) of all gs together")
    ap.add_argument("--w_struct", type=float, default=0.0, help="strand STRUCTURE loss: neighbour direction-alignment + length-smoothness (combed/coherent fur, replaces adversarial); ~0.5-2.0")
    ap.add_argument("--struct_k", type=int, default=8, help="neighbours for strand structure loss")
    ap.add_argument("--w_geo", type=float, default=0.3, help="L2 reg keeping the unfrozen geometry (root/dir/len) mild")
    ap.add_argument("--w_col", type=float, default=0.1, help="L2 pulling fur colour toward GT-sample init (optimize but no rainbow)")
    ap.add_argument("--gt_color", type=int, default=1, help="sample fur colour from GT photos (multi-view) instead of Stage-1")
    ap.add_argument("--uniform_n", type=int, default=0, help=">0: dense uniform roots covering everything except face")
    ap.add_argument("--len_floor", type=float, default=0.03, help="min fur length (×diag) so low-density areas still grow fur")
    ap.add_argument("--shrink", type=float, default=0.0, help="NeuralFur furless-body: shrink body+roots inward by this fraction of diag (fur fills the shell); 0=off")
    ap.add_argument("--tip_fade", type=float, default=0.0, help="fade strand tip opacity (0=uniform, 0.6=tips 40%% opacity) -> softer silhouette, less spiky/竖起来")
    ap.add_argument("--comb_iters", type=int, default=0, help="iterations of tangent-field neighbour smoothing -> combed/flowing fur (not messy/杂乱); 0=off, 6=moderate")
    ap.add_argument("--face_keep_thr", type=float, default=0.3, help="drop fur roots with w_face above this -> bald face/head (lower = bigger bald region)")
    ap.add_argument("--face_collar", type=float, default=0.0, help="shorten fur as it approaches the face (0=off, 0.7=boundary fur 30%% length) -> no fur draping over/occluding the head")
    ap.add_argument("--head_clear", type=float, default=0.0, help="geometric head clearance: shorten fur within head_r of the face centroid (0=off, 0.8=strong) -> clears crown/forehead where w_face is low")
    ap.add_argument("--head_r", type=float, default=0.12, help="head-clearance axial length (×diag) back from the head end (covers neck/jaw)")
    ap.add_argument("--head_body", type=float, default=0.0, help="shrink Stage-1 BODY gaussians in the head/neck region (their baked long coat is the real head long-fur); 0=off, 0.7=strong")
    ap.add_argument("--eval_every", type=int, default=1000, help="periodic held-out L1 eval (ceiling probe)")
    ap.add_argument("--out", default="exps/fur_final")
    args = ap.parse_args()
    dev = "cuda"; s = args.scale_div; os.makedirs(args.out, exist_ok=True); white = torch.ones(3, device=dev)
    scene, frames, body, A = prep(args.dog, args.root, dev, s, uniform_n=args.uniform_n, len_floor=args.len_floor, shrink=args.shrink, comb_iters=args.comb_iters, face_keep_thr=args.face_keep_thr, face_collar=args.face_collar, head_clear=args.head_clear, head_r=args.head_r, head_body=args.head_body)
    body_rgb0 = body["rgb"].clone()
    cid = A["curl_id"]; Nr = A["roots"].shape[0]; curl_override = None
    if args.v2_params:                                              # (3) v2 patch-diffusion props replace VLM curl_id
        vp = np.load(args.v2_params)
        if "centers" in vp.files:                                  # per-region -> nearest-center interpolation to per-root
            cen = torch.tensor(vp["centers"], device=dev, dtype=torch.float32)
            near = nn_argmin(A["roots"].float(), cen).cpu().numpy()
            amp = torch.tensor(vp["curl_amp"][near], device=dev); frq = torch.tensor(vp["curl_freq"][near], device=dev); drp = torch.tensor(vp["droop"][near], device=dev)
            curl_override = (amp, frq, drp); cidv = float(np.median(vp["curl_amp"]))
            print(f"[final] v2 per-region: {len(cen)} regions -> per-root curl (amp~{float(amp.mean()):.2f})", flush=True)
        else:
            curl_override = (float(vp["curl_amp"]), float(vp["curl_freq"]), float(vp["droop"])); cidv = curl_override[0]
            print(f"[final] v2 params: amp={curl_override[0]:.2f} freq={curl_override[1]:.2f} droop={curl_override[2]:.2f}", flush=True)
        cid = 3 if cidv > 0.25 else 0
    if args.no_curl:                                               # soft STRAIGHT strands, no curl (combed into clumps)
        curl_override = (0.0, 0.0, float(args.droop_val)); cid = 0
        print(f"[final] no_curl: soft straight strands, droop={args.droop_val}", flush=True)
    Kp = args.Kp if args.no_curl else (10 if (cid in (3, 4) or args.clump > 0) else args.Kp)   # no_curl -> respect --Kp (e.g. 4 => 3 gs/strand)
    len_sc = args.len_scale * (0.7 if cid in (3, 4) else 1.0)
    L_in = A["L"] * len_sc; off_in = args.off                       # (4) 2-layer: short dense undercoat + long guard hair
    if args.layer2 > 0:
        und = torch.rand(Nr, device=dev) < args.layer2
        L_in = L_in * torch.where(und, torch.tensor(args.under_len, device=dev), torch.tensor(1.0, device=dev))
        off_in = torch.where(und, torch.tensor(args.off * 0.2, device=dev), torch.tensor(args.off, device=dev))
        print(f"[final] 2-layer: {int(und.sum())} undercoat + {int((~und).sum())} guard hair", flush=True)
    fur = FurV6Flow(A["roots"], A["t"], A["b"], A["n"], L_in, A["nofur"], A["ear"], A["phase"],
                    A["albedo"], A["diag"], A["g"], cid, Kp=Kp, radius_frac=args.radius_frac,
                    fur_op=args.fur_op, tmix=args.tmix, off=off_in, op_floor=args.op_floor, tip_fade=args.tip_fade,
                    curl_override=curl_override, clump_amt=args.clump, clump_n=args.clump_n).to(dev)
    if args.w_struct > 0:                                          # strand structure loss: precompute per-strand neighbours (on roots)
        from scipy.spatial import cKDTree
        _, sk = cKDTree(A["roots"].detach().cpu().numpy()).query(A["roots"].detach().cpu().numpy(), k=args.struct_k + 1)
        struct_knn = torch.from_numpy(sk[:, 1:]).long().to(dev)    # [Nr,k] (drop self)
    body_recede = nn.Parameter(torch.full((body["means"].shape[0],), 4.0, device=dev))
    # coverage gate: per-body-anchor fur coverage = opacity-weighted local fur density (precompute kNN once)
    if args.cov_gate:
        knn_idx, knn_d = body_root_knn(body["means"], A["roots"], k=args.cov_k)
        sigma = knn_d.median().clamp(min=1e-6)
        knn_w = torch.exp(-(knn_d ** 2) / (2 * sigma ** 2))     # near roots weigh more
        knn_wsum = knn_w.sum(1).clamp(min=1e-6)
        print(f"[final] cov_gate ON: body={body['means'].shape[0]} k={args.cov_k} sigma={float(sigma):.4f}", flush=True)
    # learn appearance + MILD geometry (root offset / dir / length), kept small by w_geo reg
    geo = [] if args.freeze_geo else ([fur.d_dir, fur.d_logL] if args.freeze_root else [fur.d_root, fur.d_dir, fur.d_logL])
    train_params = [fur.op_logit, fur.d_alb, fur.tone] + geo        # freeze_root: keep d_root frozen (roots on surface), dir/len/colour learnable
    if not args.cov_gate: train_params = train_params + [body_recede]   # cov_gate makes recession deterministic (no free param)
    body_dcol = nn.Parameter(torch.zeros_like(body_rgb0)) if args.skin_opt else None   # skin colour residual, optimized with strands
    if args.skin_opt: train_params = train_params + [body_dcol]
    opt = torch.optim.Adam(train_params, lr=args.lr)
    print(f"[final] {args.dog} curl_id={A['curl_id']} fur={A['roots'].shape[0]} cover={int(A['cover'].sum())} w_recede={args.w_recede}", flush=True)
    import lpips as Lp
    lpf = Lp.LPIPS(net="alex").to(dev)
    for p in lpf.parameters(): p.requires_grad = False
    views = []
    for fr in frames:
        rgb, mask, Wd, Hd = _load_rgb_mask(scene, fr, s)
        views.append(dict(rgb=torch.from_numpy(rgb).to(dev), mask=torch.from_numpy(mask).to(dev),
                          K=intrinsics(fr["fx"]/s, fr["fy"]/s, fr["cx"]/s, fr["cy"]/s, dev),
                          c2w=torch.tensor(fr["c2w"], device=dev).float(), W=Wd, H=Hd))
    tr = [v for i, v in enumerate(views) if i % 5 != 0]; ev = [v for i, v in enumerate(views) if i % 5 == 0]

    if args.gt_color:                                               # sample true fur colour from GT photos (multi-view)
        with torch.no_grad():
            Rr = A["roots"]; Rn = A["n"]; acc = torch.zeros_like(Rr); cnt = torch.zeros(Rr.shape[0], device=dev)
            for v in views:
                w2c = torch.inverse(v["c2w"]); cam = (w2c[:3, :3] @ Rr.T + w2c[:3, 3:4]).T; z = cam[:, 2]
                uv = (v["K"] @ (cam/z[:, None].clamp(min=1e-4)).T).T[:, :2]
                u, vy = uv[:, 0], uv[:, 1]
                vis = (z > 0) & (u >= 0) & (u < v["W"]) & (vy >= 0) & (vy < v["H"])
                vdir = F.normalize(Rr - v["c2w"][:3, 3], dim=1); vis = vis & ((Rn*(-vdir)).sum(1) > 0.2)
                grid = torch.stack([u/v["W"]*2-1, vy/v["H"]*2-1], -1)[None, :, None, :]
                col = F.grid_sample(v["rgb"].permute(2, 0, 1)[None], grid, align_corners=False)[0, :, :, 0].T
                acc += col*vis[:, None].float(); cnt += vis.float()
            seen = cnt > 0; alb = (acc/cnt[:, None].clamp(min=1)).clamp(1e-3, 1-1e-3)
            alb[~seen] = fur.albedo0[~seen]; fur.albedo0.copy_(alb)
        print(f"[final] GT-colour sampled: {int(seen.sum())}/{Rr.shape[0]} roots visible", flush=True)

    def assemble():
        S = fur()
        if args.skin_opt:                                          # body = SKIN under the fur, colour optimized with strands; fur semi-transparent -> droop reveals skin (gravity look)
            brgb = (body_rgb0 + body_dcol).clamp(0, 1)
            bop = body["opacities"]
            if args.skin_cov > 0 and args.cov_gate:                # fade body opacity where fur covers -> fur dominates (no skin bleed-through), fur-sparse keeps skin
                op_root = fur.root_opacity()
                fcov = ((knn_w * op_root[knn_idx]).sum(1) / knn_wsum).clamp(0, 1)   # [Nbody] fur coverage
                bop = bop * (1 - args.skin_cov * fcov)
            bmult = torch.ones(body["means"].shape[0], 1, device=dev)
            bg = {**{k: body[k] for k in ["means", "quats", "scales"]}, "opacities": bop, "rgb": brgb}
            return bg, S, bmult
        if args.strand_only:                                       # strand carries the WHOLE appearance; body kept only on the face
            bop = body["opacities"] * (1 - A["cover"])             # cover=1 (non-face) -> drop body; cover=0 (face) -> keep skin
            bmult = torch.ones(body["means"].shape[0], 1, device=dev)
            bg = {**{k: body[k] for k in ["means", "quats", "scales"]}, "opacities": bop, "rgb": body_rgb0}
            return bg, S, bmult
        if args.cov_gate:
            # undercoat darkens only as much as fur actually covers the anchor -> thin-fur areas keep skin colour (no dark holes)
            op_root = fur.root_opacity()                       # [Nr] in [0,1]
            fcov = (knn_w * op_root[knn_idx]).sum(1) / knn_wsum  # [Nbody] opacity-weighted local fur coverage
            bmult = (1 - fcov * (1 - args.uc_floor)).clamp(args.uc_floor, 1.0)[:, None]
        else:
            bmult = (args.uc_floor + (1 - args.uc_floor) * torch.sigmoid(body_recede))[:, None]
        brgb = (body_rgb0 * bmult).clamp(0, 1)
        bg = {**{k: body[k] for k in ["means", "quats", "scales", "opacities"]}, "rgb": brgb}
        return bg, S, bmult

    def render(gs, v):
        return render_gaussians(gs["means"], gs["quats"], gs["scales"], gs["opacities"], gs["rgb"], v["c2w"], v["K"], v["W"], v["H"], bg=white)

    def comp(bg, S, v):
        g = {k: torch.cat([bg[k], S[k]]) for k in RK}; return render(g, v)

    def lp(a, b_):
        a = F.interpolate(a.permute(2, 0, 1)[None]*2-1, 256, mode="bilinear", align_corners=False)
        b_ = F.interpolate(b_.permute(2, 0, 1)[None]*2-1, 256, mode="bilinear", align_corners=False)
        return lpf(a, b_).mean()

    netD = optD = None                                              # (1) PatchGAN adversarial for sharp fur texture
    if args.w_adv > 0:
        netD = PatchD().to(dev); optD = torch.optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.5, 0.99))
        print(f"[final] adversarial ON w_adv={args.w_adv} start={args.adv_start} patch={args.adv_patch}", flush=True)
    for it in range(args.iters):
        v = tr[np.random.randint(len(tr))]; bg, S, bmult = assemble()
        rgb, alpha = comp(bg, S, v); gtw = v["rgb"]*v["mask"] + (1-v["mask"])*white
        loss_adv = torch.zeros((), device=dev); fake_p = None
        if netD is not None and it >= args.adv_start:
            fake_p, real_p = fur_patches(rgb, v["rgb"], v["mask"], args.adv_patch, args.adv_n)
            if fake_p is not None: loss_adv = -netD(fake_p).mean()
        loss_struct = torch.zeros((), device=dev)
        if args.w_struct > 0:                                       # strand structure: neighbour direction-alignment + length-smoothness (combed coherence)
            d0c = F.normalize(fur.d0 + torch.tanh(fur.d_dir) * 0.2, dim=-1)
            Lc = fur.L0 * fur.d_logL.exp()
            colc = (fur.albedo0 + torch.tanh(fur.d_alb) * 0.3)        # neighbour COLOUR smoothness -> kill per-gaussian colour speckle/noise
            loss_struct = (((d0c - d0c[struct_knn].mean(1)) ** 2).sum(-1).mean()
                           + ((Lc - Lc[struct_knn].mean(1)) ** 2).mean() / (fur.diag ** 2)
                           + ((colc - colc[struct_knn].mean(1)) ** 2).sum(-1).mean())
        loss = (F.l1_loss(rgb*v["mask"], v["rgb"]*v["mask"]) + F.l1_loss(alpha, v["mask"])
                + args.lpips_w*lp(rgb, gtw) + args.op_keep*(1-S["opacities"]).mean()
                + (0.0 if args.cov_gate else args.w_recede*(bmult[:, 0]*A["cover"]).sum()/A["cover"].sum().clamp(min=1))
                + args.w_sil*F.relu(alpha-v["mask"]).mean()         # fur must not spill past the GT silhouette
                + args.w_geo*(fur.d_root.pow(2).mean()+fur.d_dir.pow(2).mean()+fur.d_logL.pow(2).mean())   # keep geometry mild
                + args.w_col*fur.d_alb.pow(2).mean()                # colour: optimize but stay near GT-sample (no rainbow)
                + (args.w_col*body_dcol.pow(2).mean() if args.skin_opt else 0.0)   # skin colour residual reg (no speckle)
                + args.w_adv*loss_adv
                + args.w_struct*loss_struct)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        if fake_p is not None:                                      # discriminator step (hinge)
            optD.zero_grad(); d_real = netD(real_p.detach()); d_fake = netD(fake_p.detach())
            (F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()).backward(); optD.step()
        if it % 500 == 0: print(f"it{it:4d} loss={float(loss):.4f} fur_op={float(S['opacities'].mean()):.3f} undercoat_mult={float(bmult[A['cover']>0].mean()):.3f}", flush=True)
        if it % args.eval_every == 0 and it > 0:                     # ceiling probe: held-out L1 over training
            with torch.no_grad():
                bg2, S2, _ = assemble(); l1e = []
                for vv in ev:
                    rr, _ = comp(bg2, S2, vv); gw = vv["rgb"]*vv["mask"]+(1-vv["mask"])*white
                    l1e.append(float((rr.clamp(0, 1)-gw).abs().mean()))
            print(f"[eval] it{it} heldout_L1={np.mean(l1e):.4f}", flush=True)

    with torch.no_grad():
        bg, S, bmult = assemble(); l1 = []
        for v in ev:
            rgb, _ = comp(bg, S, v); gtw = v["rgb"]*v["mask"]+(1-v["mask"])*white; l1.append(float((rgb.clamp(0,1)-gtw).abs().mean()))
    uc = float(bmult[A["cover"] > 0].mean())
    print(f"[final] {args.dog} held-out L1={np.mean(l1):.4f} | fur_op={float(S['opacities'].mean()):.3f} | undercoat brightness={uc:.3f} (lower=darker undercoat)", flush=True)

    # ---- save (for animation), decomp, ply, sway (all in-process) ----
    with torch.no_grad():
        bg, S, bmult = assemble()
        if args.hard_sil:                                   # visual-hull clip: fur must not spill past the GT mask
            keep = visual_hull_keep(S["means"], views, args.sil_thr)
            S = {k: v[keep] for k, v in S.items()}
            print(f"[final] hard_sil: kept {int(keep.sum())}/{keep.numel()} fur gaussians inside mask (thr={args.sil_thr})", flush=True)
        torch.save({"fur_sd": fur.state_dict(), "body_recede": body_recede.detach(), "body": {k: body[k] for k in RK},
                    "diag": A["diag"], "curl_id": A["curl_id"], "Kp": Kp}, os.path.join(args.out, f"{args.dog}_final.pt"))
        # widest view
        best = None
        for fr in frames:
            _, mk, _, _ = _load_rgb_mask(scene, fr, s); yy, xx = np.where(mk[:, :, 0] > 0.5)
            if len(xx) < 50: continue
            ar = (xx.max()-xx.min())/max(yy.max()-yy.min(), 1)
            if best is None or ar > best[0]: best = (ar, fr)
        fr = best[1]; v = dict(K=intrinsics(fr["fx"]/s, fr["fy"]/s, fr["cx"]/s, fr["cy"]/s, dev), c2w=torch.tensor(fr["c2w"], device=dev).float(), W=fr["width"]//s, H=fr["height"]//s)
        rgbg, mask, _, _ = _load_rgb_mask(scene, fr, s)
        R = lambda gs: render(gs, v)[0].clamp(0, 1).cpu().numpy()
        s1 = R({**{k: body[k] for k in ["means", "quats", "scales", "opacities"]}, "rgb": body_rgb0})
        und = R(bg); fo = R(S); cp = comp(bg, S, v)[0].clamp(0, 1).cpu().numpy()
        m = mask[:, :, 0] > 0.5; ys, xs = np.where(m); cr = lambda a: a[max(ys.min()-10,0):ys.max()+10, max(xs.min()-10,0):xs.max()+10]
        h = min(cr(x).shape[0] for x in [s1, und, fo, cp, rgbg])
        Image.fromarray((np.concatenate([cr(x)[:h] for x in [s1, und, fo, cp, rgbg]], 1)*255).astype(np.uint8)).save(os.path.join(args.out, f"{args.dog}_decomp.png"))
        # ply (receded undercoat floored + fur)
        bsc = body["scales"].clamp(min=0.004)
        save_ply(os.path.join(args.out, f"{args.dog}.ply"), torch.cat([body["means"], S["means"]]), torch.cat([bsc, S["scales"]]),
                 torch.cat([body["quats"], S["quats"]]), torch.cat([body["opacities"], S["opacities"]]), torch.cat([bg["rgb"], S["rgb"]]))
        # sway: undercoat static, fur sways -> undercoat peeks through
        gv = A["g"]; e = torch.tensor([1., 0., 0.], device=dev)
        if abs(float((gv*e).sum())) > 0.9: e = torch.tensor([0., 1., 0.], device=dev)
        wind = F.normalize(torch.cross(gv, e, dim=0), dim=0)
        pad = int(0.05*max(ys.max()-ys.min(), xs.max()-xs.min())); crp = lambda a: a[max(ys.min()-pad,0):ys.max()+pad, max(xs.min()-pad,0):xs.max()+pad]
        fr_imgs = []
        for f in range(30):
            tt_ = f/30; fur.sway.copy_((0.5*torch.sin(torch.tensor(2*math.pi*tt_, device=dev)+1.5*A["phase"]))[:, None]*wind[None])
            Sf = fur(); fr_imgs.append((crp(comp(bg, Sf, v)[0].clamp(0,1).cpu().numpy())*255).astype(np.uint8))
        fur.sway.zero_()
        hh = min(a.shape[0] for a in fr_imgs)
        Image.fromarray(fr_imgs[0][:hh]).save(os.path.join(args.out, f"{args.dog}_sway.gif"), save_all=True,
            append_images=[Image.fromarray(a[:hh]) for a in fr_imgs[1:]], duration=60, loop=0)
    print(f"[final] saved {args.dog} decomp/ply/final.pt/sway.gif -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
