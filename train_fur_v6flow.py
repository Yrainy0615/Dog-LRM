#!/usr/bin/env python3
"""skin->fur v6-flow, TRAINABLE. v6 combed-flow geometry as PRIOR (tmix*t+(1-tmix)*n, droop,
curl, offset-shell from fur_anchors.npz); optimise only appearance + thin shape:
  per-strand opacity, colour residual, root->tip brightness gradient (hair definition),
  global radius/length residual, small per-strand dir residual.
Body (Stage-1 skin GS) frozen. Colour inherited from nearest skin GS. Face fur killed by w_face.
"""
import argparse, json, os, sys, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, ".")
from dog_lrm.model import DogLRM
from dog_lrm.render import intrinsics, render_gaussians, save_ply
from dog_lrm.smal_model import SMALModel, load_pseudo_gt, subdivided_faces
from train_dog_lrm_ddp import _load_rgb_mask

RK = ["means", "quats", "scales", "opacities", "rgb"]
CURL = {2: dict(amp=0.15, freq=2.0, droop=0.6), 3: dict(amp=0.40, freq=3.5, droop=0.40),
        4: dict(amp=0.65, freq=4.5, droop=0.30), 0: dict(amp=0.0, freq=2.0, droop=0.3),
        1: dict(amp=0.05, freq=2.0, droop=0.35), 5: dict(amp=0.1, freq=2.0, droop=0.3)}   # 3/4 = tighter curl


def quat_align_z(d):
    z = torch.zeros_like(d); z[:, 2] = 1.0
    v = torch.cross(z, d, dim=-1); c = (z * d).sum(-1)
    w = torch.sqrt(((1 + c).clamp(min=1e-8)) / 2) * 2
    return F.normalize(torch.cat([(w * 0.5)[:, None], v / w.clamp(min=1e-8)[:, None]], -1), dim=-1)


class FurV6Flow(nn.Module):
    def __init__(self, roots, t, b, n, L, nofur, ear, phase, albedo, diag, g, curl_id,
                 Kp=6, radius_frac=0.0006, fur_op=0.7, tmix=0.75, off=0.2, op_floor=0.0, tip_fade=0.0, curl_override=None,
                 clump_amt=0.0, clump_n=160):
        super().__init__()
        Nr = roots.shape[0]; dev = roots.device
        self.Kp = Kp; self.diag = float(diag); self.rf = radius_frac; self.op_floor = op_floor; self.tip_fade = tip_fade
        if curl_override is not None:                                # v2 patch-diffusion measured (amp,freq,droop) [scalar OR per-root] instead of VLM curl_id
            ca, cf, cd = curl_override
        else:
            c = CURL.get(curl_id, CURL[0]); ca, cf, cd = c["amp"], c["freq"], c["droop"]
        def col(x):                                                  # -> [Nr,1] buffer (broadcast scalar or keep per-root)
            x = torch.as_tensor(x, dtype=torch.float32, device=dev)
            return (x.expand(Nr) if x.numel() == 1 else x).reshape(Nr, 1).contiguous()
        for nm, val in dict(ca=col(ca), cf=col(cf), cd=col(cd), offv=col(off)).items():
            self.register_buffer(nm, val)                            # ca/cf/cd = per-root curl amp/freq/droop; offv = per-root shell lift (2-layer)
        self.clump_amt = float(clump_amt)                           # clump: pull strands toward FPS cluster centers (curly fur forms locks)
        if clump_amt > 0:
            with torch.no_grad():
                sel = [0]; d2 = torch.full((Nr,), 1e18, device=dev)
                for _ in range(min(clump_n, Nr) - 1):
                    d2 = torch.minimum(d2, ((roots - roots[sel[-1]]) ** 2).sum(1)); sel.append(int(d2.argmax()))
                cen = roots[torch.tensor(sel, device=dev)]; asg = torch.cdist(roots, cen).argmin(1)
                cv = cen[asg] - roots; cv = cv - (cv * n).sum(1, keepdim=True) * n        # tangent pull toward cluster center
                cvn = cv.norm(dim=1, keepdim=True).clamp(min=1e-9); cv = cv / cvn * cvn.clamp(max=0.06 * float(diag))
            self.register_buffer("clump_vec", cv)
        tm = torch.full((Nr, 1), tmix, device=dev)
        tm = torch.where(ear[:, None] > 0.3, torch.tensor(0.3, device=dev), tm)   # ears lie flat
        d0 = F.normalize(tm * t + (1 - tm) * n, dim=-1)
        for k, v in dict(roots=roots, n=n, b=b, d0=d0, L0=L, nofur=nofur, phase=phase,
                         albedo0=albedo, g=g.expand_as(n).contiguous()).items():
            self.register_buffer(k, v)
        self.register_buffer("sway", torch.zeros_like(roots))
        op0 = math.log(fur_op / (1 - fur_op))
        self.op_logit = nn.Parameter(torch.full((Nr,), op0))
        self.d_alb = nn.Parameter(torch.zeros(Nr, 3))
        self.tone = nn.Parameter(torch.zeros(Nr))            # root->tip brightness gradient
        self.d_dir = nn.Parameter(torch.zeros(Nr, 3))        # small TBN dir residual
        self.d_root = nn.Parameter(torch.zeros(Nr, 3))       # per-root position offset (bounded, regularized)
        self.d_logL = nn.Parameter(torch.zeros(Nr))          # per-strand length residual
        self.d_logr = nn.Parameter(torch.zeros(1))

    def strand_points(self):
        """[Nr, Kp, 3] strand polyline points in world coords (geometry only, for mesh+strand export)."""
        Nr, Kp = self.roots.shape[0], self.Kp
        L = (self.L0 * self.d_logL.exp())[:, None]
        d0 = F.normalize(self.d0 + torch.tanh(self.d_dir) * 0.2 + self.sway, dim=-1)
        u2 = F.normalize(torch.cross(d0, self.b, dim=-1), dim=-1)
        root = self.roots + torch.tanh(self.d_root) * (0.04 * self.diag)
        origin = root + (self.offv * L) * self.n
        pts = [origin]; pcur = origin
        for k in range(1, Kp):
            sf = k / (Kp - 1); beta = self.cd * (sf ** 1.5)
            dk = F.normalize((1 - beta) * d0 + beta * self.g, dim=-1)
            pcur = pcur + dk * L / (Kp - 1)
            th = 2 * math.pi * self.cf * sf + self.phase[:, None]
            cl = (self.clump_amt * self.clump_vec * (sf ** 1.2)) if self.clump_amt > 0 else 0.0
            pts.append(pcur + self.ca * L * sf * (th.sin() * self.b + th.cos() * u2 - u2) + cl)
        return torch.stack(pts, 1)

    def forward(self):
        Nr, Kp = self.roots.shape[0], self.Kp
        L = (self.L0 * self.d_logL.exp())[:, None]
        r = self.rf * self.diag * self.d_logr.exp()
        d0 = F.normalize(self.d0 + torch.tanh(self.d_dir) * 0.2 + self.sway, dim=-1)
        u2 = F.normalize(torch.cross(d0, self.b, dim=-1), dim=-1)
        root = self.roots + torch.tanh(self.d_root) * (0.04 * self.diag)   # bounded per-root move
        origin = root + (self.offv * L) * self.n                       # offv per-root (2-layer: undercoat off~0, guard hair off>0)
        pts = [origin]; pcur = origin
        for k in range(1, Kp):
            sf = k / (Kp - 1); beta = self.cd * (sf ** 1.5)            # cd per-root droop
            dk = F.normalize((1 - beta) * d0 + beta * self.g, dim=-1)
            pcur = pcur + dk * L / (Kp - 1)
            th = 2 * math.pi * self.cf * sf + self.phase[:, None]      # cf per-root curl freq
            cl = (self.clump_amt * self.clump_vec * (sf ** 1.2)) if self.clump_amt > 0 else 0.0   # clump toward lock center at tips
            pts.append(pcur + self.ca * L * sf * (th.sin() * self.b + th.cos() * u2 - u2) + cl)   # ca per-root curl amp
        pts = torch.stack(pts, 1)
        seg = pts[:, 1:] - pts[:, :-1]; mid = 0.5 * (pts[:, 1:] + pts[:, :-1])
        slen = seg.norm(dim=-1, keepdim=True).clamp(min=1e-6); sdir = seg / slen
        S = Kp - 1
        scales = torch.cat([torch.full((Nr, S, 2), float(r), device=mid.device), slen * 0.5], -1)
        quats = quat_align_z(sdir.reshape(Nr * S, 3)).reshape(Nr, S, 4)
        op_root = ((0.9 * torch.sigmoid(self.op_logit)).clamp(min=self.op_floor) * (1 - self.nofur))[:, None]  # [Nr,1]
        if self.tip_fade > 0:                                          # fade strand TIPS -> softer silhouette (less spiky/竖起来)
            tseg = torch.linspace(0, 1, S, device=op_root.device)[None, :]
            op = op_root * (1 - self.tip_fade * tseg)
        else:
            op = op_root.expand(Nr, S)
        base = (self.albedo0 + torch.tanh(self.d_alb) * 0.3).clamp(0, 1)             # [Nr,3] per-strand colour, multi-view optimized (init=GT sample)
        sfrac = torch.linspace(0, 1, Kp - 1, device=mid.device).view(1, S, 1)
        bright = (1 + torch.tanh(self.tone).view(Nr, 1, 1) * (sfrac - 0.5) * 0.25)   # subtle root->tip (less speckle)
        rgb = (base[:, None, :] * bright).clamp(0, 1)
        return dict(means=mid.reshape(Nr*S, 3), scales=scales.reshape(Nr*S, 3), quats=quats.reshape(Nr*S, 4),
                    opacities=op.reshape(Nr*S), rgb=rgb.reshape(Nr*S, 3))

    def root_opacity(self):
        """per-root effective opacity in [0,1] (same gate used per segment in forward)."""
        return ((0.9 * torch.sigmoid(self.op_logit)).clamp(min=self.op_floor) * (1 - self.nofur))


def itp_fn(idx, ba):
    def f(V):
        return (V[idx] * ba).sum(1) if V.dim() == 1 else (V[idx] * ba[..., None]).sum(1)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", default="00031-itsuki")
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--stage1_ckpt", default="/tmp/stage1_final.pt")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=8e-3)
    ap.add_argument("--scale_div", type=int, default=4)
    ap.add_argument("--lpips_w", type=float, default=0.15)
    ap.add_argument("--op_keep", type=float, default=0.02)
    ap.add_argument("--Kp", type=int, default=6)
    ap.add_argument("--radius_frac", type=float, default=0.0006)
    ap.add_argument("--out", default="exps/fur_v6flow")
    args = ap.parse_args()
    dev = "cuda"; s = args.scale_div; os.makedirs(args.out, exist_ok=True)
    scene = os.path.join(args.root, args.dog, "colmap"); white = torch.ones(3, device=dev)

    smal = SMALModel(dev, n_subdiv=2)
    model = DogLRM(gaussians_per_point=1).to(dev).eval()
    model.load_state_dict(torch.load(args.stage1_ckpt, map_location=dev), strict=False)
    for p in model.parameters(): p.requires_grad = False
    gt = load_pseudo_gt(scene, "preprocess", smal.num_betas, dev)
    canon = smal.canonical_verts(gt["betas"], gt["limbs"]); posed = smal.posed_verts(gt["betas"], gt["limbs"], gt["theta"], gt["trans"], gt["scale"])
    frames = json.load(open(os.path.join(scene, "preprocess", "cameras.json")))["frames"]
    rr, _, _, _ = _load_rgb_mask(scene, frames[0], s)
    ref = F.interpolate(torch.from_numpy(rr).permute(2, 0, 1)[None].to(dev), (224, 224), mode="bilinear", align_corners=False)
    with torch.no_grad():
        bf = {k: v[0].detach() for k, v in model(ref, canon, posed, subdivide=smal.subdivide).items()}
    body = {k: bf[k] for k in RK}

    fa = np.load(os.path.join(scene, "preprocess", "fur_anchors.npz")); tt = lambda k: torch.from_numpy(fa[k]).to(dev).float()
    dfaces = torch.from_numpy(np.load(os.path.join(scene, "preprocess", "dsmal_anchors.npz"))["faces"]).long().to(dev)
    faces_sub = subdivided_faces(dfaces, 1).to(dev)
    rf = torch.from_numpy(fa["root_face"]).long().to(dev); ba = tt("root_bary"); idx = faces_sub[rf]; itp = itp_fn(idx, ba)
    roots = itp(tt("roots")); t = F.normalize(itp(tt("t")), dim=-1); n = F.normalize(itp(tt("n")), dim=-1)
    b = F.normalize(torch.cross(n, t, dim=-1), dim=-1)
    L = itp(tt("L")); diag = float(fa["diag"]); g = tt("gravity"); curl_id = int(fa["curl_id"])
    w_face = itp(tt("w_face")); w_ear = itp(tt("w_ear")); phase = tt("root_phase")
    nofur = (w_face > 0.4).float(); L = L * torch.where(w_ear > 0.3, torch.tensor(0.35, device=dev), torch.tensor(1.0, device=dev))
    albedo = body["rgb"][torch.cdist(roots, body["means"]).argmin(1)].clamp(0, 1)
    print(f"[v6flow-train] {args.dog} curl_id={curl_id} roots={roots.shape[0]} nofur={int(nofur.sum())}", flush=True)

    st = FurV6Flow(roots, t, b, n, L, nofur, w_ear, phase, albedo, diag, g, curl_id,
                   Kp=args.Kp, radius_frac=args.radius_frac).to(dev)
    opt = torch.optim.Adam(st.parameters(), lr=args.lr)
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

    def comp(S, v):
        gs = {k: torch.cat([body[k], S[k]]) for k in RK}
        return render_gaussians(gs["means"], gs["quats"], gs["scales"], gs["opacities"], gs["rgb"], v["c2w"], v["K"], v["W"], v["H"], bg=white)

    def lp(a, b_):
        a = F.interpolate(a.permute(2, 0, 1)[None]*2-1, 256, mode="bilinear", align_corners=False)
        b_ = F.interpolate(b_.permute(2, 0, 1)[None]*2-1, 256, mode="bilinear", align_corners=False)
        return lpf(a, b_).mean()

    for it in range(args.iters):
        v = tr[np.random.randint(len(tr))]; S = st(); rgb, alpha = comp(S, v)
        gtw = v["rgb"]*v["mask"] + (1-v["mask"])*white
        loss = (F.l1_loss(rgb*v["mask"], v["rgb"]*v["mask"]) + F.l1_loss(alpha, v["mask"])
                + args.lpips_w*lp(rgb, gtw) + args.op_keep*(1-S["opacities"]).mean())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0); opt.step()
        if it % 500 == 0: print(f"it{it:4d} loss={float(loss):.4f} fur_op={float(S['opacities'].mean()):.3f}", flush=True)

    with torch.no_grad():
        S = st(); l1 = []
        for v in ev:
            rgb, _ = comp(S, v); gtw = v["rgb"]*v["mask"]+(1-v["mask"])*white
            l1.append(float((rgb.clamp(0, 1)-gtw).abs().mean()))
        l1b = []
        for v in ev:
            rb = render_gaussians(body["means"], body["quats"], body["scales"], body["opacities"], body["rgb"], v["c2w"], v["K"], v["W"], v["H"], bg=white)[0]
            gtw = v["rgb"]*v["mask"]+(1-v["mask"])*white; l1b.append(float((rb.clamp(0,1)-gtw).abs().mean()))
    print(f"[v6flow-train] {args.dog} held-out L1: Stage-1={np.mean(l1b):.4f} -> +fur={np.mean(l1):.4f} | fur_op={float(S['opacities'].mean()):.3f}", flush=True)

    with torch.no_grad():
        S = st(); gs = {k: torch.cat([body[k], S[k]]) for k in RK}
        save_ply(os.path.join(args.out, f"{args.dog}.ply"), gs["means"], gs["scales"], gs["quats"], gs["opacities"], gs["rgb"])
        torch.save({"sd": st.state_dict(), "diag": diag, "curl_id": curl_id, "Kp": args.Kp,
                    "body": {k: body[k] for k in RK}}, os.path.join(args.out, f"{args.dog}_fur.pt"))
        best = None
        for fr in frames:
            _, mk, _, _ = _load_rgb_mask(scene, fr, s); yy, xx = np.where(mk[:, :, 0] > 0.5)
            if len(xx) < 50: continue
            ar = (xx.max()-xx.min())/max(yy.max()-yy.min(), 1)
            if best is None or ar > best[0]: best = (ar, fr)
        v = best[1]; K = intrinsics(v["fx"]/s, v["fy"]/s, v["cx"]/s, v["cy"]/s, dev); c2w = torch.tensor(v["c2w"], device=dev).float()
        rgbg, mask, W, H = _load_rgb_mask(scene, v, s)
        R = lambda d: (render_gaussians(d["means"], d["quats"], d["scales"], d["opacities"], d["rgb"], c2w, K, W, H, bg=white)[0].clamp(0, 1).cpu().numpy())
        s1 = R(body); fo = R(S); cp = R(gs); gim = rgbg
        m = mask[:, :, 0] > 0.5; ys, xs = np.where(m); cr = lambda a: a[max(ys.min()-10,0):ys.max()+10, max(xs.min()-10,0):xs.max()+10]
        h = min(cr(x).shape[0] for x in [s1, fo, cp, gim])
        Image.fromarray((np.concatenate([cr(x)[:h] for x in [s1, fo, cp, gim]], 1)*255).astype(np.uint8)).save(os.path.join(args.out, f"{args.dog}_decomp.png"))
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cy0, cy1 = int(y0+0.25*(y1-y0)), int(y0+0.80*(y1-y0)); cx0, cx1 = int(x0+0.30*(x1-x0)), int(x0+0.75*(x1-x0))
        cz = lambda a: np.array(Image.fromarray((a[cy0:cy1, cx0:cx1]*255).astype(np.uint8)).resize(((cx1-cx0)*2, (cy1-cy0)*2)))
        Image.fromarray(np.concatenate([cz(fo), cz(cp), cz(gim)], 1)).save(os.path.join(args.out, f"{args.dog}_closeup.png"))
    print(f"[v6flow-train] saved {args.dog} decomp/closeup/ply/fur.pt -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
