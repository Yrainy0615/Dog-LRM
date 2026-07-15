#!/usr/bin/env python3
"""DiffLocks-style conditional DIFFUSION strand generator (prototype).
Per-root conditional DDPM over the low-dim strand latent (6 TBN points = 18-d; our strands are
low-dim so we diffuse params DIRECTLY -- no VAE, unlike DiffLocks' 256-pt human hair).
Condition (the "latent extracted from image") = pixel-aligned frozen-DINOv2 feature at the
projected root + canonical pos/normal (LHM-style; cross-attn is the upgrade). Trained on
synthetic 3D-strand GT. Sampling -> decode to strand control points -> render. Colour from image.
"""
import os, sys, glob, math, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
from PIL import Image
from train_strand_predictor import vnormals, tbn_frames, project, fourier
dev = "cuda"


class Denoiser(nn.Module):
    def __init__(self, zdim=18, cond=1024+39+3, tdim=64, hid=384):
        super().__init__()
        self.tdim = tdim
        self.net = nn.Sequential(nn.Linear(zdim+tdim+cond, hid), nn.SiLU(),
                                 nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU(),
                                 nn.Linear(hid, zdim))
    def temb(self, t):
        f = torch.exp(torch.arange(self.tdim//2, device=t.device) * -(math.log(10000)/(self.tdim//2-1)))
        a = t[:, None].float()*f[None]; return torch.cat([a.sin(), a.cos()], -1)
    def forward(self, z, t, c):
        return self.net(torch.cat([z, self.temb(t), c], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synth_fur/dataset")
    ap.add_argument("--template", default="synth_fur/blender_input.npz")
    ap.add_argument("--nroot", type=int, default=4000)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=2048, help="roots per step")
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--heldout_color", type=int, default=4)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--out", default="exps/strand_diff")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    tz = np.load(args.template)
    V = torch.from_numpy(tz["verts"].astype(np.float32)).to(dev); Fc = torch.from_numpy(tz["faces"].astype(np.int64)).to(dev)
    Nrm = vnormals(V, Fc); TBN = tbn_frames(V, Nrm)
    diag = float((V.max(0).values-V.min(0).values).norm())
    torch.manual_seed(0); ridx = torch.randperm(V.shape[0])[:args.nroot]
    R = V[ridx]; Rn = Nrm[ridx]; Rtbn = TBN[ridx]; posfix = R/diag
    K = 6; zdim = K*3

    from transformers import AutoModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dino = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
    for p in dino.parameters(): p.requires_grad = False
    nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); nsd = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    ph = args.res//14
    def dfeat(img):
        x = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float()
        x = (F.interpolate(x, (args.res, args.res), mode="bilinear", align_corners=False)-nm)/nsd
        with torch.no_grad(): f = dino(pixel_values=x).last_hidden_state[:, 1:]
        return f.transpose(1, 2).reshape(1, -1, ph, ph)

    train, evl = [], []
    cond_pos = torch.cat([fourier(posfix), Rn], -1)                  # [nroot, 39+3]
    for gp in sorted(glob.glob(os.path.join(args.data, "*.npz"))):
        z = np.load(gp, allow_pickle=True); style = str(z["style"]); ci = int(os.path.basename(gp)[:-4].split("_")[-1])
        strands = torch.from_numpy(z["strands"]).to(dev); roots = torch.from_numpy(z["roots"]).to(dev)
        nn_ = torch.cdist(R, roots).argmin(1); gt_world = strands[nn_]
        gt_local = torch.einsum("rij,rkj->rki", Rtbn.transpose(1, 2), gt_world-R[:, None])/diag   # [nroot,K,3]
        z0 = gt_local.reshape(args.nroot, zdim)
        Ks = torch.from_numpy(z["Ks"]).to(dev); c2ws = torch.from_numpy(z["c2ws"]).to(dev); imgs = z["imgs"]
        for vi in range(len(imgs)):
            img = np.array(Image.open(os.path.join(args.data, str(imgs[vi]))).convert("RGB"))/255.
            H, W = img.shape[:2]; uv, _ = project(R, Ks[vi], c2ws[vi])
            grid = torch.stack([uv[:, 0]/W*2-1, uv[:, 1]/H*2-1], -1)[None, :, None, :]
            samp = F.grid_sample(dfeat(img), grid, align_corners=False)[0, :, :, 0].T            # [nroot,1024]
            cond = torch.cat([samp, cond_pos], -1)
            (evl if ci == args.heldout_color else train).append(dict(z0=z0, cond=cond, style=style, name=os.path.basename(gp)[:-4], vi=vi))
    # standardize the strand latent to ~unit variance (DDPM assumes ~N(0,1) scale; our strands are tiny)
    allz = torch.cat([r["z0"] for r in train], 0)
    zmean = allz.mean(0); zstd = allz.std(0).clamp(min=1e-5)
    for r in train + evl: r["z0"] = (r["z0"] - zmean) / zstd
    print(f"[diff] train {len(train)} eval {len(evl)} samples | roots {args.nroot} zdim {zdim} | zstd~{float(zstd.mean()):.4f}", flush=True)

    # DDPM schedule
    betas = torch.linspace(1e-4, 0.02, args.T, device=dev); ac = torch.cumprod(1-betas, 0)
    net = Denoiser(zdim=zdim).to(dev); opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    for it in range(args.iters):
        r = train[np.random.randint(len(train))]
        idx = torch.randint(0, args.nroot, (args.bs,), device=dev)
        z0 = r["z0"][idx]; c = r["cond"][idx]
        t = torch.randint(0, args.T, (args.bs,), device=dev)
        eps = torch.randn_like(z0); a = ac[t][:, None]
        zt = a.sqrt()*z0 + (1-a).sqrt()*eps
        loss = F.mse_loss(net(zt, t, c), eps)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0: print(f"it{it:5d} loss={float(loss):.5f}", flush=True)

    # DDIM sample on held-out + measure strand error vs GT (+ mean baseline)
    @torch.no_grad()
    def sample(cond, steps=50):
        z = torch.randn(cond.shape[0], zdim, device=dev)
        ts = torch.linspace(args.T-1, 0, steps, device=dev).long()
        for i in range(steps):
            t = ts[i].expand(cond.shape[0]); a = ac[t][:, None]
            eps = net(z, t, cond); z0 = (z-(1-a).sqrt()*eps)/a.sqrt()
            if i < steps-1:
                a2 = ac[ts[i+1]]; z = a2.sqrt()*z0 + (1-a2).sqrt()*eps
            else: z = z0
        return z
    with torch.no_grad():
        unstd = lambda z: (z*zstd + zmean).view(args.nroot, K, 3)    # back to original ×diag units
        mean_z = torch.stack([r["z0"] for r in train]).mean(0)       # ~0 (standardized) -> unstd -> original mean
        by = {}
        for e in evl:
            zs = unstd(sample(e["cond"])); gt = unstd(e["z0"]); mb = unstd(mean_z)
            err = (zs-gt).norm(dim=-1).mean(); base = (mb-gt).norm(dim=-1).mean()
            by.setdefault(e["style"], []).append((float(err), float(base)))
    print("[diff] held-out per-style strand-point error (×diag) | diffusion vs mean-baseline:", flush=True)
    for st, v in by.items():
        v = np.array(v); print(f"   {st:6s}: diff {v[:,0].mean():.4f}  baseline {v[:,1].mean():.4f}", flush=True)
    torch.save({"net": net.state_dict(), "ridx": ridx.cpu(), "betas": betas.cpu(), "nroot": args.nroot,
                "zmean": zmean.cpu(), "zstd": zstd.cpu()}, os.path.join(args.out, "diff.pt"))
    print(f"[diff] saved diff.pt -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
