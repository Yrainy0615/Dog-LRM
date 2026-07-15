#!/usr/bin/env python3
"""DiffLocks-style image->strand GEOMETRY predictor (prototype).
Fixed root set = D-SMAL template verts (subsampled). GT per root = nearest synthetic strand,
expressed in the root's local TBN (view-independent). Predictor: frozen DINOv2 patch features,
pixel-aligned per root (project root via the groom camera) + positional enc -> MLP -> [K,3] TBN
strand offsets. Supervised by synthetic 3D GT (NOT photometric L1). Held-out colours test
generalization (does it read coat STYLE from the image?). Colour is sampled from image (not here).
"""
import os, sys, glob, math, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
from PIL import Image

dev = "cuda"


def vnormals(V, Fc):
    fn = torch.cross(V[Fc[:, 1]]-V[Fc[:, 0]], V[Fc[:, 2]]-V[Fc[:, 0]], dim=-1)
    n = torch.zeros_like(V); n.index_add_(0, Fc.reshape(-1), fn.repeat_interleave(3, 0))
    return F.normalize(n, dim=-1)


def tbn_frames(V, N):
    up = torch.tensor([0., 0., 1.], device=V.device).expand_as(N)
    t = F.normalize(up - (up*N).sum(-1, keepdim=True)*N, dim=-1)
    deg = t.norm(dim=-1) < 1e-3
    if deg.any(): t[deg] = F.normalize(torch.cross(N[deg], torch.tensor([1., 0., 0.], device=V.device).expand_as(N[deg]), dim=-1), dim=-1)
    b = torch.cross(N, t, dim=-1)
    return torch.stack([t, b, N], -1)            # [V,3,3] columns=t,b,n


def project(P, K, c2w):
    flip = torch.diag(torch.tensor([1., -1., -1.], device=P.device))
    w2c = torch.inverse(c2w); cam = (w2c[:3, :3] @ P.T + w2c[:3, 3:4]).T @ flip.T
    z = cam[:, 2].clamp(min=1e-4); uv = (K @ (cam/z[:, None]).T).T[:, :2]
    return uv, cam[:, 2]


def fourier(x, L=6):
    f = 2.0 ** torch.arange(L, device=x.device) * math.pi
    xf = x[..., None] * f
    return torch.cat([x, xf.sin().flatten(-2), xf.cos().flatten(-2)], -1)


class Head(nn.Module):
    def __init__(self, dino_dim=1024, pos_dim=3+3*2*6, K=6, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dino_dim+pos_dim+3, hid), nn.SiLU(),
                                 nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, (K)*3))
        self.K = K
    def forward(self, dino, pos, nrm):
        x = torch.cat([dino, fourier(pos), nrm], -1)
        return self.net(x).view(-1, self.K, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synth_fur/dataset")
    ap.add_argument("--template", default="synth_fur/blender_input.npz")
    ap.add_argument("--nroot", type=int, default=4000)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--heldout_color", type=int, default=4, help="color idx held out per style")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--out", default="exps/strand_pred")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    tz = np.load(args.template)
    V = torch.from_numpy(tz["verts"].astype(np.float32)).to(dev); Fc = torch.from_numpy(tz["faces"].astype(np.int64)).to(dev)
    N = vnormals(V, Fc); TBN = tbn_frames(V, N)
    diag = float((V.max(0).values-V.min(0).values).norm())
    torch.manual_seed(0); ridx = torch.randperm(V.shape[0])[:args.nroot]
    R = V[ridx]; Rn = N[ridx]; Rtbn = TBN[ridx]                       # fixed roots

    # DINOv2 frozen
    from transformers import AutoModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dino = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
    for p in dino.parameters(): p.requires_grad = False
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    ph = args.res // 14

    def dino_feat(img_np):
        x = torch.from_numpy(img_np).to(dev).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, (args.res, args.res), mode="bilinear", align_corners=False)
        x = (x - norm_mean)/norm_std
        with torch.no_grad():
            f = dino(pixel_values=x).last_hidden_state[:, 1:]        # [1,ph*ph,1024]
        return f.transpose(1, 2).reshape(1, -1, ph, ph)              # [1,1024,ph,ph]

    # build samples: per (groom,view) -> (dino featmap, per-root gt strand in TBN, projection grid)
    grooms = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    train, evl = [], []
    for gp in grooms:
        z = np.load(gp, allow_pickle=True); style = str(z["style"]); name = os.path.basename(gp)[:-4]
        ci = int(name.split("_")[-1])
        strands = torch.from_numpy(z["strands"]).to(dev); roots = torch.from_numpy(z["roots"]).to(dev)
        # GT per fixed-root: nearest emission strand, in root TBN (offsets from root)
        nn_ = torch.cdist(R, roots).argmin(1)
        gt_world = strands[nn_]                                      # [nroot,K,3]
        gt_local = torch.einsum("rij,rkj->rki", Rtbn.transpose(1, 2), gt_world - R[:, None]) / diag
        Ks = torch.from_numpy(z["Ks"]).to(dev); c2ws = torch.from_numpy(z["c2ws"]).to(dev); imgs = z["imgs"]
        for vi in range(len(imgs)):
            img = np.array(Image.open(os.path.join(args.data, str(imgs[vi]))).convert("RGB"))/255.
            W = img.shape[1]; H = img.shape[0]
            uv, zc = project(R, Ks[vi], c2ws[vi])
            grid = torch.stack([uv[:, 0]/W*2-1, uv[:, 1]/H*2-1], -1)[None, :, None, :]   # [1,nroot,1,2]
            fm = dino_feat(img)
            samp = F.grid_sample(fm, grid, align_corners=False)[0, :, :, 0].T            # [nroot,1024]
            rec = dict(dino=samp, gt=gt_local, style=style)
            (evl if ci == args.heldout_color else train).append(rec)
    print(f"[pred] train {len(train)} eval {len(evl)} | roots {args.nroot} | feat cached", flush=True)

    head = Head().to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    posfix = R/diag
    for it in range(args.iters):
        r = train[np.random.randint(len(train))]
        pred = head(r["dino"], posfix, Rn)
        loss = F.mse_loss(pred, r["gt"])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            with torch.no_grad():
                ev = np.mean([float(F.mse_loss(head(e["dino"], posfix, Rn), e["gt"])) for e in evl])
            print(f"it{it:4d} train_mse={float(loss):.5f} heldout_mse={ev:.5f}", flush=True)

    # report per-style held-out strand position error (in diag units) + a baseline (predict mean strand)
    with torch.no_grad():
        mean_gt = torch.stack([r["gt"] for r in train]).mean(0)
        by = {}
        for e in evl:
            pred = head(e["dino"], posfix, Rn)
            err = (pred-e["gt"]).norm(dim=-1).mean()              # mean point error (diag units)
            base = (mean_gt-e["gt"]).norm(dim=-1).mean()
            by.setdefault(e["style"], []).append((float(err), float(base)))
    print("[pred] held-out per-style mean strand-point error (×diag) | predictor vs mean-baseline:", flush=True)
    for st, v in by.items():
        v = np.array(v); print(f"   {st:6s}: pred {v[:,0].mean():.4f}  baseline {v[:,1].mean():.4f}", flush=True)
    torch.save({"head": head.state_dict(), "ridx": ridx.cpu(), "nroot": args.nroot}, os.path.join(args.out, "head.pt"))
    print(f"[pred] saved head.pt -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
