"""P3-pre: NeuralFur-style strand geometry init on the D-SMAL template (no training).

Tangent field: anatomical flow (head->tail on torso, downward on legs) projected to
the tangent plane and Laplacian-smoothed ON THE CANONICAL TEMPLATE (computed once,
shared by all dogs), transported to the posed mesh via per-vertex TBN frames.
Strands grow in the local (t,b,n) frame: tangent-dominant direction with a small
normal lift, drooping toward gravity; roots on the inset (de-furred) surface;
length from the per-joint VLM prior. Visualized as polyline overlays on GT views.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dsmal_region_masks import DSMAL_ROOT, SCENE_ROOT, load_smal, scene_verts
from defur_mask import vertex_normals, local_thickness, length_field_cm

LEG_JOINTS = list(range(7, 15)) + list(range(17, 25))
TAIL_JOINTS = list(range(25, 32))


def smooth_tangent_field(verts, faces, flow, iters=30):
    """Project `flow` [V,3] to tangent planes and smooth over vertex adjacency."""
    V = verts.shape[0]
    n = torch.tensor(vertex_normals(verts, faces))
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = torch.cat([e, e.flip(1)], 0)
    t = F.normalize(flow - (flow * n).sum(-1, keepdim=True) * n, dim=-1, eps=1e-8)
    deg = torch.zeros(V, 1).index_add_(0, e[:, 0], torch.ones(len(e), 1))
    for _ in range(iters):
        avg = torch.zeros(V, 3).index_add_(0, e[:, 0], t[e[:, 1]]) / deg.clamp_min(1)
        t = F.normalize(avg - (avg * n).sum(-1, keepdim=True) * n, dim=-1, eps=1e-8)
    return t, n


def vert_frames(verts, faces, nbr_idx):
    """Orthonormal per-vertex frame [V,3,3] (tangent-by-edge, bitangent, normal cols)."""
    n = torch.tensor(vertex_normals(verts, faces), dtype=torch.float32)
    v = verts
    e = v[nbr_idx] - v
    tang = F.normalize(e - (e * n).sum(-1, keepdim=True) * n, dim=-1, eps=1e-8)
    bit = torch.cross(n, tang, dim=-1)
    return torch.stack([tang, bit, n], dim=-1)


def grow_strands(roots, t, n, L, gravity, k_pts=10, tangent_mix=0.75, droop=0.6):
    """Polyline strands [N,k_pts,3]: start along the tangent flow (small normal
    lift), progressively blend toward gravity (NeuralFur's hang behavior)."""
    d0 = F.normalize(tangent_mix * t + (1 - tangent_mix) * n, dim=-1)
    pts = [roots]
    p = roots
    for k in range(1, k_pts):
        beta = droop * ((k / (k_pts - 1)) ** 1.5)
        dk = F.normalize((1 - beta) * d0 + beta * gravity.view(1, 3), dim=-1)
        p = p + dk * (L / (k_pts - 1)).view(-1, 1)
        pts.append(p)
    return torch.stack(pts, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", default="00062-bear")
    ap.add_argument("--prior", default="/home/yyang/mnt/workspace/exps/vlm_prior_bear.json")
    ap.add_argument("--views", nargs="+", default=["133", "165"])
    ap.add_argument("--n_viz", type=int, default=2500, help="strands drawn per view")
    ap.add_argument("--k_pts", type=int, default=10)
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--out", default="/home/yyang/mnt/workspace/exps/_p3_strand_init.png")
    args = ap.parse_args()

    smal = load_smal()
    prior = json.load(open(args.prior))

    # ---- canonical template: anatomical flow -> smooth tangent field ----------
    eye = torch.eye(3)[None, None].repeat(1, 35, 1, 1)
    d = np.load(os.path.join(DSMAL_ROOT, "params", f"{args.dog}.npz"))
    t_ = lambda k: torch.tensor(d["offset_" + k])
    canon = smal(beta=t_("betas"), betas_limbs=t_("betas_limbs"), pose=eye,
                 trans=torch.zeros(1, 3), vert_off_compact=t_("vert_off_compact"),
                 get_skin=True)[0][0].float()
    faces = torch.tensor(np.asarray(smal.faces)).long()
    W = torch.tensor(np.asarray(smal.weights), dtype=torch.float32)         # [V,35]
    w_leg = W[:, LEG_JOINTS].sum(1).clamp(0, 1)
    w_tail = W[:, TAIL_JOINTS].sum(1).clamp(0, 1)
    # head->tail (-x) on torso/face, downward (-z) on legs, off the tail tip (-x)
    flow = ((1 - w_leg).view(-1, 1) * torch.tensor([-1.0, 0, 0])
            + w_leg.view(-1, 1) * torch.tensor([0, 0, -1.0]))
    tan_c, _ = smooth_tangent_field(canon, faces, flow)

    # ---- transport to posed space via TBN frames ------------------------------
    posed, _, _ = scene_verts(smal, args.dog, "offset")
    nbr = torch.zeros(canon.shape[0], dtype=torch.long)
    nbr[faces[:, 0]] = faces[:, 1]
    nbr[faces[:, 1]] = faces[:, 2]
    nbr[faces[:, 2]] = faces[:, 0]
    F0 = vert_frames(canon, faces, nbr)
    Ft = vert_frames(posed, faces, nbr)
    tan_local = torch.einsum("vij,vj->vi", F0.transpose(1, 2), tan_c)
    tan_p = F.normalize(torch.einsum("vij,vj->vi", Ft, tan_local), dim=-1)
    n_p = Ft[:, :, 2]

    # ---- inset roots + length field + gravity --------------------------------
    Lcm = torch.tensor(length_field_cm(smal, prior))
    diag = float((posed.max(0).values - posed.min(0).values).norm())
    upc = diag / prior["dog_bbox_diag_cm"]
    L = Lcm * upc
    th = torch.tensor(local_thickness(posed, faces, n_p.numpy()), dtype=torch.float32)
    inset = torch.minimum(L, 0.45 * th)
    roots = posed - n_p * inset[:, None]
    paws = W[:, [10, 14, 20, 24]].sum(1) > 0.4
    gravity = F.normalize(posed[paws].mean(0) - posed.mean(0), dim=0)       # paws point down

    strands = grow_strands(roots, tan_p, n_p, L, gravity, k_pts=args.k_pts)
    print(f"{len(strands)} strands, len range [{float(L.min()):.3f},{float(L.max()):.3f}] "
          f"units, gravity {gravity.tolist()}")

    # ---- overlay on GT views ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from PIL import Image

    cams = json.load(open(os.path.join(SCENE_ROOT, args.dog, "colmap/preprocess/cameras.json")))
    frames = {os.path.splitext(f["name"])[0]: f for f in cams["frames"]}
    fig, axes = plt.subplots(1, len(args.views), figsize=(8 * len(args.views), 10))
    axes = np.atleast_1d(axes)
    for c, vn in enumerate(args.views):
        fr = frames[vn]
        img = Image.open(os.path.join(SCENE_ROOT, args.dog, "colmap", fr["image_path"]))
        img = img.resize((fr["width"] // args.ds, fr["height"] // args.ds))
        w2c = torch.tensor(np.linalg.inv(np.array(fr["c2w"])), dtype=torch.float32)
        s = 1.0 / args.ds
        sel = torch.randperm(len(strands))[:args.n_viz]
        P = strands[sel].reshape(-1, 3)
        cam = P @ w2c[:3, :3].T + w2c[:3, 3]
        uv = torch.stack([cam[:, 0] / cam[:, 2] * fr["fx"] * s + fr["cx"] * s,
                          cam[:, 1] / cam[:, 2] * fr["fy"] * s + fr["cy"] * s], -1)
        uv = uv.reshape(len(sel), args.k_pts, 2).numpy()
        z = cam[:, 2].reshape(len(sel), args.k_pts).mean(1)
        order = np.argsort(-z.numpy())                                       # far first
        segs = np.stack([uv[:, :-1], uv[:, 1:]], 2).reshape(-1, 2, 2)
        seg_order = (order[:, None] * (args.k_pts - 1) + np.arange(args.k_pts - 1)).ravel()
        frac = np.tile(np.linspace(0.2, 1.0, args.k_pts - 1), len(sel))      # root->tip shade
        cols = plt.cm.copper(frac)[seg_order.argsort().argsort()]
        ax = axes[c]
        ax.imshow(img)
        ax.add_collection(LineCollection(segs[seg_order], colors=cols[seg_order],
                                         linewidths=0.5, alpha=0.85))
        ax.set_xlim(0, img.width); ax.set_ylim(img.height, 0); ax.axis("off")
        ax.set_title(f"{args.dog} v{vn}: NeuralFur-style strand init (root dark -> tip light)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
