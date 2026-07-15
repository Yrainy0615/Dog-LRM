"""P0-3: de-furred (bald) body mask — compare two derivations on one dog.

A) 2D: GT-mask SDF eroded by the projected fur-length field,
   thin-structure guard = cap erosion at 0.6 x local medial radius.
B) 3D: inset the D-SMAL mesh per-vertex along -normal by the fur length
   (capped at 0.45 x local thickness from ray casting), z-buffer render.
"""
import os, json, argparse
import numpy as np
import torch

from dsmal_region_masks import DSMAL_ROOT, SCENE_ROOT, load_smal, scene_verts


def vertex_normals(verts, faces):
    v, f = verts.numpy(), faces.numpy()
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # area-weighted
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def local_thickness(verts, faces, normals, device="cuda"):
    """Per-vertex distance to the opposite surface along -normal (Moller-Trumbore, brute force on GPU)."""
    v = verts.to(device)
    f = faces.to(device)
    scale = float((v.max(0).values - v.min(0).values).norm())
    eps = 1e-4 * scale
    o = v - torch.as_tensor(normals, dtype=torch.float32, device=device) * eps
    d = -torch.as_tensor(normals, dtype=torch.float32, device=device)
    v0, e1, e2 = v[f[:, 0]], v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    th = torch.full((len(v),), float("inf"), device=device)
    for i in range(0, len(v), 512):                                  # chunk rays to bound memory
        oi, di = o[i:i + 512, None], d[i:i + 512, None]              # [R,1,3]
        p = torch.cross(di.expand(-1, len(f), -1), e2[None], dim=-1)
        det = (e1[None] * p).sum(-1)
        inv = torch.where(det.abs() > 1e-10, 1.0 / det, torch.zeros_like(det))
        s = oi - v0[None]
        u = (s * p).sum(-1) * inv
        q = torch.cross(s, e1[None].expand_as(s), dim=-1)
        w = (di * q).sum(-1) * inv
        t = (e2[None] * q).sum(-1) * inv
        hit = (det.abs() > 1e-10) & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > eps)
        t = torch.where(hit, t, torch.full_like(t, float("inf")))
        th[i:i + 512] = t.min(1).values
    th = th.cpu().numpy()
    th[~np.isfinite(th)] = scale                                     # open surface: effectively no clamp
    return th


def length_field_cm(smal, prior):
    w = smal.weights if torch.is_tensor(getattr(smal, "weights", None)) else torch.tensor(np.asarray(smal.weights))
    Lj = np.full(w.shape[1], float(prior.get("default_cm", 4.0)))
    for j, cm in prior["joint_lengths_cm"].items():
        Lj[int(j)] = float(cm)
    return (w.numpy() @ Lj).astype(np.float32)                       # [V] smooth cm


def rasterize_attr(verts, faces, attr, fr, ds, device="cuda"):
    """Z-buffered render: returns (mask[H,W], attr_map[H,W]) at 1/ds res."""
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
    from pytorch3d.utils import cameras_from_opencv_projection
    from pytorch3d.ops import interpolate_face_attributes
    H, W = fr["height"] // ds, fr["width"] // ds
    s = 1.0 / ds
    K = torch.tensor([[fr["fx"] * s, 0, fr["cx"] * s], [0, fr["fy"] * s, fr["cy"] * s], [0, 0, 1]],
                     dtype=torch.float32, device=device)[None]
    w2c = torch.tensor(np.linalg.inv(np.array(fr["c2w"])), dtype=torch.float32, device=device)
    cam = cameras_from_opencv_projection(w2c[None, :3, :3], w2c[None, :3, 3], K,
                                         torch.tensor([[H, W]], device=device))
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])
    frag = MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(
        image_size=(H, W), faces_per_pixel=1, bin_size=None)).forward(mesh)
    mask = (frag.pix_to_face[0, :, :, 0] >= 0).cpu().numpy()
    a = torch.as_tensor(attr, dtype=torch.float32, device=device)[faces.to(device)][:, :, None]
    amap = interpolate_face_attributes(frag.pix_to_face, frag.bary_coords, a)[0, :, :, 0, 0].cpu().numpy()
    return mask, amap


def defur_2d(gt_mask, Lcm_map, mesh_mask, px_per_cm):
    """Erode GT-mask SDF by the (nearest-propagated) projected length field."""
    from scipy.ndimage import distance_transform_edt, maximum_filter
    _, (iy, ix) = distance_transform_edt(~mesh_mask, return_indices=True)
    L_px = Lcm_map[iy, ix] * px_per_cm                               # cm field over whole image
    sdf = distance_transform_edt(gt_mask)
    win = max(int(2 * np.median(L_px[gt_mask])) | 1, 3)
    medial = maximum_filter(sdf, size=win)                           # ~ local medial radius
    tau = np.minimum(L_px, 0.6 * medial)
    return gt_mask & (sdf >= tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", default="00062-bear")
    ap.add_argument("--prior", default="/home/yyang/mnt/workspace/exps/vlm_prior_bear.json")
    ap.add_argument("--views", nargs="+", default=["133", "165"])
    ap.add_argument("--stage", default="offset")
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--viz", default="/home/yyang/mnt/workspace/exps/_p0_defur_compare.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    prior = json.load(open(args.prior))
    smal = load_smal()
    verts, faces, _ = scene_verts(smal, args.dog, args.stage)
    Lcm = length_field_cm(smal, prior)
    units_per_cm = float((verts.max(0).values - verts.min(0).values).norm()) / prior["dog_bbox_diag_cm"]

    normals = vertex_normals(verts, faces)
    th = local_thickness(verts, faces, normals)
    inset = np.minimum(Lcm * units_per_cm, 0.45 * th)
    verts_bald = verts - torch.tensor(normals * inset[:, None], dtype=torch.float32)
    print(f"units/cm={units_per_cm:.4f}  inset units [{inset.min():.3f},{inset.max():.3f}]  "
          f"thickness-clamped verts: {(inset < Lcm * units_per_cm - 1e-6).sum()}/{len(inset)}")

    cams = json.load(open(os.path.join(SCENE_ROOT, args.dog, "colmap/preprocess/cameras.json")))
    frames = {os.path.splitext(f["name"])[0]: f for f in cams["frames"]}
    fig, axes = plt.subplots(len(args.views), 3, figsize=(15, 6.5 * len(args.views)))
    axes = np.atleast_2d(axes)

    for r, vn in enumerate(args.views):
        fr = frames[vn]
        gt = np.asarray(Image.open(os.path.join(SCENE_ROOT, args.dog, "colmap/preprocess/masks", fr["name"].replace(".jpg", ".png")))
                        .resize((fr["width"] // args.ds, fr["height"] // args.ds), Image.NEAREST)) > 127
        img = np.asarray(Image.open(os.path.join(SCENE_ROOT, args.dog, "colmap", fr["image_path"]))
                         .resize((fr["width"] // args.ds, fr["height"] // args.ds))) / 255.0

        mesh_mask, Lcm_map = rasterize_attr(verts, faces, Lcm, fr, args.ds)
        diag_px = np.sqrt((np.argwhere(gt).ptp(0) ** 2).sum())
        bald2d = defur_2d(gt, Lcm_map, mesh_mask, diag_px / prior["dog_bbox_diag_cm"])
        bald3d, _ = rasterize_attr(verts_bald, faces, Lcm, fr, args.ds)

        inter, union = (bald2d & bald3d).sum(), (bald2d | bald3d).sum()
        print(f"view {vn}: 2D-vs-3D IoU {inter/union:.3f}  area ratio 2D/GT {bald2d.sum()/gt.sum():.2f} 3D/GT {bald3d.sum()/gt.sum():.2f}")

        for c, (m, ttl, col) in enumerate([(bald2d, "A) 2D SDF eroded", np.array([1.0, 0.2, 0.1])),
                                           (bald3d, "B) 3D inset render", np.array([0.1, 0.8, 1.0])),
                                           (None, "A vs B", None)]):
            ax = axes[r, c]
            ov = img.copy()
            if m is not None:
                ov[m] = ov[m] * 0.45 + col * 0.55
            else:
                both, onlyA, onlyB = bald2d & bald3d, bald2d & ~bald3d, bald3d & ~bald2d
                ov[both] = ov[both] * 0.45 + np.array([0.6, 0.4, 0.9]) * 0.55
                ov[onlyA] = ov[onlyA] * 0.45 + np.array([1.0, 0.2, 0.1]) * 0.55
                ov[onlyB] = ov[onlyB] * 0.45 + np.array([0.1, 0.8, 1.0]) * 0.55
            ax.imshow(ov)
            ax.contour(gt, levels=[0.5], colors="white", linewidths=0.7)
            ax.set_title(f"{args.dog} v{vn} {ttl}", fontsize=10)
            ax.axis("off")

    fig.suptitle("P0-3: de-fur mask — 2D SDF erosion vs 3D inset render (white = GT furry mask)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.viz, dpi=110)
    print("saved", args.viz)


if __name__ == "__main__":
    main()
