"""gsplat rendering with COLMAP (OpenCV-convention) cameras."""
import numpy as np
import torch
import gsplat

_SH0 = 0.28209479177387814
_SH1 = 0.48860251190292


def sh0_from_rgb(rgb):
    """Base color [..,3] -> SH DC coeff (gsplat: color = clamp(sum basis*coeff + 0.5, 0))."""
    return (rgb - 0.5) / _SH0


def sh1_from_vec(a_w):
    """World-frame linear color vector a_w [..,3 (xyz),3 (rgb)] -> deg-1 coeffs [..,3,3]
    so the rendered linear term is dot(a_w, view_dir) per channel.
    gsplat deg-1 basis order/signs: (-y, +z, -x) * _SH1."""
    return torch.stack([-a_w[..., 1, :], a_w[..., 2, :], -a_w[..., 0, :]], dim=-2) / _SH1


def save_ply(path, means, scales, quats, opacities, rgb):
    """Write Gaussians in the standard 3DGS .ply layout (loadable by GS viewers).
    Inputs are post-activation: scales linear, opacities in (0,1), quats wxyz, rgb in [0,1]."""
    from plyfile import PlyData, PlyElement
    xyz = means.detach().cpu().numpy().astype(np.float32)
    f_dc = ((rgb.detach().cpu().numpy() - 0.5) / _SH0).astype(np.float32)        # inverse SH0
    op = opacities.detach().cpu().clamp(1e-4, 1 - 1e-4).numpy()
    op = np.log(op / (1 - op)).reshape(-1, 1).astype(np.float32)                 # logit
    sc = np.log(scales.detach().cpu().clamp_min(1e-8).numpy()).astype(np.float32)  # log scale
    rot = quats.detach().cpu().numpy().astype(np.float32)
    n = xyz.shape[0]
    fields = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
              + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([xyz, np.zeros((n, 3), np.float32), f_dc, op, sc, rot], axis=1)
    el = np.empty(n, dtype=[(f, "f4") for f in fields])
    for i, f in enumerate(fields):
        el[f] = data[:, i]
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


def load_ply(path, device):
    """Inverse of save_ply -> post-activation tensors (means, scales, quats wxyz,
    opacities in (0,1), rgb in [0,1])."""
    from plyfile import PlyData
    import torch
    p = PlyData.read(path)["vertex"]
    g = lambda *fs: torch.from_numpy(np.stack([p[f] for f in fs], 1).astype(np.float32)).to(device)
    means = g("x", "y", "z")
    rgb = g("f_dc_0", "f_dc_1", "f_dc_2") * _SH0 + 0.5
    op = torch.sigmoid(g("opacity")[:, 0])
    scales = torch.exp(g("scale_0", "scale_1", "scale_2"))
    quats = torch.nn.functional.normalize(g("rot_0", "rot_1", "rot_2", "rot_3"), dim=-1)
    return dict(means=means, rgb=rgb, opacities=op, scales=scales, quats=quats)


def intrinsics(fx, fy, cx, cy, device):
    K = torch.zeros(3, 3, device=device)
    K[0, 0], K[1, 1], K[2, 2] = fx, fy, 1.0
    K[0, 2], K[1, 2] = cx, cy
    return K


def render_gaussians(means, quats, scales, opacities, colors, c2w, K, width, height,
                     bg=None, sh_degree=None, return_depth=False, rasterize_mode="classic"):
    """means[N,3] quats[N,4 wxyz] scales[N,3] opacities[N], world frame.
    colors[N,3] direct RGB, or [N,(sh_degree+1)^2,3] SH coeffs when sh_degree is set.
    c2w[4,4], K[3,3]. COLMAP world == OpenCV, so viewmat = inv(c2w) directly.
    Returns rgb[H,W,3], alpha[H,W,1]; if return_depth, also depth[H,W,1]
    (expected camera-space z, alpha-normalised)."""
    viewmat = torch.inverse(c2w)[None]  # [1,4,4] world->cam
    out, alpha, _ = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmat, Ks=K[None], width=width, height=height,
        render_mode="RGB+ED" if return_depth else "RGB", packed=False, sh_degree=sh_degree,
        rasterize_mode=rasterize_mode)
    out, alpha = out[0], alpha[0]
    rgb = out[..., :3] if return_depth else out
    if bg is not None:
        rgb = rgb + (1.0 - alpha) * bg
    if return_depth:
        return rgb, alpha, out[..., 3:4]            # ED already alpha-normalised
    return rgb, alpha
