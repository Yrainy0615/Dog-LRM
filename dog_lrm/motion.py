"""Retarget an AnimalML3D (OmniMotionGPT) motion onto a SMAL identity.

The dataset stores only per-frame SMAL *joint positions* [T,35,3] (BARC joint order;
verified). To drive an avatar we need per-frame pose `theta`, so we IK-fit a SMAL
(with the motion dog's own betas) to those joints -> clean pose angles, which then
transfer to any identity (rinda) via LBS.
"""
import json
import os

import numpy as np
import torch


def load_motion_joints(npy_path, device):
    """[T,35,3] joint positions for one motion."""
    j = np.load(npy_path).astype(np.float32)
    assert j.ndim == 3 and j.shape[1:] == (35, 3), j.shape
    return torch.from_numpy(j).to(device)


def load_motion_betas(template_dir, motion_name, num_betas, device):
    """Motion identity betas from the per-animal template json (e.g. 'doggieMN5')."""
    ident = motion_name.split("_")[0]
    import glob
    files = glob.glob(os.path.join(template_dir, f"{ident}_*.json"))
    assert len(files) == 1, f"{ident}: {files}"
    d = json.load(open(files[0]))
    betas = torch.zeros(1, num_betas, device=device)
    b = torch.tensor(d["beta"], device=device).float()
    betas[0, :min(num_betas, b.numel())] = b[:num_betas]
    return betas


def fit_theta_to_joints(smal, betas, target, iters=400, lr=0.05,
                        w_smooth=2.0, w_prior=0.02, device="cuda", verbose=True):
    """IK: optimize per-frame {theta[35,3], trans} + shared scale so SMAL joints match
    `target` [T,35,3]. betas/limbs fixed (motion identity). Returns dict of tensors."""
    T = target.shape[0]
    limbs = torch.zeros(1, smal.smal.num_betas_logscale, device=device)
    theta = torch.zeros(T, 35, 3, device=device, requires_grad=True)
    trans = target.mean(dim=1).clone().detach().requires_grad_(True)        # [T,3]
    log_scale = torch.zeros(1, device=device, requires_grad=True)           # shared
    opt = torch.optim.Adam([theta, trans, log_scale], lr=lr)
    bt, lt = betas.expand(T, -1), limbs.expand(T, -1)

    for it in range(iters):
        scale = torch.exp(log_scale)
        J = smal.posed_joints(bt, lt, theta, trans, scale.expand(T))         # [T,35,3]
        data = ((J - target) ** 2).sum(-1).mean()
        smooth = ((theta[1:] - theta[:-1]) ** 2).mean() if T > 1 else 0.0
        prior = (theta[:, 1:] ** 2).mean()                                   # weak, body only
        loss = data + w_smooth * smooth + w_prior * prior
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (it % 100 == 0 or it == iters - 1):
            print(f"  ik it{it:4d} data={float(data):.5f} "
                  f"smooth={float(smooth):.5f} scale={float(scale):.3f}")

    return dict(theta=theta.detach(), trans=trans.detach(),
                scale=torch.exp(log_scale).detach())


def lbs_world_affine(smal_model, betas, limbs, theta, trans, scale):
    """Per (subdivided) vertex 4x4 world affine for a pose: world = Tr(trans)·Scale·(W·A).
    Reuses BARC's exact LBS math. Returns [Vsub,4,4]. Lets us re-pose a baked Gaussian
    cloud via M = affine(theta_t) @ inv(affine(theta_0))."""
    from smal_pytorch.smal_model.batch_lbs import (
        batch_rodrigues, batch_global_rigid_transformation_biggs)
    smal = smal_model.smal
    dev = betas.device
    nB = betas.shape[1]
    v_shaped = smal.v_template + (betas @ smal.shapedirs[:nB]).reshape(1, -1, 3)
    J = torch.stack([v_shaped[:, :, i] @ smal.J_regressor for i in range(3)], dim=2)  # [1,35,3]
    Rs = batch_rodrigues(theta.reshape(-1, 3)).reshape(1, 35, 3, 3)
    betas_scale = torch.exp(limbs @ smal.betas_scale_mask.to(dev))
    scale3x3 = torch.diag_embed(betas_scale.reshape(-1, 35, 3), dim1=-2, dim2=-1)
    _, A = batch_global_rigid_transformation_biggs(Rs, J, smal.parents, scale3x3,
                                                   betas_logscale=limbs)            # [1,35,4,4]
    Wsub = torch.sparse.mm(smal_model.subdiv_M, smal.weights)                        # [Vsub,35]
    T = (Wsub @ A[0].reshape(35, 16)).reshape(-1, 4, 4)                              # [Vsub,4,4]
    Vsub = T.shape[0]
    S = torch.eye(4, device=dev).repeat(Vsub, 1, 1); S[:, :3, :3] *= float(scale)
    Tr = torch.eye(4, device=dev).repeat(Vsub, 1, 1); Tr[:, :3, 3] = trans.reshape(3)
    return Tr @ S @ T


def mat_to_quat(R):
    """[...,3,3] rotation -> wxyz quaternion (orthonormalized via the standard trace method)."""
    U, _, Vh = torch.linalg.svd(R)
    R = U @ Vh
    det = torch.linalg.det(R)
    U[..., -1] *= det.unsqueeze(-1)                      # fix reflections
    R = U @ Vh
    m = lambda i, j: R[..., i, j]
    w = torch.sqrt(torch.clamp(1 + m(0, 0) + m(1, 1) + m(2, 2), min=1e-8)) / 2
    x = (m(2, 1) - m(1, 2)) / (4 * w)
    y = (m(0, 2) - m(2, 0)) / (4 * w)
    z = (m(1, 0) - m(0, 1)) / (4 * w)
    return torch.stack([w, x, y, z], -1)


def quat_mul(a, b):
    """Hamilton product of wxyz quaternions [...,4]."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], -1)


def look_at(center, azim_deg, elev_deg, dist, device, up=(0, 0, 1)):
    """OpenCV-convention c2w (x right, y down, z forward) orbiting `center`.
    Matches render.py which uses viewmat = inv(c2w)."""
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    up = torch.tensor(up, dtype=torch.float32, device=device)
    center = center.to(device)
    eye = center + dist * torch.tensor(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        dtype=torch.float32, device=device)
    fwd = (center - eye); fwd = fwd / fwd.norm()                 # +z (cam looks along)
    right = torch.cross(fwd, up); right = right / right.norm()   # +x
    down = torch.cross(fwd, right)                               # +y (down)
    c2w = torch.eye(4, device=device)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
    return c2w
