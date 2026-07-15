#!/usr/bin/env python3
"""Animate a P1a decomp avatar with an AnimalML3D motion + fur sway dynamics.

Pipeline: text -> caption retrieval -> IK directly against D-SMAL (BARC thetas do
NOT transfer: the 'norm' variant re-orients rest joints, cross-applying axis-angles
tilts/garbles the pose — verified) -> per-frame verts at the dog's own scale ->
decomp Gaussian offsets re-anchored per frame (features are pose-invariant) ->
cantilever fur sway (face pinned, fur-length-scaled amplitude) -> orbit cam mp4.

  python animate_decomp.py --dog 00010-hanabi --text "a dog trots forward" \
      --ckpt exps/dog_lrm_decomp/model.pt
"""
import argparse
import json
import os
import sys

import imageio
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

DSMAL_ROOT = "received_data_from_Pinstudio_20260424/InterPet2026/dsmal_dataset"
SCENE_ROOT = "received_data_from_Pinstudio_20260424/unzipped/0423"
IK_CACHE = "exps/anim_cache"


def load_dsmal():
    sys.path.insert(0, os.path.abspath(os.path.join(DSMAL_ROOT, "dsmal_code")))
    cwd = os.getcwd()
    os.chdir(DSMAL_ROOT)
    import _compat_shim  # noqa: F401
    from smal_pytorch.smal_model.smal_torch_new import SMAL
    from configs.SMAL_configs import SMAL_MODEL_CONFIG
    smal = SMAL(smal_model_type="39dogs_norm_newv3", template_name="neutral",
                logscale_part_list=SMAL_MODEL_CONFIG["39dogs_norm_newv3"]["logscale_part_list"])
    os.chdir(cwd)
    return smal.to("cuda")


def dsmal_ik(dsmal, dog_params, target, iters=400, lr=0.05, w_smooth=2.0,
             w_prior=0.02, dev="cuda"):
    """Fit per-frame theta[35,3] so the dog's own D-SMAL joints match the motion
    joints `target` [T,35,3] (up to a shared scale + per-frame trans, discarded)."""
    t = lambda k: torch.tensor(dog_params["offset_" + k], device=dev)
    T_ = target.shape[0]
    beta = t("betas").expand(T_, -1)
    limbs = t("betas_limbs").expand(T_, -1)
    voff = t("vert_off_compact").expand(T_, -1)
    theta = torch.zeros(T_, 35, 3, device=dev, requires_grad=True)
    trans = torch.zeros(T_, 1, 3, device=dev, requires_grad=True)
    log_s = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta, trans, log_s], lr=lr)
    # strong prior on the spine chain: the motion targets come from the BARC-SMAL
    # skeleton, whose back joints sit elsewhere than D-SMAL's 'norm' layout — with a
    # weak prior the IK contorts the spine ~50-60 deg/joint to hit those positions
    # (joint POSITIONS match, orientations garbage) and LBS explodes the back verts.
    wp = torch.full((34,), w_prior, device=dev)
    wp[0:6] = 0.5                                                    # body joints 1-6
    for it in range(iters):
        _, J, _ = dsmal(beta=beta, betas_limbs=limbs, theta=theta,
                        trans=torch.zeros(T_, 3, device=dev), get_skin=True,
                        vert_off_compact=voff)
        J = J[:, :35]
        loss = F.mse_loss(J * torch.exp(log_s) + trans, target)
        loss = loss + w_smooth * (theta[1:] - theta[:-1]).pow(2).mean()
        loss = loss + (theta[:, 1:].pow(2) * wp.view(1, -1, 1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0:
            print(f"  ik it{it} loss {float(loss):.5f}", flush=True)
    return theta.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="a dog trots forward")
    ap.add_argument("--dog", default="00211-pon")
    ap.add_argument("--ckpt", default="exps/dog_lrm_decomp/model.pt")
    ap.add_argument("--n_subdiv", type=int, default=1)
    ap.add_argument("--gs_per_pt", type=int, default=2)
    ap.add_argument("--vlm_prior", default=None, help="fur length json; default 5cm uniform")
    ap.add_argument("--out", default=None)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--amp_frac", type=float, default=0.55)
    ap.add_argument("--stiffness", type=float, default=55.0)
    ap.add_argument("--zeta", type=float, default=0.22)
    ap.add_argument("--wind_frac", type=float, default=0.45)
    ap.add_argument("--drag_frac", type=float, default=0.8)
    ap.add_argument("--gust_hz", type=float, default=0.7)
    args = ap.parse_args()
    dev = "cuda"

    from animate_fur import retrieve_motion
    from dog_lrm.motion import load_motion_joints, look_at
    from dog_lrm.model_decomp import DogLRMDecomp
    from dog_lrm.smal_model import build_subdiv
    from dog_lrm.render import intrinsics, render_gaussians
    from train_dog_lrm_ddp import _load_rgb_mask
    from train_dog_lrm_decomp import _label_grid
    import train_dog_lrm_fur as T

    seq, npy = retrieve_motion(args.text)
    print(f"motion {seq}")
    dsmal = load_dsmal()
    d = np.load(os.path.join(DSMAL_ROOT, "params", f"{args.dog}.npz"))

    # ---- IK against the dog's OWN D-SMAL skeleton (cached) -------------------
    os.makedirs(IK_CACHE, exist_ok=True)
    cache = os.path.join(IK_CACHE, f"{args.dog}_{seq}.theta.npy")
    if os.path.exists(cache):
        theta_seq = torch.tensor(np.load(cache), device=dev)
    else:
        target = load_motion_joints(npy, dev)
        theta_seq = dsmal_ik(dsmal, d, target, dev=dev)
        np.save(cache, theta_seq.cpu().numpy())
    Tm = theta_seq.shape[0]
    print(f"{Tm} motion frames")

    # ---- all-frame verts at the dog's training metric scale -------------------
    # training space: (fit_world - center) * nscale, fit_world = fwd * exp(log_scale)
    # + fit trans. For animation only the METRIC factor matters (orbit cam frames
    # freely): v = fwd(theta) * exp(log_scale) * nscale.
    sn = json.load(open(os.path.join(SCENE_ROOT, args.dog, "colmap/preprocess/scene_norm.json")))
    t = lambda k: torch.tensor(d["offset_" + k], device=dev)
    metric = float(torch.exp(t("log_scale"))) * float(sn["scale"])
    with torch.no_grad():
        V, J, _ = dsmal(beta=t("betas").expand(Tm, -1), betas_limbs=t("betas_limbs").expand(Tm, -1),
                        theta=theta_seq, trans=torch.zeros(Tm, 3, device=dev), get_skin=True,
                        vert_off_compact=t("vert_off_compact").expand(Tm, -1))
        verts_seq = V * metric                                              # [Tm,V,3]
        # IK aligns the dog to the MOTION world (AnimalML3D is not z-up); rotate
        # to z-up for the orbit cam: up = -(paws - body center), paws point down.
        paws = [10, 14, 20, 24]
        down = (J[:, paws].mean(dim=(0, 1)) - J[:, :35].mean(dim=(0, 1)))
        up = F.normalize(-down, dim=0)
        z = torch.tensor([0.0, 0.0, 1.0], device=dev)
        v_ax = torch.cross(up, z)
        s, c = v_ax.norm(), torch.dot(up, z)
        if float(s) > 1e-6:
            vx = torch.tensor([[0, -v_ax[2], v_ax[1]], [v_ax[2], 0, -v_ax[0]],
                               [-v_ax[1], v_ax[0], 0]], device=dev)
            R = torch.eye(3, device=dev) + vx + vx @ vx * ((1 - c) / (s * s))
            verts_seq = verts_seq @ R.T

    # ---- decomp model: features are pose-invariant -> forward once -----------
    scene = os.path.join(SCENE_ROOT, args.dog, "colmap")
    a = np.load(os.path.join(scene, "preprocess", "dsmal_anchors.npz"))
    model = DogLRMDecomp(torch.from_numpy(a["w_face"]), gaussians_per_point=args.gs_per_pt).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev), strict=False)
    model.eval()
    subdiv_M = build_subdiv(torch.from_numpy(a["faces"]), args.n_subdiv, dev)
    sub = lambda x: torch.stack([torch.sparse.mm(subdiv_M, x[b]) for b in range(x.shape[0])])
    faces_sub = T.subdivided_faces(torch.from_numpy(a["faces"]), args.n_subdiv).to(dev)

    frames = sorted(json.load(open(os.path.join(scene, "preprocess/cameras.json")))["frames"],
                    key=lambda f: f["name"])
    ref = frames[0]
    rgb_r, mask_r, _, _ = _load_rgb_mask(scene, ref, 8)
    label = _label_grid(scene, ref, mask_r)[None].to(dev)
    inp = F.interpolate(torch.from_numpy(rgb_r).permute(2, 0, 1)[None].to(dev), (518, 518),
                        mode="bilinear", align_corners=False)
    canon = torch.from_numpy(a["canon"])[None].to(dev)
    posed_fit = torch.from_numpy(a["posed"])[None].to(dev)
    K_pp = args.gs_per_pt

    def vert_frames(posed_sub, nbr_idx):
        """Per-vertex orthonormal frame [V,3,3] (tangent, bitangent, normal columns)."""
        n = T.orient_normals_outward(posed_sub, T.vertex_normals(posed_sub, faces_sub))[0]
        v = posed_sub[0]
        e = v[nbr_idx] - v
        tang = F.normalize(e - (e * n).sum(-1, keepdim=True) * n, dim=-1, eps=1e-6)
        bit = torch.cross(n, tang, dim=-1)
        return torch.stack([tang, bit, n], dim=-1)

    with torch.no_grad():
        gs = model(inp, label, canon, posed_fit, subdivide=sub)
        posed_fit_sub = sub(posed_fit)
        anchor_fit = posed_fit_sub.repeat_interleave(K_pp, dim=1)[0]        # [N*K,3]
        offset = gs["means"][0] - anchor_fit                                # in fit-world axes
        # fixed neighbor per vertex (for the tangent) + fit-pose reference frame:
        # offsets must rotate with the body or they stick out as fins after re-posing
        nbr_idx = torch.zeros(posed_fit_sub.shape[1], dtype=torch.long, device=dev)
        nbr_idx[faces_sub[:, 0]] = faces_sub[:, 1]
        nbr_idx[faces_sub[:, 1]] = faces_sub[:, 2]
        nbr_idx[faces_sub[:, 2]] = faces_sub[:, 0]
        F0 = vert_frames(posed_fit_sub, nbr_idx)                            # [V,3,3]
        off_local = torch.einsum("vij,vkj->vki", F0.transpose(1, 2),
                                 offset.view(-1, K_pp, 3))                  # local coords

    # ---- fur sway fields ------------------------------------------------------
    w_face_sub = sub(torch.from_numpy(a["w_face"]).view(1, -1, 1).to(dev))[0, :, 0]
    mob_v = (1.0 - w_face_sub).clamp(0, 1)
    body_diag = float((posed_fit[0].max(0).values - posed_fit[0].min(0).values).norm())
    if args.vlm_prior:
        prior = json.load(open(args.vlm_prior))
        Wsk = torch.as_tensor(dsmal.weights).detach().cpu().float()
        Lj = torch.full((Wsk.shape[1],), float(prior.get("default_cm", 4.0)))
        for j, cm in prior["joint_lengths_cm"].items():
            Lj[int(j)] = float(cm)
        L_cm = (Wsk @ Lj).view(1, -1, 1).to(dev)
        L_v = sub(L_cm)[0, :, 0] * (body_diag / float(prior["dog_bbox_diag_cm"]))
    else:
        L_v = torch.full_like(mob_v, 5.0 * body_diag / 70.0)                # ~5cm default
    mob = mob_v.repeat_interleave(K_pp).view(-1, 1)
    amp = (args.amp_frac * L_v).repeat_interleave(K_pp).view(-1, 1)

    # ---- auto-framed orbit camera over the whole motion -----------------------
    allv = verts_seq.reshape(-1, 3)
    ctr = allv.mean(0)
    diag = float((allv.max(0).values - allv.min(0).values).norm())
    best_az, best_area = 0.0, -1.0
    for az in range(0, 360, 30):
        c2w = look_at(ctr, float(az), -8.0, 1.45 * diag, dev)
        cam = (torch.linalg.inv(c2w)[:3, :3] @ allv.T).T
        ext = cam.max(0).values - cam.min(0).values
        if float(ext[0] * ext[1]) > best_area:
            best_area, best_az = float(ext[0] * ext[1]), float(az)
    res = 800
    c2w = look_at(ctr, best_az, -8.0, 1.45 * diag, dev)
    Kc = intrinsics(900.0, 900.0, res / 2, res / 2, dev)
    print(f"camera azimuth {best_az}")

    # ---- spring loop -----------------------------------------------------------
    bg = torch.full((3,), 0.45, device=dev)                  # gray: white dogs readable
    k_s, dt = args.stiffness, 1.0 / args.fps
    c_s = 2.0 * (k_s ** 0.5) * args.zeta
    wind_dir = F.normalize(torch.tensor([1.0, 0.1, 0.2], device=dev), dim=0)
    gen = torch.Generator().manual_seed(0)
    Nv = sub(posed_fit).shape[1]
    phi = (torch.rand(Nv, 1, generator=gen) * 2 * np.pi).repeat_interleave(K_pp, 0).to(dev)

    x_state = v_state = prev_roots = None
    frames_out = []
    with torch.no_grad():
        from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
        for ti in range(Tm * args.loops):
            posed_sub = sub(verts_seq[ti % Tm][None])
            Ft = vert_frames(posed_sub, nbr_idx)                            # [V,3,3]
            roots = posed_sub.repeat_interleave(K_pp, dim=1)[0]
            rest = torch.einsum("vij,vkj->vki", Ft, off_local).reshape(-1, 3)
            # rotate the Gaussian orientations with the surface too
            qR = matrix_to_quaternion(Ft @ F0.transpose(1, 2)).repeat_interleave(K_pp, 0)
            quats_t = quaternion_multiply(qR, gs["quats"][0])
            n_g = Ft[:, :, 2].repeat_interleave(K_pp, dim=0)
            exposure = (n_g @ wind_dir).clamp_min(0).unsqueeze(-1)
            if x_state is None:
                x_state, v_state, prev_roots = rest.clone(), torch.zeros_like(rest), roots
            v_root = (roots - prev_roots) / dt
            prev_roots = roots
            vmag = v_root.norm(dim=-1, keepdim=True)
            drag = -v_root / vmag.clamp_min(1e-6) * (vmag / (2.0 * body_diag)).clamp(0, 1)
            gust = torch.sin(torch.tensor(2 * np.pi * args.gust_hz * ti / args.fps) + phi.cpu()).to(dev)
            drive = (args.wind_frac * gust * exposure * wind_dir.view(1, 3)
                     + args.drag_frac * drag)
            force = k_s * (rest - x_state) - c_s * v_state + k_s * (amp * mob) * drive
            v_state = v_state + dt * force
            x_state = x_state + dt * v_state
            means_t = roots + x_state
            rgb, _ = render_gaussians(means_t, quats_t, gs["scales"][0],
                                      gs["opacities"][0], gs["rgb"][0], c2w, Kc, res, res, bg=bg)
            frames_out.append((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            if ti % 24 == 0:
                print(f"frame {ti}/{Tm * args.loops}", flush=True)

    out = args.out or f"exps/anim_decomp_{args.dog}_{seq}.mp4"
    imageio.mimwrite(out, frames_out, fps=args.fps, quality=8)
    print("saved", out)


if __name__ == "__main__":
    main()
