#!/usr/bin/env python3
"""Fur-only dynamics close-ups on the shba701 animation-ready sample. (v2)

v2 changes vs v1:
  - kNN-smoothed coat-depth h (k=16, 2 iters) -> no speckle in motion weights
  - weight band shifted: w = smoothstep((h-5mm)/20mm) -> roots strictly rigid,
    kills the "whole surface swimming" look
  - wind: coherent ripple share reduced (45%), per-gaussian frequency/phase
    jitter dominates -> reads as strands fluttering, not surface waves
  - wag lag: continuous per-gaussian tau via lerp between bin poses (no
    quantization jumps -> no detached wisps)

Outputs: wind_v6.gif [tail | back], wag_close_v6.gif, combo_v6.gif [full | tail].
"""
import json, math, os, sys
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from dog_lrm.render import intrinsics, render_gaussians, load_ply
import animate_gs_coatdepth as base

dev = "cuda"
DATA, OUT = "train_data", "exps/coatdepth_demo"
GS_PLY = os.environ.get("GS_PLY", f"{DATA}/shba701_canonical_gs_7k.ply")
H_NPY = os.environ.get("H_NPY", f"{OUT}/shba701_h.npy")
WJ_JSON = os.environ.get("WJ_JSON", f"{DATA}/shba701_vertex_weights.json")
HIRES = os.environ.get("HIRES")             # "1" = 1080p combo mp4, "4k" = 2160p
H_LO, H_HI = 0.005, 0.025      # rigid below 5mm, full motion at 25mm
TAU_MAX = base.TAU_MAX
N_BINS = base.N_BINS
T, FPS = 72, 24
WIND_DIR = torch.tensor([0.55, 0.15, 0.82])
WIND_AMP = 0.012


def lookat(eye, center, fx, W, H):
    eye = torch.tensor(eye, device=dev)
    center = torch.tensor(center, device=dev)
    fwd = torch.nn.functional.normalize(center - eye, dim=0)
    up = torch.tensor([0.0, 1.0, 0.0], device=dev)
    right = torch.nn.functional.normalize(torch.cross(fwd, up, dim=0), dim=0)
    upv = torch.cross(right, fwd, dim=0)
    c2w = torch.eye(4, device=dev)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, -upv, fwd, eye
    return c2w, intrinsics(fx, fx, W / 2, H / 2, dev)


def knn_smooth(vals, xyz, k=16, iters=2):
    idx = cKDTree(xyz).query(xyz, k=k)[1]
    v = vals.copy()
    for _ in range(iters):
        v = v[idx].mean(1)
    return v


def smoothband(h, lo, hi):
    x = np.clip((h - lo) / (hi - lo), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def hash_vec(mu, seed):
    return torch.sin(mu @ (torch.tensor([127.1, 311.7, 74.7], device=dev) * seed))


def save_gif(frames, path):
    Image.fromarray(frames[0]).save(path, save_all=True,
                                    append_images=[Image.fromarray(x) for x in frames[1:]],
                                    duration=1000 // FPS, loop=0)
    print("saved", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    gs = load_ply(GS_PLY, dev)
    opg = float(os.environ.get("OP_GAMMA", "1.0"))
    if opg != 1.0:                              # brighten semi-transparent fringe
        gs["opacities"] = gs["opacities"] ** opg
    m = gs["means"]
    gs["means"] = torch.stack([m[:, 0], m[:, 2], -m[:, 1]], 1)
    r_fix = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0, 0], device=dev)
    gs["quats"] = base.quat_mul(r_fix.expand_as(gs["quats"]), gs["quats"])
    mu0 = gs["means"]
    xyz = mu0.cpu().numpy()

    h = np.load(H_NPY)
    h_s = knn_smooth(h, xyz)
    # local relative coat coordinate, normalized WITHIN the local gaussian shell:
    # the template mesh is not the skin (on the flank it sits at/outside the fur
    # surface, h<=0), so h=0 must not be treated as the root. Instead the local
    # innermost gaussians (p20 of h over 64 neighbors) are roots, the outermost
    # (p90) are tips, wherever the mesh happens to sit.
    idx64 = cKDTree(xyz).query(xyz, k=64)[1]
    hn = np.clip(h_s, -0.06, 0.06)[idx64]
    h_lo = knn_smooth(np.percentile(hn, 20, axis=1), xyz)
    h_hi = knn_smooth(np.percentile(hn, 90, axis=1), xyz)
    t_loc = np.maximum(h_hi - h_lo, 0.004)               # local shell thickness
    rel = np.clip((h_s - h_lo) / t_loc, 0.0, 1.0)
    w = rel * rel                                        # cantilever-ish: root rigid, tip max

    # amplitude prior: fur length is a body-part property (the GS shell is a thin
    # ~4mm crust everywhere, so shell thickness says nothing about fur length).
    # Route joint weights through the 2-level binding: joint -> vertex -> gaussian.
    nodes, joint_ids, ibm, verts, vj, vw = base.parse_glb(f"{DATA}/shba701_mesh_7k_noroot.glb")
    wj = json.load(open(WJ_JSON))["weights"]
    K = max(len(x) for x in wj)
    bid = np.zeros((len(wj), K), np.int64)
    bw = np.zeros((len(wj), K), np.float32)
    for i, x in enumerate(wj):
        for kk, (vi, wv) in enumerate(x):
            bid[i, kk], bw[i, kk] = vi, wv
    bw /= bw.sum(1, keepdims=True).clip(1e-8)

    def part_scale(name):
        n = name.lower()
        if n.startswith("tail"): return 1.0
        if "ear" in n: return 0.5
        if any(k in n for k in ["head", "nose", "eye", "mouth", "tongue"]): return 0.15
        if any(k in n for k in ["claw", "foot", "shin"]): return 0.2
        if any(k in n for k in ["leg", "thigh", "hip"]): return 0.45
        return 0.65                                       # spine / neck

    ps = np.array([part_scale(nodes[j].get("name", "")) for j in joint_ids], np.float32)
    vscale = (vw * ps[vj]).sum(1)                         # per-vertex fur-length scale
    amp_scale = knn_smooth((bw * vscale[bid]).sum(1), xyz, k=16, iters=1)
    # tailness: the tail already carries wag+lag energy; full wind on top over-
    # energizes the plume (scraggly blown-out tips), so halve wind there
    istail = np.array([1.0 if nodes[j].get("name", "").startswith("Tail") else 0.0
                       for j in joint_ids], np.float32)
    vtail = (vw * istail[vj]).sum(1)
    tailness = knn_smooth((bw * vtail[bid]).sum(1), xyz, k=16, iters=1)
    wind_scale = 1.0 - 0.55 * tailness

    # freeze canonical floaters: GS sitting far above the plausible fur length of
    # their body part (e.g. the tuft 4-6cm over the rump) ride rigid LBS only —
    # any lag/wind on them reads as scraggly blown-out spikes
    def part_hcap(name):
        n = name.lower()
        if n.startswith("tail"): return 0.060
        if "ear" in n: return 0.030
        if any(k in n for k in ["head", "nose", "eye", "mouth", "tongue"]): return 0.010
        if any(k in n for k in ["claw", "foot", "shin", "leg", "thigh", "hip"]): return 0.015
        return 0.025                                      # spine / neck
    hc = np.array([part_hcap(nodes[j].get("name", "")) for j in joint_ids], np.float32)
    vcap = (vw * hc[vj]).sum(1)
    gcap = knn_smooth((bw * vcap[bid]).sum(1), xyz, k=16, iters=1)
    float_damp = np.exp(-np.maximum(h_s - gcap, 0.0) / 0.010)
    w = w * float_damp
    print(f"w>0.3: {(w>0.3).mean():.3f}  w>0.7: {(w>0.7).mean():.3f}  "
          f"amp_scale flank~{np.median(amp_scale[np.abs(xyz[:,2]+0.05)<0.08]):.2f} "
          f"tail~{np.median(amp_scale[xyz[:,2]<-0.30]):.2f}")
    w_t = torch.from_numpy((w * amp_scale * wind_scale).astype(np.float32)).to(dev)[:, None]

    # per-gaussian outward normal from the template mesh (for ruffle lift + bend axis)
    import trimesh
    tm = trimesh.load(f"{DATA}/shba701_mesh_7k_noroot.glb", process=False, force="mesh")
    vn = np.asarray(tm.vertex_normals, dtype=np.float32)
    wj0 = json.load(open(WJ_JSON))["weights"]
    top1 = np.array([x[0][0] for x in wj0], np.int64)
    nrm_np = knn_smooth(vn[top1], xyz, k=16, iters=1)
    nrm = torch.nn.functional.normalize(torch.from_numpy(nrm_np).to(dev), dim=-1)

    # ---------------- wind (body strictly static) ----------------
    wind = torch.nn.functional.normalize(WIND_DIR, dim=0).to(dev)[None]
    jit = hash_vec(mu0, 1.0) * math.pi                     # phase jitter
    u1 = hash_vec(mu0, 2.7) * 0.5 + 0.5                    # per-gaussian freq in [0,1]
    kx = mu0 @ (torch.tensor([0.3, 0.5, 1.0], device=dev) * 2 * math.pi / 0.05)
    bend_axis = torch.nn.functional.normalize(torch.linalg.cross(nrm, wind.expand_as(nrm)), dim=-1)
    ROT_AMP = 0.35                                          # max splat bend [rad] at tips

    # needle-shaped splats poking out of the silhouette when rotated -> damp their
    # bend by elongation (longest/mid axis; p90 is ~7:1 on this dog)
    s_sorted = gs["scales"].sort(dim=1).values
    rot_damp = (4.0 / (s_sorted[:, 2] / s_sorted[:, 1].clamp_min(1e-9))).clamp(0.25, 1.0)

    def wind_fields(t):
        gust = 0.55 + 0.45 * math.sin(2 * math.pi * 0.35 * t + 1.2)
        wave = torch.sin(2 * math.pi * 1.1 * t - kx + 0.3 * jit)   # 5cm travelling ruffle
        ind = torch.sin(2 * math.pi * (1.6 + 1.2 * u1) * t + 7.0 * jit)
        return gust, wave, ind

    def wind_disp(t, wdir=None, boost=1.0):
        wd = wind if wdir is None else wdir[None]
        gust, wave, ind = wind_fields(t)
        drive = 0.6 * wave + 0.4 * ind
        lift = nrm * (boost * WIND_AMP * 0.6 * gust * drive)[:, None]   # normal ruffle
        slide = wd * (boost * WIND_AMP * 0.7 * gust * drive)[:, None]
        return w_t * (lift + slide)

    def wind_quat(t, q, bend=None, boost=1.0):
        ba = bend_axis if bend is None else bend
        gust, wave, ind = wind_fields(t)
        drive = 0.85 * wave + 0.15 * ind        # bend coherently: lone rotated needles = spikes
        ang = 0.5 * boost * ROT_AMP * rot_damp * w_t[:, 0] * gust * drive   # half-angle
        qd = torch.cat([torch.cos(ang)[:, None], ba * torch.sin(ang)[:, None]], -1)
        return base.quat_mul(qd, q)

    views = [lookat([0.42, 0.52, -0.72], [0.0, 0.35, -0.30], 950.0, 640, 640),
             lookat([0.52, 0.62, 0.05], [0.0, 0.32, -0.05], 950.0, 640, 640)]
    white = torch.ones(3, device=dev)

    def render(mu, q, c2w, K, W=640, H=640, bg=None):
        rgb, _ = render_gaussians(mu, q, gs["scales"], gs["opacities"], gs["rgb"],
                                  c2w, K, W, H, bg=white if bg is None else bg)
        return (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    if not HIRES:
        frames = []
        with torch.no_grad():
            for f in range(T):
                t = f / FPS
                mu = mu0 + wind_disp(t)
                q = wind_quat(t, gs["quats"])
                frames.append(np.concatenate([render(mu, q, c, K) for c, K in views], 1))
        save_gif(frames, f"{OUT}/wind_v6.gif")

    # ---------------- LBS wag with continuous coat-depth lag ----------------
    name2id = {nodes[i].get("name", ""): i for i in range(len(nodes))}
    pose_fn = base.make_pose_deltas(nodes, name2id)
    verts_t = torch.from_numpy(verts).to(dev)
    vj_t = torch.from_numpy(vj).to(dev)
    vw_t = torch.from_numpy(vw).to(dev)
    bid_t, bw_t = torch.from_numpy(bid).to(dev), torch.from_numpy(bw).to(dev)

    # continuous per-gaussian lag: position of w on the bin axis
    fbin = torch.from_numpy((w * (N_BINS - 1)).astype(np.float32)).to(dev)
    b0 = fbin.floor().long().clamp(0, N_BINS - 2)
    frac = (fbin - b0.float())[:, None]
    taus = np.linspace(0.0, TAU_MAX, N_BINS)
    ar = torch.arange(len(w), device=dev)

    def gaussians_at(tsec):
        Sk = base.vertex_affines(nodes, joint_ids, ibm, pose_fn(tsec))
        _, A = base.skin_verts(Sk, verts_t, vj_t, vw_t)
        Ag = (A[bid_t] * bw_t[..., None, None]).sum(1)
        mu = torch.einsum("nij,nj->ni", Ag[:, :3, :3], gs["means"]) + Ag[:, :3, 3]
        q = base.quat_mul(base.rotmat_to_quat(base.orthonormalize(Ag[:, :3, :3])), gs["quats"])
        return mu, q

    # floater control: smooth the secondary-motion field over neighbors, then
    # soft-clamp its magnitude (stray gaussians far off the tail otherwise get
    # flung: full lag weight x large lever arm from the wag axis)
    idx16_t = torch.from_numpy(idx64[:, :16].copy()).to(dev)
    d_cap = torch.from_numpy((0.02 + 0.035 * amp_scale).astype(np.float32)).to(dev)[:, None]

    def lagged(t, extra=None):
        mus, qs, mu_rigid = [], [], None
        for b in range(N_BINS):
            mu_b, q_b = gaussians_at(t - taus[b])
            if b == 0:
                mu_rigid = mu_b
            if extra is not None:
                mu_b, q_b = extra(t - taus[b], mu_b, q_b)
            mus.append(mu_b)
            qs.append(q_b)
        mus, qs = torch.stack(mus), torch.stack(qs)
        mu = torch.lerp(mus[b0, ar], mus[b0 + 1, ar], frac)
        d = mu - mu_rigid                                    # secondary offset vs rigid LBS
        d = 0.5 * d + 0.5 * d[idx16_t].mean(1)               # motion-field smoothness
        d = 0.5 * d + 0.5 * d[idx16_t].mean(1)
        n = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        mu = mu_rigid + d * (d_cap * torch.tanh(n / d_cap) / n)   # soft clamp
        qa, qb = qs[b0, ar], qs[b0 + 1, ar]
        qb = torch.where((qa * qb).sum(-1, True) < 0, -qb, qb)
        q = torch.nn.functional.normalize(torch.lerp(qa, qb, frac), dim=-1)
        return mu, q

    if HIRES:                                   # hi-res mp4 of the combo, 360-degree orbit
        import subprocess
        four_k = HIRES == "4k"
        mode = os.environ.get("MODE", "dynamic")        # dynamic | static (no motion at all)
        bgname = os.environ.get("BG", "white")          # white | black
        bgc = torch.ones(3, device=dev) if bgname == "white" else torch.zeros(3, device=dev)
        W2, H2 = (3840, 2160) if four_k else (1920, 1080)
        FX = 4700.0 if four_k else 2350.0
        NF = 480                                # 20s = one very slow full orbit
        ctr = [0.0, 0.24, -0.06]
        path = f"{OUT}/kotori_{mode}_{bgname}_{'4k' if four_k else 'hires'}.mp4"
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W2}x{H2}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264",
             "-preset", "medium" if four_k else "slow", "-crf", "17",
             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        BOOST = 1.35                        # orbiting global motion masks small fur motion
        with torch.no_grad():
            for f in range(NF):
                th = 2 * math.pi * f / NF
                eye = [ctr[0] + 1.30 * math.cos(th), 0.46, ctr[2] + 1.30 * math.sin(th)]
                cam, Kc = lookat(eye, ctr, FX, W2, H2)
                if mode == "static":
                    mu, q = gs["means"], gs["quats"]
                else:
                    # camera-locked crosswind so every view angle sees transverse motion
                    wdir = torch.nn.functional.normalize(0.5 * wind[0] + 0.65 * cam[:3, 0], dim=0)
                    bend = torch.nn.functional.normalize(
                        torch.linalg.cross(nrm, wdir[None].expand_as(nrm)), dim=-1)
                    mu, q = lagged(f / FPS, extra=lambda ts, m, qq: (
                        m + wind_disp(ts, wdir, BOOST), wind_quat(ts, qq, bend, BOOST)))
                proc.stdin.write(render(mu, q, cam, Kc, W2, H2, bg=bgc).tobytes())
        proc.stdin.close()
        proc.wait()
        print("saved", path)
        return

    c2w, Km = lookat([0.55, 0.55, -0.65], [0.0, 0.36, -0.30], 900.0, 720, 640)
    frames = []
    with torch.no_grad():
        for f in range(48):
            mu, q = lagged(f / FPS)
            frames.append(render(mu, q, c2w, Km, 720, 640))
    save_gif(frames, f"{OUT}/wag_close_v6.gif")

    # ---------------- combo: skeletal wag + wind, [full body | tail] ----------------
    vf = lookat([1.15, 0.38, -0.05], [0.0, 0.24, -0.05], 700.0, 640, 640)
    vt = lookat([0.55, 0.55, -0.65], [0.0, 0.36, -0.30], 900.0, 640, 640)
    frames = []
    with torch.no_grad():
        for f in range(T):
            t = f / FPS
            mu, q = lagged(t, extra=lambda ts, m, qq: (m + wind_disp(ts), wind_quat(ts, qq)))
            frames.append(np.concatenate([render(mu, q, c, K) for c, K in [vf, vt]], 1))
    save_gif(frames, f"{OUT}/combo_v6.gif")


if __name__ == "__main__":
    main()
