#!/usr/bin/env python3
"""Pom (00048-tenten) rig-driven fur video: body_rig.glb (35-joint skin, 32 morph
targets, bind pose != rest pose) + gaussians_3dgs.ply.

Differences vs the shiba pipeline (animate_fur_wind.py):
  - mesh canonical = morphed verts posed by REST node TRS (matches the GS); motion
    is transferred per-vertex as delta affines D = A(pose) @ A(rest)^-1
  - binding (K=8 nearest tris) and coat-depth h are computed in-script against the
    posed mesh — no external json/npy needed
  - joint names: tail_0..6 / neck / head / earL,earR / *paw,*mid,*upper / spine,core...
All fur-quality machinery matches the shiba version: local-shell rel, part amp prior,
tailness wind halving, float freeze, continuous lag + motion smoothing + soft clamp,
needle bend damp, coherent bend drive, camera-locked crosswind, PRUNE_OP/OP_GAMMA.
Env: MODE=dynamic|static BG=white|black HIRES=4k|1 NF PRUNE_OP OP_GAMMA
"""
import json, math, os, struct, subprocess, sys
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from dog_lrm.render import render_gaussians, load_ply
import animate_gs_coatdepth as base
from animate_fur_wind import knn_smooth, lookat

dev = "cuda"
DATA, OUT = "train_data", "exps/coatdepth_demo"
GLB, PLY = f"{DATA}/body_rig.glb", f"{DATA}/gaussians_3dgs.ply"
FPS = 24
N_BINS, TAU_MAX = base.N_BINS, base.TAU_MAX
WIND_AMP, ROT_AMP = 0.012, 0.35


def parse_glb_full(path):
    """base.parse_glb + faces and morph-applied vertices (mesh.weights)."""
    raw = open(path, "rb").read()
    jlen = struct.unpack("<I", raw[12:16])[0]
    j = json.loads(raw[20:20 + jlen])
    boff = 20 + jlen + 8
    bin_chunk = raw[boff:boff + struct.unpack("<I", raw[20 + jlen:24 + jlen])[0]]
    CT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
    NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

    def acc(i):
        a = j["accessors"][i]
        bv = j["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        n = a["count"] * NC[a["type"]]
        return np.frombuffer(bin_chunk, dtype=np.dtype(CT[a["componentType"]]).newbyteorder("<"),
                             count=n, offset=off).reshape(a["count"], -1).copy()

    nodes, joint_ids, ibm, verts, vj, vw = base.parse_glb(path)
    m = j["meshes"][0]
    p = m["primitives"][0]
    mw = np.array(m.get("weights", []), np.float32)
    for k, t in enumerate(p.get("targets", [])):
        if k < len(mw) and abs(mw[k]) > 1e-8:
            verts = verts + mw[k] * acc(t["POSITION"]).astype(np.float32)
    faces = acc(p["indices"]).astype(np.int64).reshape(-1, 3)
    return nodes, joint_ids, ibm, verts, vj, vw, faces


def part_scale(name):
    n = name.lower()
    if n.startswith("tail"): return 1.0
    if "ear" in n: return 0.5
    if any(k in n for k in ["head", "jaw"]): return 0.15
    if any(k in n for k in ["paw", "mid"]): return 0.2
    if any(k in n for k in ["upper", "scapula", "hip"]): return 0.45
    return 0.65                                           # spine/core/withers/neck/root


def part_hcap(name):
    n = name.lower()
    if n.startswith("tail"): return 0.070                 # pom plume is long
    if "ear" in n: return 0.035
    if any(k in n for k in ["head", "jaw"]): return 0.015
    if any(k in n for k in ["paw", "mid", "upper", "scapula", "hip"]): return 0.020
    return 0.040                                          # torso coat is deep on a pom


def make_pose_fn(nodes, name2id):
    Wrest = base.world_matrices(nodes, {})

    def f(t):
        d = {}
        sway = math.sin(2 * math.pi * base.SWAY_HZ * t)
        for k in range(7):
            name = f"tail_{k}"
            if name in name2id:
                nid = name2id[name]
                amp = base.TAIL_AMP if k == 0 else base.TAIL_WHIP
                ang = amp * math.sin(2 * math.pi * base.WAG_HZ * t - 0.35 * k)
                M = Wrest[nid][:3, :3]
                M = M / np.linalg.norm(M, axis=0, keepdims=True).clip(1e-8)
                d[nid] = M.T @ base.rot((0, 1, 0), ang) @ M
        for name, amp in [("neck", 6.0), ("head", 8.0)]:
            if name in name2id:
                d[name2id[name]] = base.rot((0, 1, 0), amp * sway)
        flop = math.sin(2 * math.pi * base.WAG_HZ * t + 1.0)
        for name in ["earL", "earR"]:
            if name in name2id:
                d[name2id[name]] = base.rot((1, 0, 0), 6.0 * flop)
        return d

    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    gs = load_ply(PLY, dev)
    prune_op = float(os.environ.get("PRUNE_OP", "0"))
    if prune_op > 0:
        keep = gs["opacities"] > prune_op
        gs = {k: v[keep] for k, v in gs.items()}
        print(f"pruned op<{prune_op}: kept {int(keep.sum())}/{len(keep)}")
    opg = float(os.environ.get("OP_GAMMA", "1.0"))
    if opg != 1.0:
        gs["opacities"] = gs["opacities"] ** opg
    m = gs["means"]                                       # GS ply is Z-up -> glb Y-up
    gs["means"] = torch.stack([m[:, 0], m[:, 2], -m[:, 1]], 1)
    r_fix = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0, 0], device=dev)
    gs["quats"] = base.quat_mul(r_fix.expand_as(gs["quats"]), gs["quats"])
    mu0 = gs["means"]
    xyz = mu0.cpu().numpy().astype(np.float64)

    nodes, joint_ids, ibm, verts, vj, vw, faces = parse_glb_full(GLB)
    name2id = {nodes[i].get("name", ""): i for i in range(len(nodes))}
    pose_fn = make_pose_fn(nodes, name2id)
    verts_t = torch.from_numpy(verts).to(dev)
    vj_t = torch.from_numpy(vj).to(dev)
    vw_t = torch.from_numpy(vw).to(dev)

    # rest (canonical) posed mesh + per-vertex rest affines
    def vertex_affines(extra):
        return base.vertex_affines(nodes, joint_ids, ibm, extra)

    Sk0 = vertex_affines({})
    _, A0 = base.skin_verts(Sk0, verts_t, vj_t, vw_t)
    A0_inv = torch.inverse(A0)
    v_rest = (torch.einsum("vij,vj->vi", A0[:, :3, :3], verts_t) + A0[:, :3, 3]).cpu().numpy()
    tm = trimesh.Trimesh(v_rest, faces, process=False)

    # binding: K=8 nearest tris of the posed mesh, inv-dist, 1/3 per vertex
    tc = tm.triangles.mean(1)
    td, ti = cKDTree(tc).query(xyz, k=8)
    tw = 1.0 / np.maximum(td, 1e-6)
    tw /= tw.sum(1, keepdims=True)
    KB = 24
    bid = np.zeros((len(xyz), KB), np.int64)
    bw = np.zeros((len(xyz), KB), np.float32)
    for i in range(len(xyz)):
        accd = {}
        for k in range(8):
            for vidx in faces[ti[i, k]]:
                accd[int(vidx)] = accd.get(int(vidx), 0.0) + tw[i, k] / 3.0
        for k, (vi, wv) in enumerate(sorted(accd.items(), key=lambda x: -x[1])[:KB]):
            bid[i, k], bw[i, k] = vi, wv
    bw /= bw.sum(1, keepdims=True).clip(1e-8)
    bid_t, bw_t = torch.from_numpy(bid).to(dev), torch.from_numpy(bw).to(dev)

    # coat-depth h against the posed mesh
    pts, fid = trimesh.sample.sample_surface(tm, 400000)
    d, idx = cKDTree(pts).query(xyz, k=1)
    fn = tm.face_normals[fid[idx]]
    h = d * np.sign(np.einsum("ij,ij->i", xyz - pts[idx], fn))
    h_s = knn_smooth(h, xyz)
    idx64 = cKDTree(xyz).query(xyz, k=64)[1]
    hn = np.clip(h_s, -0.08, 0.08)[idx64]
    h_lo = knn_smooth(np.percentile(hn, 20, axis=1), xyz)
    h_hi = knn_smooth(np.percentile(hn, 90, axis=1), xyz)
    t_loc = np.maximum(h_hi - h_lo, 0.004)
    rel = np.clip((h_s - h_lo) / t_loc, 0.0, 1.0)
    w = rel * rel

    ps = np.array([part_scale(nodes[jj].get("name", "")) for jj in joint_ids], np.float32)
    vscale = (vw * ps[vj]).sum(1)
    amp_scale = knn_smooth((bw * vscale[bid]).sum(1), xyz, k=16, iters=1)
    istail = np.array([1.0 if nodes[jj].get("name", "").startswith("tail") else 0.0
                       for jj in joint_ids], np.float32)
    tailness = knn_smooth((bw * (vw * istail[vj]).sum(1)[bid]).sum(1), xyz, k=16, iters=1)
    wind_scale = 1.0 - 0.55 * tailness
    hc = np.array([part_hcap(nodes[jj].get("name", "")) for jj in joint_ids], np.float32)
    gcap = knn_smooth((bw * (vw * hc[vj]).sum(1)[bid]).sum(1), xyz, k=16, iters=1)
    w = w * np.exp(-np.maximum(h_s - gcap, 0.0) / 0.010)
    print(f"h med {np.median(h_s)*1000:.1f}mm p90 {np.percentile(h_s,90)*1000:.1f}mm | "
          f"w>0.3 {(w>0.3).mean():.3f} | tailness>0.5 {(tailness>0.5).mean():.3f}")
    w_t = torch.from_numpy((w * amp_scale * wind_scale).astype(np.float32)).to(dev)[:, None]

    # normals from posed mesh via top-1 binding
    nrm_np = knn_smooth(np.asarray(tm.vertex_normals, np.float32)[bid[:, 0]], xyz, k=16, iters=1)
    nrm = torch.nn.functional.normalize(torch.from_numpy(nrm_np).to(dev), dim=-1)

    # wind (Y-up frame, head +x)
    wind = torch.nn.functional.normalize(torch.tensor([0.8, 0.15, 0.45], device=dev), dim=0)[None]
    jit = torch.sin(mu0 @ torch.tensor([127.1, 311.7, 74.7], device=dev)) * math.pi
    u1 = torch.sin(mu0 @ (torch.tensor([127.1, 311.7, 74.7], device=dev) * 2.7)) * 0.5 + 0.5
    kx = mu0 @ (torch.tensor([1.0, 0.3, 0.5], device=dev) * 2 * math.pi / 0.05)
    bend_axis0 = torch.nn.functional.normalize(torch.linalg.cross(nrm, wind.expand_as(nrm)), dim=-1)
    s_sorted = gs["scales"].sort(dim=1).values
    rot_damp = (4.0 / (s_sorted[:, 2] / s_sorted[:, 1].clamp_min(1e-9))).clamp(0.25, 1.0)

    def wind_fields(t):
        gust = 0.55 + 0.45 * math.sin(2 * math.pi * 0.35 * t + 1.2)
        wave = torch.sin(2 * math.pi * 1.1 * t - kx + 0.3 * jit)
        ind = torch.sin(2 * math.pi * (1.6 + 1.2 * u1) * t + 7.0 * jit)
        return gust, wave, ind

    def wind_disp(t, wdir, boost):
        gust, wave, ind = wind_fields(t)
        drive = 0.6 * wave + 0.4 * ind
        lift = nrm * (boost * WIND_AMP * 0.6 * gust * drive)[:, None]
        slide = wdir[None] * (boost * WIND_AMP * 0.7 * gust * drive)[:, None]
        return w_t * (lift + slide)

    def wind_quat(t, q, bend, boost):
        gust, wave, ind = wind_fields(t)
        drive = 0.85 * wave + 0.15 * ind
        ang = 0.5 * boost * ROT_AMP * rot_damp * w_t[:, 0] * gust * drive
        qd = torch.cat([torch.cos(ang)[:, None], bend * torch.sin(ang)[:, None]], -1)
        return base.quat_mul(qd, q)

    def gaussians_at(tsec):
        Sk = vertex_affines(pose_fn(tsec))
        _, A = base.skin_verts(Sk, verts_t, vj_t, vw_t)
        D = A @ A0_inv                                     # rest-world -> posed-world
        Dg = (D[bid_t] * bw_t[..., None, None]).sum(1)
        mu = torch.einsum("nij,nj->ni", Dg[:, :3, :3], mu0) + Dg[:, :3, 3]
        q = base.quat_mul(base.rotmat_to_quat(base.orthonormalize(Dg[:, :3, :3])), gs["quats"])
        return mu, q

    fbin = torch.from_numpy((w * (N_BINS - 1)).astype(np.float32)).to(dev)
    b0 = fbin.floor().long().clamp(0, N_BINS - 2)
    frac = (fbin - b0.float())[:, None]
    taus = np.linspace(0.0, TAU_MAX, N_BINS)
    ar = torch.arange(len(w), device=dev)
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
        d = mu - mu_rigid
        d = 0.5 * d + 0.5 * d[idx16_t].mean(1)
        d = 0.5 * d + 0.5 * d[idx16_t].mean(1)
        n = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        mu = mu_rigid + d * (d_cap * torch.tanh(n / d_cap) / n)
        qa, qb = qs[b0, ar], qs[b0 + 1, ar]
        qb = torch.where((qa * qb).sum(-1, True) < 0, -qb, qb)
        q = torch.nn.functional.normalize(torch.lerp(qa, qb, frac), dim=-1)
        return mu, q

    # ---- global alignment to the shiba pose (user-measured in Blender by hand-
    # aligning the pom onto the shiba GS: Z-up XYZ-Euler + location). Conjugate the
    # Blender Z-up transform into this Y-up render frame: R_r = P R_b P^T, t_r = P t_b,
    # where P: (x,y,z)_zup -> (x,z,-y)_yup. Applied at display time (motion math
    # stays in the original frame). Override via POSE_EULER / POSE_LOC. ----
    ex, ey, ez = [float(x) for x in os.environ.get("POSE_EULER", "0,-24.425,-80.356").split(",")]
    lx, ly, lz = [float(x) for x in os.environ.get("POSE_LOC", "-0.01385,0.147,0.32824").split(",")]
    P = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)
    Rb = base.rot((0, 0, 1), ez) @ base.rot((0, 1, 0), ey) @ base.rot((1, 0, 0), ex)  # Rz@Ry@Rx
    Rf = torch.tensor((P @ Rb @ P.T).astype(np.float32), device=dev)
    tf = torch.tensor((P @ np.array([lx, ly, lz])).astype(np.float32), device=dev)
    rq = base.rotmat_to_quat(Rf[None])[0]

    def display(mu, q):
        return mu @ Rf.T + tf, base.quat_mul(rq.expand(len(q), 4), q)

    # ---- orbit render ----
    four_k = os.environ.get("HIRES", "4k") == "4k"
    mode = os.environ.get("MODE", "dynamic")
    bgname = os.environ.get("BG", "black")
    bgc = torch.ones(3, device=dev) if bgname == "white" else torch.zeros(3, device=dev)
    W2, H2 = (3840, 2160) if four_k else (1920, 1080)
    FX = 4700.0 if four_k else 2350.0
    NF = int(os.environ.get("NF", "480"))
    BOOST = 1.35
    ctr = [float(c) for c in ((xyz @ Rf.T.cpu().numpy().astype(np.float64))
                              + tf.cpu().numpy().astype(np.float64)).mean(0)]
    path = f"{OUT}/pomrig_{mode}_{bgname}_{'4k' if four_k else 'hires'}.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W2}x{H2}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-preset", "medium" if four_k else "slow", "-crf", "17",
         "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    with torch.no_grad():
        for f in range(NF):
            th = 2 * math.pi * f / NF
            eye = [ctr[0] + 1.25 * math.cos(th), ctr[1] + 0.20, ctr[2] + 1.25 * math.sin(th)]
            c2w, K = lookat(eye, ctr, FX, W2, H2)
            if mode == "static":
                mu, q = mu0, gs["quats"]
            else:
                wdir = torch.nn.functional.normalize(0.5 * wind[0] + 0.65 * c2w[:3, 0], dim=0)
                bend = torch.nn.functional.normalize(
                    torch.linalg.cross(nrm, wdir[None].expand_as(nrm)), dim=-1)
                mu, q = lagged(f / FPS, extra=lambda ts, m_, q_: (
                    m_ + wind_disp(ts, wdir, BOOST), wind_quat(ts, q_, bend, BOOST)))
            mu, q = display(mu, q)
            rgb, _ = render_gaussians(mu, q, gs["scales"], gs["opacities"], gs["rgb"],
                                      c2w, K, W2, H2, bg=bgc)
            proc.stdin.write((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print("saved", path)


if __name__ == "__main__":
    main()
