#!/usr/bin/env python3
"""Coat-depth secondary-motion demo on an animation-ready GS sample.

Loads canonical 3DGS + rigged template mesh (glb, 51-joint skin) + per-gaussian
vertex binding weights. Drives a procedural clip (tail wag / head sway / ear flop)
via LBS, then adds fur secondary motion as a per-gaussian phase lag proportional
to coat-depth h (signed distance to the template surface, computed post-hoc).
Renders [rigid LBS | +coat-depth lag] side-by-side -> gif."""
import json, math, os, struct, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from dog_lrm.render import intrinsics, render_gaussians, load_ply

dev = "cuda"
DATA = "train_data"
OUT = "exps/coatdepth_demo"
H_REF = 0.020          # h that reaches full lag [m]
TAU_MAX = 0.085        # max lag [s]
N_BINS = 6
T, FPS = 48, 24
WAG_HZ, SWAY_HZ = 2.2, 0.6

# ---------------- glb parsing (nodes, skin, mesh attributes) ----------------
CTYPE = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def parse_glb(path):
    raw = open(path, "rb").read()
    jlen = struct.unpack("<I", raw[12:16])[0]
    j = json.loads(raw[20:20 + jlen])
    boff = 20 + jlen + 8
    bin_chunk = raw[boff:boff + struct.unpack("<I", raw[20 + jlen:24 + jlen])[0]]

    def acc(i):
        a = j["accessors"][i]
        bv = j["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        n = a["count"] * NCOMP[a["type"]]
        arr = np.frombuffer(bin_chunk, dtype=np.dtype(CTYPE[a["componentType"]]).newbyteorder("<"),
                            count=n, offset=off)
        return arr.reshape(a["count"], -1).copy()

    prim = j["meshes"][0]["primitives"][0]["attributes"]
    verts = acc(prim["POSITION"]).astype(np.float32)
    vj = acc(prim["JOINTS_0"]).astype(np.int64)
    vw = acc(prim["WEIGHTS_0"]).astype(np.float32)
    vw /= vw.sum(1, keepdims=True).clip(1e-8)
    skin = j["skins"][0]
    ibm = acc(skin["inverseBindMatrices"]).astype(np.float32).reshape(-1, 4, 4).transpose(0, 2, 1)
    return j["nodes"], skin["joints"], ibm, verts, vj, vw


def node_local(n):
    if "matrix" in n:
        return np.array(n["matrix"], np.float32).reshape(4, 4).T
    t = np.array(n.get("translation", [0, 0, 0]), np.float32)
    q = np.array(n.get("rotation", [0, 0, 0, 1]), np.float32)  # xyzw
    s = np.array(n.get("scale", [1, 1, 1]), np.float32)
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R * s[None, :]
    M[:3, 3] = t
    return M


def world_matrices(nodes, extra_rot):
    """extra_rot: {node_id: 3x3 local delta rotation} applied after the rest local."""
    children = {i: n.get("children", []) for i, n in enumerate(nodes)}
    parents = {c: i for i, cs in children.items() for c in cs}
    W = {}

    def rec(i, pm):
        L = node_local(nodes[i]).copy()
        if i in extra_rot:
            D = np.eye(4, dtype=np.float32)
            D[:3, :3] = extra_rot[i]
            L = L @ D
        W[i] = pm @ L
        for c in children[i]:
            rec(c, W[i])

    roots = [i for i in range(len(nodes)) if i not in parents]
    for r in roots:
        rec(r, np.eye(4, dtype=np.float32))
    return W


def rot(axis, deg):
    a = np.asarray(axis, np.float32)
    a = a / np.linalg.norm(a)
    th = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], np.float32)
    return np.eye(3, dtype=np.float32) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


# ---------------- procedural clip: joint deltas at time t ----------------
def make_pose_deltas(nodes, name2id):
    """Returns f(t) -> {node_id: local delta rot}. The tail wags LEFT-RIGHT around
    the WORLD up axis (a local-axis wag unwinds a curled tail): the world rotation
    is conjugated into each joint's rest local frame, D = M^-1 R_world M. Base
    joint carries the swing, later joints add a small phase-lagged whip."""
    Wrest = world_matrices(nodes, {})

    def f(t):
        d = {}
        sway = math.sin(2 * math.pi * SWAY_HZ * t)
        for k, name in enumerate(["Tail_01", "Tail_02", "Tail_03", "Tail_04", "Tail_05"]):
            if name in name2id:
                nid = name2id[name]
                amp = TAIL_AMP if k == 0 else TAIL_WHIP
                ang = amp * math.sin(2 * math.pi * WAG_HZ * t - 0.35 * k)
                M = Wrest[nid][:3, :3]
                M = M / np.linalg.norm(M, axis=0, keepdims=True).clip(1e-8)
                d[nid] = M.T @ rot(WORLD_UP, ang) @ M
        for name, amp in [("neck", 6.0), ("head", 8.0)]:
            if name in name2id:
                d[name2id[name]] = rot(HEAD_AXIS, amp * sway)
        flop = math.sin(2 * math.pi * WAG_HZ * t + 1.0)
        for name in ["Ear_01_L", "Ear_02_L", "Ear_01_R", "Ear_02_R"]:
            if name in name2id:
                d[name2id[name]] = rot(EAR_AXIS, 6.0 * flop)
        return d

    return f


WORLD_UP = (0, 1, 0)                    # glb is Y-up
TAIL_AMP, TAIL_WHIP = 26.0, 7.0         # base swing / per-joint whip [deg]
HEAD_AXIS = (0, 1, 0)
EAR_AXIS = (1, 0, 0)

# ---------------- LBS + gaussian binding ----------------
def vertex_affines(nodes, joint_ids, ibm, extra_rot):
    W = world_matrices(nodes, extra_rot)
    Sk = np.stack([W[j] for j in joint_ids]) @ ibm          # [J,4,4]
    return torch.from_numpy(Sk).to(dev)


def skin_verts(Sk, verts_t, vj_t, vw_t):
    A = (Sk[vj_t] * vw_t[..., None, None]).sum(1)           # [V,4,4]
    v = torch.einsum("vij,vj->vi", A[:, :3, :3], verts_t) + A[:, :3, 3]
    return v, A


def rotmat_to_quat(R):
    """[N,3,3] -> wxyz quats (batch, torch)."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    t = 1 + m00 + m11 + m22
    q = torch.zeros(R.shape[0], 4, device=R.device)
    s = torch.sqrt(t.clamp_min(1e-8)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s
    return torch.nn.functional.normalize(q, dim=-1)


def orthonormalize(M):
    r1 = torch.nn.functional.normalize(M[:, :, 0], dim=-1)
    r2 = torch.nn.functional.normalize(M[:, :, 1] - (M[:, :, 1] * r1).sum(-1, True) * r1, dim=-1)
    r3 = torch.cross(r1, r2, dim=-1)
    return torch.stack([r1, r2, r3], dim=-1)


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def main():
    os.makedirs(OUT, exist_ok=True)
    nodes, joint_ids, ibm, verts, vj, vw = parse_glb(f"{DATA}/shba701_mesh_7k_noroot.glb")
    name2id = {nodes[i].get("name", ""): i for i in range(len(nodes))}
    pose_fn = make_pose_deltas(nodes, name2id)
    verts_t = torch.from_numpy(verts).to(dev)
    # JOINTS_0 indexes into skin.joints, and Sk is stacked in that same order
    vj_t = torch.from_numpy(vj).to(dev)
    vw_t = torch.from_numpy(vw).to(dev)

    gs = load_ply(f"{DATA}/shba701_canonical_gs_7k.ply", dev)
    # GS ply is Z-up; glb is Y-up: (x,y,z) -> (x,z,-y) == Rx(-90deg)
    m = gs["means"]
    gs["means"] = torch.stack([m[:, 0], m[:, 2], -m[:, 1]], 1)
    r_fix = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0, 0], device=dev)
    gs["quats"] = quat_mul(r_fix.expand_as(gs["quats"]), gs["quats"])

    # gaussian -> vertex binding (pad to K)
    wj = json.load(open(f"{DATA}/shba701_vertex_weights.json"))["weights"]
    K = max(len(w) for w in wj)
    bid = np.zeros((len(wj), K), np.int64)
    bw = np.zeros((len(wj), K), np.float32)
    for i, w in enumerate(wj):
        for k, (vi, wv) in enumerate(w):
            bid[i, k], bw[i, k] = vi, wv
    bw /= bw.sum(1, keepdims=True).clip(1e-8)
    bid_t = torch.from_numpy(bid).to(dev)
    bw_t = torch.from_numpy(bw).to(dev)

    # coat-depth h + lag bins
    h = np.load(os.environ.get("H_NPY", f"{OUT}/shba701_h.npy"))
    w_lag = np.clip(h / H_REF, 0.0, 1.0)
    w_lag = w_lag * w_lag * (3 - 2 * w_lag)                                 # smoothstep
    bins = np.rint(w_lag * (N_BINS - 1)).astype(np.int64)
    bins_t = torch.from_numpy(bins).to(dev)
    taus = np.linspace(0.0, TAU_MAX, N_BINS)
    print(f"lag bins: {np.bincount(bins, minlength=N_BINS)}")

    # rest-pose sanity: skinned verts at rest should match raw verts
    Sk0 = vertex_affines(nodes, joint_ids, ibm, {})
    v0, _ = skin_verts(Sk0, verts_t, vj_t, vw_t)
    err = (v0 - verts_t).norm(dim=-1)
    print(f"rest-pose skin residual: mean {err.mean()*1000:.3f}mm max {err.max()*1000:.3f}mm")

    def gaussians_at(tsec):
        Sk = vertex_affines(nodes, joint_ids, ibm, pose_fn(tsec))
        _, A = skin_verts(Sk, verts_t, vj_t, vw_t)                          # [V,4,4]
        Ag = (A[bid_t] * bw_t[..., None, None]).sum(1)                      # [N,4,4]
        mu = torch.einsum("nij,nj->ni", Ag[:, :3, :3], gs["means"]) + Ag[:, :3, 3]
        Rg = orthonormalize(Ag[:, :3, :3])
        q = quat_mul(rotmat_to_quat(Rg), gs["quats"])
        return mu, q

    # camera: fixed side view
    center = torch.tensor([0.0, 0.24, -0.05], device=dev)
    eye = torch.tensor([1.15, 0.38, -0.05], device=dev)
    fwd = torch.nn.functional.normalize(center - eye, dim=0)
    up = torch.tensor([0.0, 1.0, 0.0], device=dev)
    right = torch.nn.functional.normalize(torch.cross(fwd, up, dim=0), dim=0)
    upv = torch.cross(right, fwd, dim=0)
    c2w = torch.eye(4, device=dev)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, -upv, fwd, eye  # OpenCV: x right, y down, z fwd
    Wpx = Hpx = 640
    Kmat = intrinsics(700.0, 700.0, Wpx / 2, Hpx / 2, dev)
    white = torch.ones(3, device=dev)

    frames = []
    with torch.no_grad():
        for f in range(T):
            t = f / FPS
            mu0, q0 = gaussians_at(t)                                       # rigid LBS
            mu_l, q_l = mu0.clone(), q0.clone()
            for b in range(1, N_BINS):                                      # lagged variants
                sel = bins_t == b
                if sel.any():
                    mu_b, q_b = gaussians_at(t - taus[b])
                    mu_l[sel], q_l[sel] = mu_b[sel], q_b[sel]
            imgs = []
            for mu, q in [(mu0, q0), (mu_l, q_l)]:
                rgb, _ = render_gaussians(mu, q, gs["scales"], gs["opacities"], gs["rgb"],
                                          c2w, Kmat, Wpx, Hpx, bg=white)
                imgs.append((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            frames.append(np.concatenate(imgs, 1))
            if f == 0:
                Image.fromarray(frames[0]).save(f"{OUT}/frame0.png")
    Image.fromarray(frames[0]).save(
        f"{OUT}/lbs_vs_coatdepth.gif", save_all=True,
        append_images=[Image.fromarray(x) for x in frames[1:]], duration=1000 // FPS, loop=0)
    print(f"saved {OUT}/lbs_vs_coatdepth.gif  [left: rigid LBS | right: +coat-depth lag]")


if __name__ == "__main__":
    main()
