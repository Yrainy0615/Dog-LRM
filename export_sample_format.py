#!/usr/bin/env python3
"""Export one dog (default bear) into the tools/sample_data exchange format:
  strands_lines.ply       polyline per strand (xyz+normal+rgb verts, edges)
  strand_gaussians.ply    one gaussian per strand segment, deg-3 SH layout + strand_id
  gaussians_with_fur.ply   body+fur 3DGS (standard, deg-0)
  gaussians_no_fur.ply     body-only 3DGS
  body_rig.glb            SMAL mesh (canon frame) as a skinned glTF rig (35 joints)

All five share the canonical D-SMAL frame (the frame the gaussians live in).

  PATH=$ENV/bin:$PATH python export_sample_format.py --dog 00062-bear \
       --ckpt exps/dog_lrm_fur_v6/model.pt --v6_repr --fur_aspect 10 --out tools/sample_data/bear_example
"""
import argparse, json, os, struct, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.abspath("."))
from dog_lrm.model_fur import DogLRMFurV2, load_fur_ckpt
from dog_lrm.render import intrinsics, save_ply, _SH0
from dog_lrm.smal_model import build_subdiv, subdivided_faces
from train_dog_lrm_ddp import _load_rgb_mask
from train_dog_lrm_decomp import _label_grid
from train_dog_lrm_fur_v2 import FurScenes, list_scenes

JOINT_NAMES = ['root', 'spine_0', 'spine_1', 'spine_2', 'core_0', 'core_1', 'withers',
               'frontL_scapula', 'frontL_upper', 'frontL_mid', 'frontL_paw',
               'frontR_scapula', 'frontR_upper', 'frontR_mid', 'frontR_paw',
               'neck', 'head', 'backL_hip', 'backL_upper', 'backL_mid', 'backL_paw',
               'backR_hip', 'backR_upper', 'backR_mid', 'backR_paw',
               'tail_0', 'tail_1', 'tail_2', 'tail_3', 'tail_4', 'tail_5', 'tail_6',
               'jaw', 'earL', 'earR']


def write_strands_lines(path, pts, vrgb, vnrm):
    """pts [N,K,3], vrgb [N,K,3] in [0,1], vnrm [N,K,3]. Polyline per strand."""
    from plyfile import PlyData, PlyElement
    N, K, _ = pts.shape
    xyz = pts.reshape(-1, 3).astype(np.float32)
    nrm = vnrm.reshape(-1, 3).astype(np.float32)
    rgb = (vrgb.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    vert = np.empty(N * K, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, f in enumerate(["x", "y", "z"]):
        vert[f] = xyz[:, i]
    for i, f in enumerate(["nx", "ny", "nz"]):
        vert[f] = nrm[:, i]
    for i, f in enumerate(["red", "green", "blue"]):
        vert[f] = rgb[:, i]
    # edges: (i*K+k, i*K+k+1) for k in 0..K-2
    base = (np.arange(N) * K)[:, None]
    v1 = (base + np.arange(K - 1)[None]).reshape(-1)
    v2 = (base + np.arange(1, K)[None]).reshape(-1)
    edge = np.empty(N * (K - 1), dtype=[("vertex1", "i4"), ("vertex2", "i4")])
    edge["vertex1"] = v1
    edge["vertex2"] = v2
    PlyData([PlyElement.describe(vert, "vertex"), PlyElement.describe(edge, "edge")]).write(path)
    return N * K, N * (K - 1)


def write_strand_gaussians(path, means, sh, op, scales, quats, strand_id):
    """Per-segment gaussians, INRIA 3DGS deg-3 SH layout (channel-major f_rest) + strand_id.
    sh [Nf,4,3] (deg-1: DC + 3 directional). Higher orders zero-padded to deg-3 (45 rest)."""
    from plyfile import PlyData, PlyElement
    Nf = means.shape[0]
    f_dc = sh[:, 0, :].astype(np.float32)                          # [Nf,3]
    f_rest = np.zeros((Nf, 45), np.float32)                        # 15 coeffs x 3 channels
    for c in range(3):                                             # channel-major: R then G then B
        f_rest[:, c * 15 + 0] = sh[:, 1, c]                        # our 3 deg-1 coeffs -> first 3 slots
        f_rest[:, c * 15 + 1] = sh[:, 2, c]
        f_rest[:, c * 15 + 2] = sh[:, 3, c]
    opl = np.log(np.clip(op, 1e-4, 1 - 1e-4) / (1 - np.clip(op, 1e-4, 1 - 1e-4))).astype(np.float32)
    scl = np.log(np.clip(scales, 1e-8, None)).astype(np.float32)
    fields = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
              + [f"f_rest_{i}" for i in range(45)] + ["opacity"]
              + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    dtype = [(f, "f4") for f in fields] + [("strand_id", "i4")]
    el = np.empty(Nf, dtype=dtype)
    el["x"], el["y"], el["z"] = means[:, 0], means[:, 1], means[:, 2]
    el["nx"] = el["ny"] = el["nz"] = 0.0
    for i in range(3):
        el[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(45):
        el[f"f_rest_{i}"] = f_rest[:, i]
    el["opacity"] = opl[:, 0] if opl.ndim == 2 else opl
    for i in range(3):
        el[f"scale_{i}"] = scl[:, i]
    for i in range(4):
        el[f"rot_{i}"] = quats[:, i]
    el["strand_id"] = strand_id.astype(np.int32)
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


def _pad(b):                                                       # 4-byte align for glTF
    return b + b"\x00" * ((4 - len(b) % 4) % 4)


def write_body_rig_glb(path, verts, faces, J, parents, weights):
    """Skinned glTF (.glb). Bind pose == canon pose (identity joint rotations, joints at
    regressed positions), so the rendered skin reproduces `verts` exactly and the rig is
    drivable.  verts [V,3], faces [Fc,3], J [35,3], parents [35], weights [V,35]."""
    import pygltflib as g
    V = verts.shape[0]
    nJ = J.shape[0]
    # top-4 skinning weights per vertex
    order = np.argsort(-weights, axis=1)[:, :4]
    w4 = np.take_along_axis(weights, order, axis=1).astype(np.float32)
    w4 = w4 / np.clip(w4.sum(1, keepdims=True), 1e-8, None)
    j4 = order.astype(np.uint8)

    pos = verts.astype(np.float32)
    idx = faces.reshape(-1).astype(np.uint32)
    # inverse bind = translate(-J)  (column-major 4x4)
    ibm = np.tile(np.eye(4, dtype=np.float32), (nJ, 1, 1))
    ibm[:, 3, :3] = -J                                             # row 3 cols 0..2 == column-major translation
    ibm = ibm.reshape(nJ, 16)

    blobs, views, accs = [], [], []

    def add(arr, target=None):
        raw = _pad(arr.tobytes())
        bv = len(views)
        views.append(g.BufferView(buffer=0, byteOffset=sum(len(b) for b in blobs),
                                   byteLength=arr.nbytes, target=target))
        blobs.append(raw)
        return bv

    CT = {np.dtype("float32"): 5126, np.dtype("uint32"): 5125,
          np.dtype("uint8"): 5121, np.dtype("uint16"): 5123}
    TYP = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}

    def acc(arr, ncomp, target=None, mn=None, mx=None):
        bv = add(arr, target)
        a = g.Accessor(bufferView=bv, componentType=CT[arr.dtype], count=arr.shape[0],
                       type=TYP[ncomp], min=mn, max=mx)
        accs.append(a)
        return len(accs) - 1

    a_pos = acc(pos, 3, target=g.ARRAY_BUFFER,
                mn=pos.min(0).tolist(), mx=pos.max(0).tolist())
    a_idx = acc(idx, 1, target=g.ELEMENT_ARRAY_BUFFER)
    a_j = acc(j4, 4, target=g.ARRAY_BUFFER)
    a_w = acc(w4, 4, target=g.ARRAY_BUFFER)
    a_ibm = acc(ibm, 16)

    # nodes: 0..nJ-1 joints, nJ = dog_mesh, nJ+1 = dog_root
    nodes = []
    children = {i: [] for i in range(nJ)}
    for i in range(nJ):
        p = int(parents[i])
        if p >= 0 and p != i:
            children[p].append(i)
    for i in range(nJ):
        p = int(parents[i])
        local = (J[i] - J[p]) if (p >= 0 and p != i) else J[i]
        nodes.append(g.Node(name=JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"joint_{i}",
                            translation=local.astype(float).tolist(),
                            rotation=[0.0, 0.0, 0.0, 1.0],
                            children=children[i] or None))
    nodes.append(g.Node(name="dog_mesh", mesh=0, skin=0))          # node nJ
    nodes.append(g.Node(name="dog_root", translation=[0.0, 0.0, 0.0],
                        children=[0, nJ]))                         # node nJ+1

    mesh = g.Mesh(primitives=[g.Primitive(
        attributes=g.Attributes(POSITION=a_pos, JOINTS_0=a_j, WEIGHTS_0=a_w),
        indices=a_idx)])
    skin = g.Skin(joints=list(range(nJ)), inverseBindMatrices=a_ibm, skeleton=0)

    blob = b"".join(blobs)
    gltf = g.GLTF2(scene=0, scenes=[g.Scene(nodes=[nJ + 1])], nodes=nodes,
                   meshes=[mesh], skins=[skin], accessors=accs, bufferViews=views,
                   buffers=[g.Buffer(byteLength=len(blob))])
    gltf.set_binary_blob(blob)
    gltf.save_binary(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--dog", default="00062-bear")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_root", type=int, default=26000)
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--v6_repr", action="store_true")
    ap.add_argument("--len_short", type=float, default=0.5)
    ap.add_argument("--paw_len", type=float, default=0.3)
    ap.add_argument("--offset_shell", type=float, default=0.2)
    ap.add_argument("--fur_aspect", type=float, default=10.0)
    ap.add_argument("--paw_mask", default="synth_fur/paw_mask.npy")
    ap.add_argument("--out", default="tools/sample_data/bear_example")
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    scenes = [s for s in list_scenes(args.root) if s.split("/")[-2] == args.dog]
    assert scenes, f"dog {args.dog} not found"
    scene = scenes[0]
    ds = FurScenes(scenes, 4, 1, args.n_root)
    da = np.load(os.path.join(scene, "preprocess", "dsmal_anchors.npz"))
    w_face = torch.from_numpy(da["w_face"])
    faces0 = torch.from_numpy(da["faces"]).long()
    subdiv_M = build_subdiv(faces0, 1, dev)
    sub = lambda x: torch.stack([torch.sparse.mm(subdiv_M, x[b]) for b in range(x.shape[0])])
    w_face_s = torch.sparse.mm(subdiv_M, w_face[:, None].float().to(dev))[:, 0].cpu()
    model = DogLRMFurV2(w_face, faces_sub=subdivided_faces(faces0, 1), w_face_s=w_face_s,
                        K=args.K).to(dev)
    load_fur_ckpt(model, args.ckpt, dev)
    model.eval()

    paw_short_v = None
    if args.v6_repr:
        paw = torch.from_numpy(np.load(args.paw_mask)).float().to(dev)
        paw_short_v = 1.0 - (1.0 - args.paw_len) * paw

    frames = ds.frames[0]
    fsp = os.path.join(scene, "preprocess", "face_scores.json")
    fsc = json.load(open(fsp)) if os.path.exists(fsp) else {}
    rid = max(ds.train_ids[0], key=lambda t: fsc.get(frames[t]["name"], 0.0)) if fsc else ds.train_ids[0][0]
    ref = frames[rid]
    rgb_r, mask_r, _, _ = _load_rgb_mask(scene, ref, 8)
    label = _label_grid(scene, ref, mask_r)[None].to(dev)
    inp = F.interpolate(torch.from_numpy(rgb_r).permute(2, 0, 1)[None].to(dev),
                        (518, 518), mode="bilinear", align_corners=False)
    canon = ds.canon[0][None].to(dev)
    anc = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in ds.anc[0].items()}
    rgb4, _, _, _ = _load_rgb_mask(scene, ref, 4)
    anc["ref_rgb"] = torch.from_numpy(rgb4).permute(2, 0, 1)[None].to(dev)
    anc["ref_K"] = intrinsics(ref["fx"] / 4, ref["fy"] / 4, ref["cx"] / 4, ref["cy"] / 4, dev)[None]
    anc["ref_c2w"] = torch.tensor(ref["c2w"], device=dev).float()[None]
    if args.v6_repr:
        anc["len_short"] = args.len_short
        anc["paw_short"] = paw_short_v[None]
        anc["offset_shell"] = args.offset_shell
    if args.fur_aspect > 0:
        anc["fur_aspect"] = args.fur_aspect

    with torch.no_grad():
        fur, body = model(inp, label, canon, anc, sub)
    f0, b0 = fur[0], body[0]
    Kp = args.K
    N = f0["pts"].shape[0]
    cpu = lambda t: t.detach().cpu().numpy()

    # ---- 1. strand polylines -------------------------------------------------
    pts = cpu(f0["pts"])                                           # [N,Kp,3]
    seg_rgb = cpu(f0["rgb"]).reshape(N, Kp - 1, 3)                 # per-segment color
    vrgb = np.concatenate([seg_rgb, seg_rgb[:, -1:]], axis=1)      # [N,Kp,3] vertex color
    vnrm = np.repeat(cpu(f0["nrm"])[:, None], Kp, axis=1)          # root normal per vertex
    nv, ne = write_strands_lines(os.path.join(args.out, "strands_lines.ply"), pts, vrgb, vnrm)

    # ---- 2. strand gaussians (deg-3 SH layout + strand_id) -------------------
    sid = np.repeat(np.arange(N), Kp - 1)
    write_strand_gaussians(os.path.join(args.out, "strand_gaussians.ply"),
                           cpu(f0["means"]), cpu(f0["sh"]), cpu(f0["opacities"]).reshape(-1, 1),
                           cpu(f0["scales"]), cpu(f0["quats"]), sid)

    # ---- 3/4. full scene gaussians (with / without fur) ----------------------
    save_ply(os.path.join(args.out, "gaussians_no_fur.ply"),
             b0["means"], b0["scales"], b0["quats"], b0["opacities"], b0["rgb"])
    full = {k: torch.cat([b0[k], f0[k]]) for k in ("means", "quats", "scales", "opacities", "rgb")}
    save_ply(os.path.join(args.out, "gaussians_with_fur.ply"),
             full["means"], full["scales"], full["quats"], full["opacities"], full["rgb"])

    # ---- 5. body rig glb (SMAL, canon frame) ---------------------------------
    from animate_decomp import load_dsmal
    smal = load_dsmal()
    arr = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    Jreg = torch.from_numpy(arr(smal.J_regressor)).float()         # [V,35]
    parents = arr(smal.parents)                                    # [35]
    weights = arr(smal.weights)                                    # [V,35]
    cverts = da["canon"].astype(np.float32)                        # [3889,3] same frame as gaussians
    cfaces = da["faces"].astype(np.int64)
    J = torch.einsum("vj,vc->jc", Jreg, torch.from_numpy(cverts)).numpy()   # [35,3]
    write_body_rig_glb(os.path.join(args.out, "body_rig.glb"), cverts, cfaces, J, parents, weights)

    print(f"dog={args.dog}  strands N={N} K={Kp}  fur_gauss={f0['means'].shape[0]}  "
          f"body_gauss={b0['means'].shape[0]}", flush=True)
    print(f"  strands_lines.ply : {nv} verts / {ne} edges", flush=True)
    print(f"  strand_gaussians.ply / gaussians_with_fur.ply / gaussians_no_fur.ply / body_rig.glb", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
