"""P3 prep: per-dog fur strand anchors at subdiv-1 level -> <scene>/preprocess/fur_anchors.npz.

Mirrors the geometry block of train_fur_v2.py: anatomical tangent field on the dog's
canonical D-SMAL mesh, TBN-transported to the posed fit; roots on the inset surface
(inset = min(prior length, 0.45 x ray-cast thickness)); per-vertex prior length;
gravity; curl class id + stiffness from the P2 priors.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dsmal_region_masks import DSMAL_ROOT, SCENE_ROOT, load_smal, scene_verts
from fur_strand_init import smooth_tangent_field, vert_frames, LEG_JOINTS
from defur_mask import length_field_cm, local_thickness

CURL_CLASSES = ["short_smooth", "double_coat", "long_straight", "wavy", "curly", "wire"]
PRIORS = "/home/yyang/mnt/workspace/exps/vlm_priors"


def subdivided_faces(faces, n):
    f = faces.long().cpu()
    V = int(f.max()) + 1
    for _ in range(n):
        e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
        e = torch.unique(torch.sort(e, dim=1).values, dim=0)
        eidx = {tuple(t.tolist()): V + k for k, t in enumerate(e)}
        mid = lambda a, b: eidx[tuple(sorted((int(a), int(b))))]
        nf = []
        for tri in f:
            a, b, c = tri.tolist()
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        f = torch.tensor(nf)
        V = V + len(e)
    return f


def carve_tail(roots, normals, w_tail, tail_J, tail_r, t_max, n_steps=24):
    """Truncate non-tail vertices' envelopes where the normal ray enters the tail
    capsule (tail bone chain, radius = tail fur length + bone). The tail's fur
    volume must be owned by tail strands or back strands grow into it and the
    'bump' rides the back when the tail animates away."""
    ts = torch.linspace(0, t_max, n_steps)
    pts = roots[None] + normals[None] * ts.view(-1, 1, 1)           # [T,V,3]
    a, b = tail_J[:-1], tail_J[1:]                                  # segments [S,3]
    ab = b - a
    ab2 = (ab * ab).sum(-1).clamp_min(1e-9)
    p = pts[:, :, None]                                             # [T,V,1,3]
    u = (((p - a) * ab).sum(-1) / ab2).clamp(0, 1)                  # [T,V,S]
    d = (p - (a + u[..., None] * ab)).norm(dim=-1).min(-1).values   # [T,V]
    free = torch.cumprod((d > tail_r).float(), dim=0).sum(0)        # steps before entry
    L_carve = (free.clamp(min=1) - 1) / (n_steps - 1) * t_max
    L_carve[w_tail > 0.5] = t_max                                   # tail keeps its own
    return L_carve


def measure_L_geo(scene, roots, normals, diag, n_views=12, n_steps=24, s=8):
    """Pose-invariant max fur length per vertex: march from the bald root along the
    normal and find where the GT mask ends; min over views (occlusion only inflates,
    so the min is a sound upper bound). v1's w_len_geo, attached to the surface."""
    from PIL import Image
    cams = json.load(open(os.path.join(scene, "preprocess/cameras.json")))["frames"]
    cams = sorted(cams, key=lambda f: f["name"])
    cams = cams[:: max(len(cams) // n_views, 1)][:n_views]
    t_max = 0.14 * diag
    ts = torch.linspace(0, t_max, n_steps)
    pts = roots[None] + normals[None] * (ts.view(-1, 1, 1))          # [T,V,3]
    L_geo = torch.full((roots.shape[0],), t_max)
    for fr in cams:
        stem = os.path.splitext(fr["name"])[0]
        mp = os.path.join(scene, "preprocess", f"cache_s{s}", stem + ".png")
        if not os.path.exists(mp):
            continue
        m = torch.from_numpy(np.asarray(Image.open(mp).convert("L"), np.float32) / 255.)
        H, W = m.shape
        w2c = torch.tensor(np.linalg.inv(np.array(fr["c2w"])), dtype=torch.float32)
        cam_p = pts @ w2c[:3, :3].T + w2c[:3, 3]
        u = (cam_p[..., 0] / cam_p[..., 2].clamp_min(1e-6) * fr["fx"] / s + fr["cx"] / s)
        v = (cam_p[..., 1] / cam_p[..., 2].clamp_min(1e-6) * fr["fy"] / s + fr["cy"] / s)
        inside = ((u >= 0) & (u < W) & (v >= 0) & (v < H))
        ui = u.clamp(0, W - 1).long()
        vi = v.clamp(0, H - 1).long()
        in_mask = (m[vi, ui] > 0.5) & inside                          # [T,V]
        # longest contiguous in-mask run from t=0
        run = torch.cumprod(in_mask.float(), dim=0).sum(0)            # [V] steps
        L_view = (run.clamp(min=1) - 1) / (n_steps - 1) * t_max
        vis = in_mask[0]                                              # root visible in mask
        L_geo = torch.where(vis, torch.minimum(L_geo, L_view), L_geo)
    return L_geo


def view_visibility_stats(scene, posed_s, faces_sub, w_muzzle_s, diag, s=8, device="cuda"):
    """Z-buffered per-vertex visibility over all train views.
    vert_conf [V] = fraction of visible views whose pixel lies inside the GT mask
    (never-visible verts -> 1.0: they cannot hurt the silhouette);
    face_scores {view_name: visible fraction of muzzle verts} -- the muzzle is only
    z-visible when the face points toward the camera, so this discriminates
    front/side views from the back of the head (raw w_face area cannot)."""
    from PIL import Image
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
    from pytorch3d.utils import cameras_from_opencv_projection

    frames = json.load(open(os.path.join(scene, "preprocess/cameras.json")))["frames"]
    frames = sorted(frames, key=lambda f: f["name"])
    verts = posed_s.to(device)
    mesh = Meshes(verts=[verts], faces=[faces_sub.to(device)])
    vis_n = torch.zeros(verts.shape[0], device=device)
    in_n = torch.zeros(verts.shape[0], device=device)
    muzzle = (w_muzzle_s > 0.5).to(device)
    eps = 0.005 * diag
    face_scores = {}
    for fr in frames:
        stem = os.path.splitext(fr["name"])[0]
        mp = os.path.join(scene, "preprocess", f"cache_s{s}", stem + ".png")
        if not os.path.exists(mp):
            continue
        m = torch.from_numpy(np.asarray(Image.open(mp).convert("L"), np.float32) / 255.).to(device)
        H, W = m.shape
        K = torch.tensor([[fr["fx"] / s, 0, fr["cx"] / s], [0, fr["fy"] / s, fr["cy"] / s],
                          [0, 0, 1]], dtype=torch.float32, device=device)[None]
        w2c = torch.tensor(np.linalg.inv(np.array(fr["c2w"])), dtype=torch.float32, device=device)
        cam = cameras_from_opencv_projection(w2c[None, :3, :3], w2c[None, :3, 3], K,
                                             torch.tensor([[H, W]], device=device))
        # bin_size=0: naive rasterization -- the coarse binning overflows at this
        # face count and silently drops faces (incomplete zbuf -> wrong visibility)
        frag = MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(
            image_size=(H, W), faces_per_pixel=1, bin_size=0)).forward(mesh)
        zbuf = frag.zbuf[0, :, :, 0]
        cp = verts @ w2c[:3, :3].T + w2c[:3, 3]
        z = cp[:, 2].clamp_min(1e-6)
        u = (cp[:, 0] / z * fr["fx"] / s + fr["cx"] / s).round().long()
        v = (cp[:, 1] / z * fr["fy"] / s + fr["cy"] / s).round().long()
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (cp[:, 2] > 0)
        ui, vi = u.clamp(0, W - 1), v.clamp(0, H - 1)
        vis = inb & (zbuf[vi, ui] > 0) & (cp[:, 2] <= zbuf[vi, ui] + eps)
        vis_n += vis.float()
        in_n += (vis & (m[vi, ui] > 0.5)).float()
        face_scores[fr["name"]] = float(vis[muzzle].float().mean()) if bool(muzzle.any()) else 0.0
    conf = torch.where(vis_n > 0, in_n / vis_n.clamp_min(1), torch.ones_like(vis_n))
    return conf.cpu(), face_scores


def sample_roots(posed_s, faces_sub, w_face_s, vert_conf, n=40000, tau=0.5, seed=0):
    """Area x (1 + mean w_face) weighted barycentric strand roots (~2x density on the
    face); roots with interpolated conf < tau rejected (fit-error regions); output is
    SHUFFLED so any prefix is a uniform subsample (--n_root tunable without re-cache).
    Also draws the fixed per-root randomness (curl phase, tone) that used to live in
    the model's per-(vertex,S) buffers."""
    g = torch.Generator().manual_seed(seed)
    tri = posed_s[faces_sub]
    area = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    w = area * (1 + w_face_s[faces_sub].mean(1))
    m_ = n * 4
    fi = torch.multinomial(w, m_, replacement=True, generator=g)
    r = torch.rand(m_, 2, generator=g)
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    ba = torch.stack([1 - r[:, 0] - r[:, 1], r[:, 0], r[:, 1]], 1)
    conf = (vert_conf[faces_sub[fi]] * ba).sum(1)
    keep = (conf >= tau).nonzero()[:, 0]
    if keep.shape[0] < n:                       # degenerate fit: fill with best rejected
        rej = (conf < tau).nonzero()[:, 0]
        extra = rej[conf[rej].argsort(descending=True)[: n - keep.shape[0]]]
        keep = torch.cat([keep, extra])
    keep = keep[:n][torch.randperm(min(keep.shape[0], n), generator=g)]
    fi, ba = fi[keep], ba[keep]
    phase = torch.rand(n, generator=g) * 2 * np.pi
    tone = 1.0 + (torch.rand(n, generator=g) - 0.5) * 0.36
    return fi, ba, conf[keep], phase, tone


def main():
    from dog_lrm.smal_model import build_subdiv
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--n_root", type=int, default=40000)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out_name", default="fur_anchors.npz",
                    help="output npz name (v7 uses fur_anchors_v7.npz to keep v6 intact)")
    args = ap.parse_args()
    smal = load_smal()
    os.chdir("/home/yyang/mnt/workspace")
    faces = torch.tensor(np.asarray(smal.faces)).long()
    Wsk = torch.tensor(np.asarray(smal.weights), dtype=torch.float32)
    w_leg = Wsk[:, LEG_JOINTS].sum(1).clamp(0, 1)
    flow = ((1 - w_leg).view(-1, 1) * torch.tensor([-1.0, 0, 0])
            + w_leg.view(-1, 1) * torch.tensor([0, 0, -1.0]))
    sub_M = build_subdiv(faces, 1, "cpu")
    sub = lambda x: torch.sparse.mm(sub_M, x)
    faces_sub = subdivided_faces(faces, 1)
    nbr = torch.zeros(int(faces_sub.max()) + 1, dtype=torch.long)
    nbr[faces_sub[:, 0]] = faces_sub[:, 1]
    nbr[faces_sub[:, 1]] = faces_sub[:, 2]
    nbr[faces_sub[:, 2]] = faces_sub[:, 0]
    eye = torch.eye(3)[None, None].repeat(1, 35, 1, 1)
    paws_m = Wsk[:, [10, 14, 20, 24]].sum(1) > 0.4

    dogs = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(DSMAL_ROOT, "params")))
    for dog in dogs:
        if args.only and args.only not in dog:
            continue
        scene = os.path.join(SCENE_ROOT, dog, "colmap")
        pj = os.path.join(PRIORS, f"{dog}.json")
        if not (os.path.exists(os.path.join(scene, "preprocess/cameras.json")) and os.path.exists(pj)):
            print(f"[skip] {dog}")
            continue
        prior = json.load(open(pj))
        d = np.load(os.path.join(DSMAL_ROOT, "params", f"{dog}.npz"))
        t_ = lambda k: torch.tensor(d["offset_" + k])
        canon = smal(beta=t_("betas"), betas_limbs=t_("betas_limbs"), pose=eye,
                     trans=torch.zeros(1, 3), vert_off_compact=t_("vert_off_compact"),
                     get_skin=True)[0][0].float()
        tan_c, _ = smooth_tangent_field(canon, faces, flow)
        posed, _, w_face = scene_verts(smal, dog, "offset")
        canon_s, posed_s = sub(canon), sub(posed)
        F0 = vert_frames(canon_s, faces_sub, nbr)
        Ft = vert_frames(posed_s, faces_sub, nbr)
        tan_p = F.normalize(torch.einsum("vij,vj->vi", Ft,
                            torch.einsum("vij,vj->vi", F0.transpose(1, 2),
                                         F.normalize(sub(tan_c), dim=-1))), dim=-1)
        n_p = Ft[:, :, 2]
        b_p = torch.cross(n_p, tan_p, dim=-1)
        Lcm_s = sub(torch.tensor(length_field_cm(smal, prior)).view(-1, 1))[:, 0]
        diag = float((posed.max(0).values - posed.min(0).values).norm())
        L = Lcm_s * (diag / prior["dog_bbox_diag_cm"])
        th = torch.tensor(local_thickness(posed_s, faces_sub, n_p.numpy()), dtype=torch.float32)
        inset = torch.minimum(L, 0.45 * th)
        # roots ON the fitted surface (user, 2026-06-12): the per-vertex inset mesh is
        # structurally broken (thin parts crushed/lumpy, see _diag_inset_vs_surface);
        # the fit is to the furry silhouette anyway and tangent strands hug it.
        roots = posed_s
        d0 = F.normalize(0.75 * tan_p + 0.25 * n_p, dim=-1)          # envelope along growth dir
        gravity = F.normalize(posed[paws_m].mean(0) - posed.mean(0), dim=0)
        w_face_s = sub(w_face.view(-1, 1).float())[:, 0].clamp(0, 1)
        w_tail_s = sub(Wsk[:, list(range(25, 32))].sum(1, keepdim=True))[:, 0].clamp(0, 1)
        w_ear_s = sub(Wsk[:, [33, 34]].sum(1, keepdim=True))[:, 0].clamp(0, 1)
        L_geo = measure_L_geo(scene, roots, d0, diag)
        # carve the tail's territory out of other parts' envelopes
        sn = json.load(open(os.path.join(scene, "preprocess/scene_norm.json")))
        _, J, _ = smal(beta=t_("betas"), betas_limbs=t_("betas_limbs"), pose=t_("pose"),
                       trans=t_("trans"), vert_off_compact=t_("vert_off_compact"),
                       get_skin=True, uniform_scale=torch.exp(t_("log_scale")))
        tail_J = (J[0, 25:32].detach().cpu() - torch.tensor(sn["center"]).float()) * sn["scale"]
        tail_r = float(L[w_tail_s > 0.5].median()) * 0.8 + 0.01 * diag if (w_tail_s > 0.5).any() else 0.05 * diag
        L_geo = torch.minimum(L_geo, carve_tail(roots, d0, w_tail_s, tail_J, tail_r, 0.14 * diag))
        # r5: fit-error confidence + budgeted root sampling + face-visibility scores
        w_muzzle_s = sub(Wsk[:, 32:33])[:, 0].clamp(0, 1)
        # v7 face/nose mask: head (16) + jaw (32) -> short fur there (avoid face beard)
        w_head_s = sub(Wsk[:, [16, 32]].sum(1, keepdim=True))[:, 0].clamp(0, 1)
        vert_conf, face_scores = view_visibility_stats(scene, posed_s, faces_sub, w_muzzle_s, diag)
        root_face, root_bary, root_conf, root_phase, root_tone = sample_roots(
            posed_s, faces_sub, w_face_s, vert_conf, n=args.n_root, tau=args.tau)
        json.dump(face_scores, open(os.path.join(scene, "preprocess/face_scores.json"), "w"))
        np.savez(os.path.join(scene, "preprocess", args.out_name),
                 w_ear=w_ear_s.numpy().astype(np.float32),
                 w_head=w_head_s.numpy().astype(np.float32),
                 vert_conf=vert_conf.numpy().astype(np.float32),
                 root_face=root_face.numpy().astype(np.int64),
                 root_bary=root_bary.numpy().astype(np.float32),
                 root_conf=root_conf.numpy().astype(np.float32),
                 root_phase=root_phase.numpy().astype(np.float32),
                 root_tone=root_tone.numpy().astype(np.float32),
                 w_tail=w_tail_s.numpy().astype(np.float32),
                 L_geo=L_geo.numpy().astype(np.float32),
                 roots=roots.numpy().astype(np.float32), t=tan_p.numpy().astype(np.float32),
                 b=b_p.numpy().astype(np.float32), n=n_p.numpy().astype(np.float32),
                 L=L.numpy().astype(np.float32), inset=inset.numpy().astype(np.float32),
                 w_face=w_face_s.numpy().astype(np.float32),
                 gravity=gravity.numpy().astype(np.float32), diag=np.float32(diag),
                 curl_id=np.int64(CURL_CLASSES.index(prior["curl_class"])),
                 stiffness=np.float32(prior["stiffness"]))
        fs = np.array(list(face_scores.values()))
        print(f"[ok] {dog} curl={prior['curl_class']} | conf<0.5 {float((vert_conf < 0.5).float().mean()):.1%} "
              f"<0.8 {float((vert_conf < 0.8).float().mean()):.1%} | "
              f"face_score max {fs.max():.2f} >=0.3max {int((fs >= 0.3 * fs.max()).sum())}/{len(fs)} views",
              flush=True)


if __name__ == "__main__":
    main()
