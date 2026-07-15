#!/usr/bin/env python3
"""FROM-SCRATCH free-Gaussian fur, gsplat multi-view style + our strand-FLOW structural loss.

Pipeline: COLMAP SfM points -> normalize into the body/flow frame (scene_norm.json) -> visual-hull
filter to the dog -> standard gsplat 3DGS optimization (per-attr Adam + DefaultStrategy densify) with
photometric losses, PLUS the 3D structural loss that keeps the free cloud combed:
  align  : each Gaussian's LONG axis -> combed tangent flow d0 (mod-pi, target from KNN to body roots)
  aniso  : prolate (sliver) not blob        coh : neighbour long-axes agree
  col    : neighbour colour smoothness (kill speckle)
No strand seed, no body layer -- the free cloud IS the whole dog. Records snapshots -> local HTML.
kotori, ALL views. Run: PATH=<env>/bin:$PATH TORCH_EXTENSIONS_DIR=.torch_ext_lhm python train_fur_sfm.py
"""
import argparse, json, os, sys, math, base64, io
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from gsplat import DefaultStrategy, rasterization
from dog_lrm.render import intrinsics
from dog_lrm.smal_model import subdivided_faces
from train_fur_final import nn_argmin, visual_hull_keep
from train_fur_free import quat_to_rotmat, long_axis, knn_idx
from train_fur_v6flow import itp_fn
from train_dog_lrm_ddp import _load_rgb_mask


def read_points3d_txt(path):
    xyz, rgb = [], []
    for ln in open(path):
        if ln.startswith("#") or not ln.strip(): continue
        t = ln.split()
        xyz.append([float(t[1]), float(t[2]), float(t[3])]); rgb.append([float(t[4]), float(t[5]), float(t[6])])
    return np.asarray(xyz, np.float32), np.asarray(rgb, np.float32) / 255.0


def save_sh_ply(path, means, scales, quats, opacities, sh0, shN):
    """Standard 3DGS SH ply (f_dc + f_rest, INRIA layout) so viewers get view-dependent colour like the baseline."""
    from plyfile import PlyData, PlyElement
    n = means.shape[0]
    xyz = means.detach().cpu().numpy().astype(np.float32)
    f_dc = sh0.detach().cpu().numpy().reshape(n, 3).astype(np.float32)
    f_rest = shN.detach().cpu().transpose(1, 2).reshape(n, -1).numpy().astype(np.float32)   # [N,3,K-1]->flat (channel-major)
    op = np.log(opacities.detach().cpu().clamp(1e-4, 1 - 1e-4).numpy() / (1 - opacities.detach().cpu().clamp(1e-4, 1 - 1e-4).numpy())).reshape(n, 1).astype(np.float32)
    scl = np.log(scales.detach().cpu().clamp_min(1e-8).numpy()).astype(np.float32)
    rot = quats.detach().cpu().numpy().astype(np.float32)
    fields = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
              + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([xyz, np.zeros((n, 3), np.float32), f_dc, f_rest, op, scl, rot], axis=1)
    el = np.empty(n, dtype=[(f, "f4") for f in fields])
    for i, f in enumerate(fields): el[f] = data[:, i]
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


def load_flow(scene, dev, tmix):
    """combed tangent flow field on the D-SMAL body (roots + d0), same normalized frame as cameras.json."""
    fa = np.load(os.path.join(scene, "preprocess", "fur_anchors.npz")); tt = lambda k: torch.from_numpy(fa[k]).to(dev).float()
    dfaces = torch.from_numpy(np.load(os.path.join(scene, "preprocess", "dsmal_anchors.npz"))["faces"]).long().to(dev)
    faces_sub = subdivided_faces(dfaces, 1).to(dev)
    rf = torch.from_numpy(fa["root_face"]).long().to(dev); ba = tt("root_bary"); itp = itp_fn(faces_sub[rf], ba)
    roots = itp(tt("roots")); t = F.normalize(itp(tt("t")), dim=-1); n = F.normalize(itp(tt("n")), dim=-1)
    d0 = F.normalize(tmix * t + (1 - tmix) * n, dim=-1)
    return roots.detach(), d0.detach(), float(fa["diag"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", default="00085-kotori")
    ap.add_argument("--root", default="received_data_from_Pinstudio_20260424/unzipped/0423")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--scale_div", type=int, default=4)
    ap.add_argument("--crop", type=int, default=1, help="crop each view to the dog bbox (adjust cx/cy/W/H) -> concentrate resolution on the dog (fixes soft fur)")
    ap.add_argument("--crop_pad", type=float, default=0.12, help="bbox padding fraction")
    ap.add_argument("--sfm_thr", type=float, default=0.55, help="visual-hull: keep sfm point if inside GT mask in >= this fraction of visible views")
    ap.add_argument("--init_tau", type=float, default=0.4, help="also drop init sfm points farther than this*diag from the D-SMAL body (robust bg removal, needed under --crop)")
    ap.add_argument("--init_op", type=float, default=0.1)
    ap.add_argument("--sh_degree", type=int, default=3, help="SH degree for view-dependent colour (0=flat RGB); baseline shba701 uses 3")
    ap.add_argument("--sh_every", type=int, default=1000, help="raise active SH degree by 1 every N iters (progressive, stabler)")
    # densify (DefaultStrategy)
    ap.add_argument("--refine_start", type=int, default=500)
    ap.add_argument("--refine_stop_frac", type=float, default=0.8)
    ap.add_argument("--refine_every", type=int, default=100)
    ap.add_argument("--reset_every", type=int, default=1500)
    ap.add_argument("--grow_grad2d", type=float, default=2e-4)
    ap.add_argument("--prune_opa", type=float, default=0.05)
    # lrs
    ap.add_argument("--lr_pos_frac", type=float, default=1.6e-4, help="means lr = this * scene_scale")
    ap.add_argument("--lr_rot", type=float, default=1e-3)
    ap.add_argument("--lr_scale", type=float, default=5e-3)
    ap.add_argument("--lr_op", type=float, default=5e-2)
    ap.add_argument("--lr_col", type=float, default=2.5e-3)
    # photometric
    ap.add_argument("--lpips_w", type=float, default=0.15)
    ap.add_argument("--w_sil", type=float, default=0.3)
    # structural
    ap.add_argument("--tmix", type=float, default=0.88)
    ap.add_argument("--w_align", type=float, default=1.0)
    ap.add_argument("--w_aniso", type=float, default=0.05)
    ap.add_argument("--w_coh", type=float, default=0.3)
    ap.add_argument("--w_col_coh", type=float, default=0.1)
    ap.add_argument("--struct_k", type=int, default=8)
    ap.add_argument("--dog_tau", type=float, default=0.15, help="structural loss only on gaussians within this*diag of the D-SMAL body (excludes the platform)")
    ap.add_argument("--anneal_frac", type=float, default=0.4)
    ap.add_argument("--anneal_floor", type=float, default=0.2)
    ap.add_argument("--requery", type=int, default=100)
    # viz
    ap.add_argument("--snap_every", type=int, default=100)
    ap.add_argument("--snap_w", type=int, default=380, help="snapshot width in the HTML (downscaled)")
    ap.add_argument("--out", default="exps/fur_sfm")
    args = ap.parse_args()
    dev = "cuda"; s = args.scale_div; os.makedirs(args.out, exist_ok=True); white = torch.ones(3, device=dev)
    scene = os.path.join(args.root, args.dog, "colmap")

    # ---- views (cameras.json + masks), normalized frame ----
    frames = json.load(open(os.path.join(scene, "preprocess", "cameras.json")))["frames"]
    views = []
    for fr in frames:
        rgb, mask, Wd, Hd = _load_rgb_mask(scene, fr, s)
        fx, fy, cx, cy = fr["fx"] / s, fr["fy"] / s, fr["cx"] / s, fr["cy"] / s
        if args.crop:                                          # crop to dog bbox -> dog fills the frame at full res
            m = mask[:, :, 0] > 0.5; ys, xs = np.where(m)
            if len(xs) >= 50:
                pad = int(args.crop_pad * max(int(xs.max() - xs.min()), int(ys.max() - ys.min())))
                x0 = max(int(xs.min()) - pad, 0); y0 = max(int(ys.min()) - pad, 0)
                x1 = min(int(xs.max()) + pad + 1, Wd); y1 = min(int(ys.max()) + pad + 1, Hd)
                rgb = rgb[y0:y1, x0:x1]; mask = mask[y0:y1, x0:x1]; cx -= x0; cy -= y0; Wd = x1 - x0; Hd = y1 - y0
        views.append(dict(rgb=torch.from_numpy(rgb).to(dev), mask=torch.from_numpy(mask).to(dev),
                          K=intrinsics(fx, fy, cx, cy, dev),
                          c2w=torch.tensor(fr["c2w"], device=dev).float(), W=Wd, H=Hd))

    # ---- flow field (roots + combed d0) on the body; robust scene_scale = body diag ----
    flow_pts, flow_dir, diag = load_flow(scene, dev, args.tmix)
    scene_scale = diag                                                   # robust (body diag ~1.3), not point-bbox (outlier-sensitive)

    # ---- SfM points -> normalized frame -> filter to the dog (near-body AND visual-hull) ----
    nrm = json.load(open(os.path.join(scene, "preprocess", "scene_norm.json")))
    center = torch.tensor(nrm["center"], device=dev).float(); sc = float(nrm["scale"])
    xyz, rgb = read_points3d_txt(os.path.join(scene, "sparse", "0", "points3D.txt"))
    pts = (torch.from_numpy(xyz).to(dev) - center) * sc                  # -> body/flow frame
    col = torch.from_numpy(rgb).to(dev)
    near = (pts - flow_pts[nn_argmin(pts, flow_pts)]).norm(dim=-1) < args.init_tau * diag   # drop far background (robust under crop)
    keep = near & visual_hull_keep(pts, views, args.sfm_thr)             # + in-silhouette
    pts, col = pts[keep], col[keep]
    N = pts.shape[0]
    d, _ = cKDTree(pts.detach().cpu().numpy()).query(pts.detach().cpu().numpy(), k=4)        # scale init = mean dist to 3 NN
    s0 = np.clip(d[:, 1:].mean(1), 1e-4, None)
    print(f"[sfm] {args.dog} raw={len(xyz)} -> dog={N} (near+vh) scene_scale={scene_scale:.3f}", flush=True)

    # ---- free-gaussian params (raw / pre-activation); SH deg-N colour like the baseline ----
    C0 = 0.28209479177387814; Ksh = (args.sh_degree + 1) ** 2
    sh0 = ((col - 0.5) / C0)[:, None, :]                                 # [N,1,3] DC term from point colour
    shN = torch.zeros(N, Ksh - 1, 3, device=dev)                        # [N,K-1,3] higher bands start at 0
    params = torch.nn.ParameterDict({
        "means": nn.Parameter(pts.clone()),
        "quats": nn.Parameter(torch.tensor([1., 0., 0., 0.], device=dev).repeat(N, 1)),
        "scales": nn.Parameter(torch.log(torch.from_numpy(s0).to(dev).float())[:, None].repeat(1, 3)),
        "opacities": nn.Parameter(torch.logit(torch.full((N,), args.init_op, device=dev))),
        "sh0": nn.Parameter(sh0.clone()), "shN": nn.Parameter(shN.clone())}).to(dev)
    LRS = dict(means=args.lr_pos_frac * scene_scale, quats=args.lr_rot, scales=args.lr_scale,
               opacities=args.lr_op, sh0=args.lr_col, shN=args.lr_col / 20.0)   # higher bands slower (3DGS convention)
    optimizers = {k: torch.optim.Adam([{"params": [params[k]], "lr": LRS[k], "name": k}], eps=1e-15) for k in params}
    strategy = DefaultStrategy(refine_start_iter=args.refine_start, refine_stop_iter=int(args.refine_stop_frac * args.iters),
                               refine_every=args.refine_every, reset_every=args.reset_every,
                               grow_grad2d=args.grow_grad2d, prune_opa=args.prune_opa, verbose=False)
    strategy.check_sanity(params, optimizers)
    strat_state = strategy.initialize_state(scene_scale=scene_scale)

    # ---- per-gaussian flow target (recomputed on count change / drift) ----
    dog_tau = args.dog_tau * diag
    def refresh():
        """nearest-root flow target, kNN, and dog-mask (near the body -> excludes the platform)."""
        with torch.no_grad():
            m = params["means"].detach(); idx = nn_argmin(m, flow_pts)
            return flow_dir[idx], knn_idx(m, args.struct_k), (m - flow_pts[idx]).norm(dim=-1) < dog_tau
    dflow_tgt, nbr, dogmask = refresh()

    import lpips as Lp
    lpf = Lp.LPIPS(net="alex").to(dev)
    for p in lpf.parameters(): p.requires_grad = False

    def render(v, sh_deg=None):
        m = params["means"]; q = F.normalize(params["quats"], dim=-1); scl = params["scales"].exp()
        op = params["opacities"].sigmoid()
        sh = torch.cat([params["sh0"], params["shN"]], dim=1)            # [N,K,3] SH coeffs -> view-dependent colour
        rc, ra, info = rasterization(means=m, quats=q, scales=scl, opacities=op, colors=sh,
                                     viewmats=torch.inverse(v["c2w"])[None], Ks=v["K"][None],
                                     width=v["W"], height=v["H"], packed=False, absgrad=strategy.absgrad,
                                     render_mode="RGB", sh_degree=args.sh_degree if sh_deg is None else sh_deg)
        rgb = rc[0] + (1 - ra[0]) * white
        return rgb, ra[0], info

    def lp(a, b_):
        a = F.interpolate(a.permute(2, 0, 1)[None] * 2 - 1, 256, mode="bilinear", align_corners=False)
        b_ = F.interpolate(b_.permute(2, 0, 1)[None] * 2 - 1, 256, mode="bilinear", align_corners=False)
        return lpf(a, b_).mean()

    def dmean(x):                                              # mean over dog gaussians only (skip the platform)
        return x[dogmask].mean() if bool(dogmask.any()) else x.sum() * 0.0

    def struct_loss():
        axis = long_axis(params["quats"], params["scales"])
        l_align = dmean(1 - (axis * dflow_tgt).sum(-1).abs())
        ss, _ = params["scales"].exp().sort(dim=1)
        l_aniso = dmean(ss[:, 1] / ss[:, 2].clamp_min(1e-8))
        l_coh = dmean(1 - (axis * F.normalize(axis[nbr].mean(1), dim=-1, eps=1e-8)).sum(-1).abs())
        dc = params["sh0"][:, 0, :]                            # DC colour term for neighbour smoothness
        l_col = ((dc - dc[nbr].mean(1)) ** 2).sum(-1).mean()
        return l_align, l_aniso, l_coh, l_col

    # snapshot view: widest bbox aspect (side profile -> dog-dominated, minimal platform)
    def bbox_aspect(v):
        ys, xs = torch.where(v["mask"][:, :, 0] > 0.5)
        if len(xs) < 50: return 0.0
        return float((xs.max() - xs.min()).item() / max(int((ys.max() - ys.min()).item()), 1))
    snap_v = max(views, key=bbox_aspect)
    snaps, metrics = [], dict(it=[], loss=[], psnr=[], coh=[], count=[])

    def snapshot(it):
        with torch.no_grad():
            rgb, _, _ = render(snap_v); r = rgb.clamp(0, 1); mk = snap_v["mask"]
            gtw = snap_v["rgb"] * mk + (1 - mk) * white
            psnr = float(-10 * math.log10((((r - gtw) ** 2 * mk).sum() / (mk.sum() * 3).clamp_min(1)).clamp_min(1e-10)))  # masked
            axis = long_axis(params["quats"], params["scales"])
            cc = (axis * F.normalize(axis[nbr].mean(1), dim=-1, eps=1e-8)).sum(-1).abs()
            coh = float(cc[dogmask].mean() if bool(dogmask.any()) else cc.mean())
            im = Image.fromarray((r.cpu().numpy() * 255).astype(np.uint8))
            gt = Image.fromarray(((snap_v["rgb"] * mk + (1 - mk) * white).cpu().numpy() * 255).astype(np.uint8))
            Image.fromarray(np.concatenate([np.asarray(im), np.asarray(gt)], 1)).save(   # peek mid-run
                os.path.join(args.out, "_progress.jpg"), quality=85)
            w = args.snap_w; im = im.resize((w, int(im.height * w / im.width)))
            buf = io.BytesIO(); im.save(buf, format="JPEG", quality=82)
            snaps.append(base64.b64encode(buf.getvalue()).decode())
        metrics["it"].append(it); metrics["psnr"].append(round(psnr, 2)); metrics["coh"].append(round(coh, 4))
        metrics["count"].append(int(params["means"].shape[0]))

    for it in range(1, args.iters + 1):
        v = views[np.random.randint(len(views))]
        cur_deg = min(args.sh_degree, it // args.sh_every)      # progressive SH: learn DC first, add bands gradually
        rgb, alpha, info = render(v, sh_deg=cur_deg)
        strategy.step_pre_backward(params, optimizers, strat_state, it, info)
        gtw = v["rgb"] * v["mask"] + (1 - v["mask"]) * white
        la, lan, lc, lcol = struct_loss()
        frac = min(it / max(args.anneal_frac * args.iters, 1), 1.0); aw = 1.0 - (1.0 - args.anneal_floor) * frac
        loss = (F.l1_loss(rgb * v["mask"], v["rgb"] * v["mask"]) + F.l1_loss(alpha, v["mask"])
                + args.lpips_w * lp(rgb, gtw) + args.w_sil * F.relu(alpha - v["mask"]).mean()
                + args.w_col_coh * lcol + aw * (args.w_align * la + args.w_aniso * lan + args.w_coh * lc))
        metrics_loss = float(loss)
        loss.backward()
        for opt in optimizers.values(): opt.step(); opt.zero_grad(set_to_none=True)
        M0 = params["means"].shape[0]
        strategy.step_post_backward(params, optimizers, strat_state, it, info, packed=False)
        if params["means"].shape[0] != M0 or it % args.requery == 0:      # count changed (densify) or drift -> refresh
            dflow_tgt, nbr, dogmask = refresh()
        if it % args.snap_every == 0 or it == 1:
            snapshot(it); metrics["loss"].append(round(metrics_loss, 4))
            print(f"it{it:4d} loss={metrics_loss:.4f} psnr={metrics['psnr'][-1]:.2f} coh={metrics['coh'][-1]:.4f} "
                  f"align={float(la):.3f} N={metrics['count'][-1]}", flush=True)

    # ---- final metrics over ALL views ----
    with torch.no_grad():
        l1s, pss, lps = [], [], []
        for v in views:
            rgb, _, _ = render(v); mk = v["mask"]; gtw = v["rgb"] * mk + (1 - mk) * white; r = rgb.clamp(0, 1)
            den = (mk.sum() * 3).clamp_min(1)
            l1s.append(float(((r - gtw).abs() * mk).sum() / den))                                  # masked L1
            pss.append(float(-10 * math.log10((((r - gtw) ** 2 * mk).sum() / den).clamp_min(1e-10))))  # masked PSNR
            lps.append(float(lp(r, gtw)))
    print(f"[sfm] {args.dog} train-view L1={np.mean(l1s):.4f} PSNR={np.mean(pss):.2f} LPIPS={np.mean(lps):.4f} "
          f"| N={params['means'].shape[0]} coh={metrics['coh'][-1]:.4f}", flush=True)

    # ---- save ply + free.pt ----
    with torch.no_grad():
        save_sh_ply(os.path.join(args.out, f"{args.dog}.ply"), params["means"], params["scales"].exp(),
                    F.normalize(params["quats"], dim=-1), params["opacities"].sigmoid(), params["sh0"], params["shN"])
        torch.save({k: params[k].detach() for k in params} | {"scene_scale": scene_scale, "diag": diag},
                   os.path.join(args.out, f"{args.dog}_sfm.pt"))
        # final composite vs GT strip
        rgb, _, _ = render(snap_v); r = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        gt = ((snap_v["rgb"] * snap_v["mask"] + (1 - snap_v["mask"]) * white).cpu().numpy() * 255).astype(np.uint8)
        buf = io.BytesIO(); Image.fromarray(np.concatenate([r, gt], 1)).save(buf, format="JPEG", quality=88)
        final_b64 = base64.b64encode(buf.getvalue()).decode()

    # ---- self-contained local HTML (offline) ----
    html_path = os.path.join(args.out, f"{args.dog}_training.html")
    build_html(html_path, args.dog, snaps, final_b64, metrics, dict(
        L1=round(float(np.mean(l1s)), 4), PSNR=round(float(np.mean(pss)), 2), LPIPS=round(float(np.mean(lps)), 4),
        N=int(params["means"].shape[0]), coh=metrics["coh"][-1]))
    print(f"[sfm] saved ply/sfm.pt + training viz -> {html_path}", flush=True)


def build_html(path, dog, snaps, final_b64, metrics, summary):
    data = json.dumps(dict(snaps=snaps, final=final_b64, m=metrics, s=summary))
    html = """<!doctype html><html><head><meta charset="utf-8"><title>fur-sfm __DOG__</title>
<style>
 body{margin:0;background:#0e0f13;color:#e6e6e6;font:14px/1.5 -apple-system,system-ui,sans-serif}
 .wrap{max-width:1080px;margin:0 auto;padding:22px}
 h1{font-size:18px;font-weight:600;margin:0 0 4px} .sub{color:#8b90a0;margin-bottom:18px}
 .card{background:#171922;border:1px solid #232633;border-radius:12px;padding:16px;margin-bottom:16px}
 .row{display:flex;gap:16px;flex-wrap:wrap} .col{flex:1;min-width:320px}
 img{max-width:100%;border-radius:8px;display:block} canvas{width:100%;height:120px;background:#0e0f13;border-radius:8px}
 .ctrl{display:flex;align-items:center;gap:12px;margin-top:10px}
 input[type=range]{flex:1} button{background:#2b64ff;color:#fff;border:0;border-radius:8px;padding:7px 14px;cursor:pointer;font-weight:600}
 .kv{display:flex;gap:18px;flex-wrap:wrap;color:#c4c8d4} .kv b{color:#fff} .lab{font-size:12px;color:#8b90a0;margin:2px 0 6px}
 .it{font-variant-numeric:tabular-nums;color:#7fe0a0}
</style></head><body><div class="wrap">
 <h1>Free-Gaussian fur from SfM &mdash; __DOG__</h1>
 <div class="sub">from-scratch COLMAP points &rarr; gsplat densify + 3D strand-flow structural loss (align/aniso/coh/colour)</div>
 <div class="card"><div class="kv" id="sum"></div></div>
 <div class="row">
  <div class="col card"><div class="lab">training progression &nbsp; iter <span class="it" id="itl"></span></div>
   <img id="frame"><div class="ctrl"><button id="play">&#9658; play</button>
   <input type="range" id="sl" min="0" value="0"></div></div>
  <div class="col card"><div class="lab">metrics</div>
   <div class="lab">PSNR</div><canvas id="cpsnr"></canvas>
   <div class="lab">flow-coherence</div><canvas id="ccoh"></canvas>
   <div class="lab"># gaussians</div><canvas id="ccnt"></canvas></div>
 </div>
 <div class="card"><div class="lab">final render (left) vs GT (right)</div><img src="data:image/jpeg;base64,__FINAL__" id="fin"></div>
</div><script>
const D=__DATA__;
const sum=document.getElementById('sum');
sum.innerHTML=Object.entries(D.s).map(([k,v])=>`<span>${k} <b>${v}</b></span>`).join('');
const fr=document.getElementById('frame'),sl=document.getElementById('sl'),itl=document.getElementById('itl');
sl.max=D.snaps.length-1;
function show(i){fr.src='data:image/jpeg;base64,'+D.snaps[i];itl.textContent=D.m.it[i];sl.value=i;draw(i);}
sl.oninput=e=>show(+e.target.value);
let playing=false,timer=null;const btn=document.getElementById('play');
btn.onclick=()=>{playing=!playing;btn.innerHTML=playing?'&#10073;&#10073; pause':'&#9658; play';
 if(playing)timer=setInterval(()=>{let i=(+sl.value+1)%D.snaps.length;show(i);},220);else clearInterval(timer);};
function curve(id,arr,color){const c=document.getElementById(id),ctx=c.getContext('2d');
 const W=c.width=c.clientWidth*2,H=c.height=240;ctx.clearRect(0,0,W,H);
 const mn=Math.min(...arr),mx=Math.max(...arr),pad=18;
 ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();
 arr.forEach((v,i)=>{const x=pad+i/(arr.length-1)*(W-2*pad),y=H-pad-(v-mn)/((mx-mn)||1)*(H-2*pad);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
 ctx.stroke();ctx.fillStyle='#8b90a0';ctx.font='20px sans-serif';
 ctx.fillText(mx.toFixed(id=='ccnt'?0:2),4,20);ctx.fillText(mn.toFixed(id=='ccnt'?0:2),4,H-4);
 return {mn,mx,pad,W,H};}
let G={};
function draw(i){G.cpsnr=curve('cpsnr',D.m.psnr,'#2b64ff');G.ccoh=curve('ccoh',D.m.coh,'#7fe0a0');G.ccnt=curve('ccnt',D.m.count,'#ff9f43');
 [['cpsnr',D.m.psnr],['ccoh',D.m.coh],['ccnt',D.m.count]].forEach(([id,arr])=>{const c=document.getElementById(id),ctx=c.getContext('2d'),g=G[id];
  const x=g.pad+i/(arr.length-1)*(g.W-2*g.pad);ctx.strokeStyle='#ffffff55';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,g.H);ctx.stroke();});}
show(0);window.onresize=()=>draw(+sl.value);
</script></body></html>"""
    html = html.replace("__DOG__", dog).replace("__FINAL__", final_b64).replace("__DATA__", data)
    open(path, "w").write(html)


if __name__ == "__main__":
    main()
