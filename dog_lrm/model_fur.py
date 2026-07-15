"""Feedforward strand-fur head on the decomposed Dog-LRM backbone (FUR_V2_PLAN P3, r5).

r5: strands grow from N budgeted barycentric roots (cached in fur_anchors.npz,
area x (1+w_face) weighted, fit-error-confidence pruned) instead of every subdiv
vertex x S; per-root learnable opacity gate / radius / SH deg-1 view-dependent
color (stored in the local TBN frame, rotated to world per pose -> animation-safe).
Body layer: opaque on the face, invisible elsewhere, densified with face edge
midpoints, ref-projection sampled RGB + learnable extinguish gate. All per-root
quantities accept additive residual overrides from `anc` (per-dog refinement and
animation share this code path).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dog_lrm.model_decomp import DogLRMDecomp
from dog_lrm.render import sh0_from_rgb, sh1_from_vec
from train_fur_strand import quat_align_z

N_CURL = 6


def face_edge_midpoints(faces_sub, w_face_s, thr=0.2):
    """Unique subdiv-1 edges with both endpoints on the face -> [Ne,2] vertex pairs.
    Their midpoints densify the body layer one extra level on the face; every
    per-vertex field interpolates as the endpoint mean. Template-level (shared)."""
    e = torch.cat([faces_sub[:, [0, 1]], faces_sub[:, [1, 2]], faces_sub[:, [2, 0]]], 0)
    e = torch.unique(torch.sort(e, dim=1).values, dim=0)
    keep = (w_face_s[e[:, 0]] > thr) & (w_face_s[e[:, 1]] > thr)
    return e[keep]


class DogLRMFurV2(DogLRMDecomp):
    def __init__(self, w_face, faces_sub=None, w_face_s=None, K=11, radius_frac=0.0032,
                 tangent_mix=0.75, droop=0.6, fur_op=0.7, **kw):
        super().__init__(w_face, **kw)
        # parent's per-vertex gaussian heads are unused here; drop them so DDP
        # (find_unused_parameters=False) sees every parameter in the graph
        for h in ("head_offset", "head_scale", "head_quat", "head_opacity", "head_rgb"):
            delattr(self, h)
        self.Kp = K
        self.radius_frac = radius_frac
        self.tangent_mix, self.droop0 = tangent_mix, droop
        self.op_bias = float(np.log((fur_op / 0.9) / (1 - fur_op / 0.9)))  # 0.9*sig = fur_op
        D = self.img_proj.out_features
        # 21 = dir_delta(3)+len(1)+albedo(3)+blend(1)+op_logit(1)+radius_logit(1)+sh1_tbn(9)
        #      +droop_gamma(1)+droop_logit(1)
        self.fur_head = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 21))
        self.droop_bias = float(np.log(droop / (1 - droop)))         # logit so zero-row -> droop0
        # 14 = rgb(3)+blend(1)+gate(1)+sh1_tbn(9)
        self.body_rgb_head = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 14))
        for head in (self.fur_head, self.body_rgb_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        # v6 coat code (FUR_V6_PLAN s8.3): curl_emb is the continuous coat embedding
        # (indexed by VLM curl_id); curl_mlp decodes it into curl_amp/freq (r5) PLUS
        # offset_shell residual + len_scale (the two new outputs are zero-init -> neutral
        # -> warm-start safe). coat_head classifies the embedding into the 6 VLM coat
        # classes (CE-supervised) so the embedding becomes a meaningful coat representation.
        self.curl_emb = nn.Embedding(N_CURL, 8)
        self.curl_mlp = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))
        nn.init.zeros_(self.curl_mlp[-1].weight)
        nn.init.zeros_(self.curl_mlp[-1].bias)
        self.coat_head = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, N_CURL))
        if faces_sub is not None:
            self.register_buffer("faces_sub", faces_sub.long())
            self.register_buffer("face_edges", face_edge_midpoints(faces_sub.long(), w_face_s))

    def _augment_root_feat(self, xr, anc, idx, ba):
        """Hook: add extra per-root features before the fur head. Identity in v2;
        v7 adds the triplane feature-field sample at the root's canonical position."""
        return xr

    @staticmethod
    def _interp(f, idx, ba):
        """Barycentric gather: f [B,Vs] or [B,Vs,C], idx [B,N,3], ba [B,N,3] -> per-root."""
        squeeze = f.dim() == 2
        if squeeze:
            f = f[..., None]
        bb = torch.arange(f.shape[0], device=f.device).view(-1, 1, 1)
        out = (f[bb, idx] * ba[..., None]).sum(2)
        return out[..., 0] if squeeze else out

    @staticmethod
    def _mids(f, e):
        """Append face-edge midpoints: f [B,Vs] or [B,Vs,C] -> [B,Vs+Ne(,C)]."""
        squeeze = f.dim() == 2
        if squeeze:
            f = f[..., None]
        out = torch.cat([f, 0.5 * (f[:, e[:, 0]] + f[:, e[:, 1]])], 1)
        return out[..., 0] if squeeze else out

    @staticmethod
    def _tbn(t_raw, n_raw):
        n = F.normalize(n_raw, dim=-1)
        t = F.normalize(t_raw - (t_raw * n).sum(-1, keepdim=True) * n, dim=-1)
        return t, torch.cross(n, t, dim=-1), n

    @staticmethod
    def _sh1_world(a_local, t, b, n):
        """Local-TBN linear color vec [B,M,3(axis),3(rgb)] -> world deg-1 SH coeffs.
        The deg-1 band is exactly a vector under rotation, so re-expressing in the
        current pose's TBN transports view-dependence through animation."""
        a_w = (t[..., :, None] * a_local[..., 0:1, :]
               + b[..., :, None] * a_local[..., 1:2, :]
               + n[..., :, None] * a_local[..., 2:3, :])
        return sh1_from_vec(a_w)

    def sample_ref(self, pts, n, anc):
        """Project (fit-pose) points into the ref view and sample its RGB.
        pts [B,M,3], n [B,M,3] -> samp [B,M,3], vis [B,M,1]."""
        samp, vis = [], []
        for bi in range(pts.shape[0]):
            c2w = anc["ref_c2w"][bi]
            K = anc["ref_K"][bi]
            img = anc["ref_rgb"][bi]                                 # [3,H,W]
            H, W = img.shape[1:]
            w2c = torch.linalg.inv(c2w)
            r = pts[bi]
            cam = r @ w2c[:3, :3].T + w2c[:3, 3]
            z = cam[:, 2].clamp_min(1e-6)
            u = cam[:, 0] / z * K[0, 0] + K[0, 2]
            v = cam[:, 1] / z * K[1, 1] + K[1, 2]
            gx = (u / (W - 1)) * 2 - 1
            gy = (v / (H - 1)) * 2 - 1
            g = torch.stack([gx, gy], -1).view(1, -1, 1, 2)
            s = F.grid_sample(img[None], g, mode="bilinear", align_corners=True,
                              padding_mode="border")[0, :, :, 0].T   # [M,3]
            front = (F.normalize(c2w[:3, 3] - r, dim=-1) * n[bi]).sum(-1) > 0.1
            inside = (gx.abs() < 1) & (gy.abs() < 1) & (cam[:, 2] > 0)
            samp.append(s)
            vis.append((front & inside).float()[:, None])
        return torch.stack(samp), torch.stack(vis)

    def fur_gaussians(self, xs, anc):
        """xs [B,Vs,D] subdivided features; anc per-dog buffers incl. root_face/_bary
        [B,N(,3)]. Returns per-item list of gaussian dicts (equal counts)."""
        B, Kp = xs.shape[0], self.Kp
        idx = self.faces_sub[anc["root_face"]]                       # [B,N,3]
        ba = anc["root_bary"]
        itp = lambda f: self._interp(f, idx, ba)
        xr = itp(xs)                                                 # [B,N,D]
        xr = self._augment_root_feat(xr, anc, idx, ba)              # v7 triplane (identity in v2)
        N = xr.shape[1]
        p = self.fur_head(xr)                                        # [B,N,19]
        dd = torch.tanh(p[..., 0:3]) * 0.35
        if "d_dir" in anc:
            dd = dd + anc["d_dir"]
        len_mult = 0.6 + 0.8 * torch.sigmoid(p[..., 3:4])
        albedo = torch.sigmoid(p[..., 4:7])
        blend = torch.sigmoid(p[..., 7:8])                           # sampled-vs-predicted
        op_logit = p[..., 8] + self.op_bias
        if "d_op" in anc:
            op_logit = op_logit + anc["d_op"]
        op = 0.9 * torch.sigmoid(op_logit)                           # [B,N] learnable gate
        if "nofur" in anc:                                           # v8: SMAL face/nose/paw region -> kill fur
            op = op * (1.0 - itp(anc["nofur"]).clamp(0, 1))
        r_logit = p[..., 9]
        if "d_radius" in anc:
            r_logit = r_logit + anc["d_radius"]
        radius = self.radius_frac * anc["diag"].view(B, 1) * torch.pow(3.0, torch.tanh(r_logit))
        t, b, n = self._tbn(itp(anc["t"]), itp(anc["n"]))            # [B,N,3]
        roots = itp(anc["roots"])
        L = itp(anc["L"]) * len_mult[..., 0]                         # [B,N]
        if "d_logL" in anc:
            L = L * anc["d_logL"].exp()
        # v6 coat-adaptive length recipe (FUR_V6_PLAN s8.2): per-part FIXED short
        # factors fed via anc (not optimized). All keys absent -> 1.0 -> r5 behavior.
        if "len_short" in anc:                                       # global shorten (~0.5)
            ls = anc["len_short"]
            L = L * (ls.view(B, 1) if torch.is_tensor(ls) and ls.dim() else float(ls))
        if "paw_short" in anc:                                       # per-vertex paw-mask shorten
            L = L * itp(anc["paw_short"])                            # [Vs] -> per-root, ~1 off-paw
        if "L_max" in anc:                                           # measured envelope:
            L = torch.minimum(L, itp(anc["L_max"]))                  # fur never exceeds the
        L = L.unsqueeze(-1)                                          # GT mask, any pose
        tmix = itp(anc["tmix"]).unsqueeze(-1) if "tmix" in anc else self.tangent_mix
        if "d_tmix" in anc:
            tmix = (tmix + anc["d_tmix"].unsqueeze(-1)).clamp(0, 0.97)
        d0 = F.normalize(tmix * t + (1 - tmix) * n
                         + dd[..., 0:1] * t + dd[..., 1:2] * b + dd[..., 2:3] * n, dim=-1)
        coat = self.curl_mlp(self.curl_emb(anc["curl_id"]))          # [B,4]
        ca, cf, coat_off, coat_lsc = coat.split(1, dim=-1)           # [B,1] each
        if "coat_curl" in anc:                                       # v6 manual curl override -> [B,2]
            ca = ca + anc["coat_curl"][..., 0:1]                     # logit residuals; absent -> r5
            cf = cf + anc["coat_curl"][..., 1:2]
        # coat len_scale: per-coat multiplicative length adjustment (zero-init -> 1.0).
        # bounded to [0.5,2.0] so the coat code nudges length without free optimization.
        L = L * (0.5 * 4.0 ** torch.sigmoid(coat_lsc)).view(B, 1, 1)
        if "d_curl_amp" in anc:
            ca = ca + anc["d_curl_amp"]
        if "d_curl_freq" in anc:
            cf = cf + anc["d_curl_freq"]
        ca_e, cf_e = ca.view(B, 1, 1), cf.view(B, 1, 1)
        if p.shape[-1] >= 23:               # v8 per-strand image-conditioned curl (zero-init -> neutral)
            ca_e = ca_e + p[..., 21:22]     # [B,N,1] residual on coat-class base curl
            cf_e = cf_e + p[..., 22:23]
        curl_amp = torch.sigmoid(ca_e) * 0.35                        # [B,1,1] (v7) or [B,N,1] (v8)
        fmax = 1.5 if Kp <= 11 else 3.0     # short polylines undersample fast helices
        curl_freq = 1.0 + fmax * torch.sigmoid(cf_e)
        phase = anc["root_phase"].unsqueeze(-1)                      # [B,N,1]
        tone = anc["root_tone"].unsqueeze(-1)
        g = anc["gravity"].view(B, 1, 3)
        u2 = F.normalize(torch.cross(d0, b, dim=-1), dim=-1)
        if "droop" in anc:
            droop = itp(anc["droop"]).unsqueeze(-1)                  # explicit override (animate)
        else:
            droop = torch.sigmoid(p[..., 20:21] + self.droop_bias)   # network-predicted; logit 0 -> droop0
        if "d_droop" in anc:
            droop = (droop + anc["d_droop"].unsqueeze(-1)).clamp(0, 1)
        gamma_logit = p[..., 19:20]                                  # per-root droop-profile exponent
        if "d_gamma" in anc:                                         # (zero-padded on r5 warm start ->
            gamma_logit = gamma_logit + anc["d_gamma"].unsqueeze(-1)  # logit 0 -> gamma 1.5 == old fixed)
        gamma = 1.5 * (3.0 ** torch.tanh(gamma_logit))               # in [0.5, 4.5]; lets the strand
        # bend sharply at the tip (the 'rise-then-flop' physics shape the fixed s^1.5 could not fit)

        # v6 offset-shell (s8.2): lift the strand origin off the surface along the
        # normal by off*L so short+curly coats read as fluff, not skin-tight strands.
        # roots (the true surface point) is kept for color sampling / pen reference.
        off = float(anc.get("offset_shell", 0.0)) if not torch.is_tensor(anc.get("offset_shell")) \
            else anc["offset_shell"]
        # coat offset residual: per-coat shell-lift adjustment (zero-init -> 0).
        # tanh-bounded to +/-0.2 so the coat code modulates fluffiness around the base off.
        off = off + (0.2 * torch.tanh(coat_off)).view(B, 1, 1)       # [B,1,1] (broadcasts on L,n)
        origin = roots + (off * L) * n
        pts, pcur = [origin], origin
        for k in range(1, Kp):
            s_frac = k / (Kp - 1)
            beta = droop * (s_frac ** gamma)
            dk = F.normalize((1 - beta) * d0 + beta * g, dim=-1)
            pcur = pcur + dk * L / (Kp - 1)
            th = 2 * np.pi * curl_freq * s_frac + phase
            curl = curl_amp * L * s_frac * (th.sin() * b + th.cos() * u2 - u2)
            pts.append(pcur + curl)
        pts = torch.stack(pts, 2)                                    # [B,N,K,3]

        mid = 0.5 * (pts[..., 1:, :] + pts[..., :-1, :])
        seg = pts[..., 1:, :] - pts[..., :-1, :]
        slen = seg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sdir = seg / slen
        frac = torch.linspace(0.2, 1.0, Kp - 1, device=xs.device).view(1, 1, -1, 1)
        # per-strand base color: ref-image root-projection sample where the root is
        # front-facing in the ref view, predicted albedo elsewhere (learned blend)
        base = albedo
        if not getattr(self, "pure_color", False):                  # v10: color = pure predicted (no ref-sample crutch)
            pf = getattr(self, "proj_floor", 0.0)                    # v11: floor projection where visible so
            bw = blend.clamp(min=pf) if pf > 0 else blend           # multi-view L1 can't suppress it (washed v8)
            if "samp_rgb" in anc:                                    # precomputed (anim)
                w = bw * anc["samp_vis"]
                base = w * anc["samp_rgb"] + (1 - w) * base
            elif "ref_rgb" in anc:
                samp, vis = self.sample_ref(roots, n, anc)
                w = bw * vis
                base = w * samp + (1 - w) * base
        if "d_rgb" in anc:
            base = base + anc["d_rgb"]
        tm = 1.0 if getattr(self, "pure_color", False) else tone[..., None]   # v10: drop per-strand tone mottle
        rgb = (base[:, :, None] * tm * (0.9 + 0.1 * frac)).clamp(0, 1)
        a_local = torch.tanh(p[..., 10:19]).view(B, N, 3, 3) * 0.25
        if "d_sh" in anc:                       # clamp to the BASE range (tanh*0.25) so
            a_local = (a_local + anc["d_sh"]).clamp(-0.25, 0.25)  # unbounded residuals
            # can't saturate channels under novel anim views (base stays unclipped)
        sh1 = self._sh1_world(a_local, t, b, n)                      # [B,N,3,3]
        sh = torch.cat([sh0_from_rgb(rgb)[..., None, :],
                        sh1[:, :, None].expand(B, N, Kp - 1, 3, 3)], dim=3)  # [B,N,K-1,4,3]

        # fur thinness: cap the cross-section radius to (half-seg / aspect) so each
        # gaussian is a thin prolate sliver aligned to the strand (quat_align_z puts
        # +z on sdir). aspect<=0 -> off (legacy ~spherical blobs). [B,N,Kp-1,1]
        aspect = float(anc["fur_aspect"]) if "fur_aspect" in anc else 0.0
        rxy = radius.view(B, N, 1, 1).expand(B, N, Kp - 1, 1)
        if aspect > 0:
            rxy = torch.minimum(rxy, (slen * 0.5) / aspect)
        out = []
        for bi in range(B):
            Nf = N * (Kp - 1)
            out.append(dict(
                means=mid[bi].reshape(Nf, 3),
                quats=quat_align_z(sdir[bi].reshape(Nf, 3)),
                scales=torch.cat([rxy[bi].expand(N, Kp - 1, 2),
                                  slen[bi] * 0.5], dim=-1).reshape(Nf, 3),
                opacities=op[bi].view(N, 1).expand(N, Kp - 1).reshape(Nf),
                sh=sh[bi].reshape(Nf, 4, 3),
                rgb=rgb[bi].reshape(Nf, 3),
                tangent=sdir[bi].reshape(Nf, 3),
                pts=pts[bi],
                root=roots[bi],
                nrm=n[bi],
                droop=droop[bi, :, 0],                               # [N] for drape-GT supervision
                gamma=gamma[bi, :, 0],
                len_mult=len_mult[bi, :, 0]))                        # [N] network length factor (v6 sym-geo)
        return out

    def forward(self, img, label, canon, anc, subdivide):  # overrides body-only parent
        x = self.features(img, label, canon, anc.get("face_crop"))
        return self.from_features(subdivide(x), anc)

    def from_features(self, xs, anc):
        """Gaussians from precomputed subdivided features (per-dog refinement runs the
        frozen encoder once and re-decodes with residual overrides in `anc`)."""
        fur = self.fur_gaussians(xs, anc)
        e = self.face_edges
        md = lambda f: self._mids(f, e)
        xb = md(xs)                                                  # [B,Vb,D]
        q = self.body_rgb_head(xb)                                   # [B,Vb,14]
        roots_b = md(anc["roots"])
        t_b, b_b, n_b = self._tbn(md(anc["t"]), md(anc["n"]))
        wf = md(anc["w_face"])
        body_rgb = torch.sigmoid(q[..., 0:3])
        blend_b = torch.sigmoid(q[..., 3:4])
        if "samp_rgb_b" in anc:                                      # precomputed (anim)
            w = blend_b * anc["samp_vis_b"]
            body_rgb = w * anc["samp_rgb_b"] + (1 - w) * body_rgb
        elif "ref_rgb" in anc:                                       # face detail from ref px
            samp, vis = self.sample_ref(roots_b, n_b, anc)
            w = blend_b * vis
            body_rgb = w * samp + (1 - w) * body_rgb
        if "d_rgb_b" in anc:
            body_rgb = body_rgb + anc["d_rgb_b"]
        body_rgb = body_rgb.clamp(0, 1)
        gate = q[..., 4]
        if "d_op_b" in anc:
            gate = gate + anc["d_op_b"]
        body_op = torch.sigmoid(gate + (-8.0 * (1 - wf) + 3.0 * wf))  # r4 field at init
        a_local = torch.tanh(q[..., 5:14]).view(*q.shape[:2], 3, 3) * 0.25
        if "d_sh_b" in anc:
            a_local = (a_local + anc["d_sh_b"]).clamp(-0.25, 0.25)
        sh_b = torch.cat([sh0_from_rgb(body_rgb)[..., None, :],
                          self._sh1_world(a_local, t_b, b_b, n_b)], dim=2)   # [B,Vb,4,3]
        B, Vb, _ = xb.shape
        Vs = xs.shape[1]
        sc1 = torch.cat([torch.full((Vs,), 0.008, device=xs.device),
                         torch.full((e.shape[0],), 0.004, device=xs.device)])  # face finer
        q0 = torch.zeros(Vb, 4, device=xs.device)
        q0[:, 0] = 1
        body = []
        for bi in range(B):
            body.append(dict(means=roots_b[bi], quats=q0,
                             scales=(sc1 * anc["diag"][bi]).view(Vb, 1).expand(Vb, 3),
                             opacities=body_op[bi], sh=sh_b[bi], rgb=body_rgb[bi]))
        return fur, body


class DogLRMFurV7(DogLRMFurV2):
    """v7 (FUR_V7_PLAN): real capacity to cure body blur.
      - backbone scale-up via **kw (dim/n_layers/n_heads -> DogLRMDecomp)
      - 3-layer MLP heads
      - TGS-style TRIPLANE feature field predicted from the image; every output point
        (fur root + body vert) samples it at its CANONICAL xyz (animation-safe) and the
        sampled feature is ADDED before the heads -> dense high-freq decoupled from the
        coarse 3889-token transformer (the real blur fix)
      - BODY on subdiv level-2 (62194 verts) instead of level-1+face-edges (29121)
    Region short-fur (face/nose/ear/leg) is handled in the data recipe (FurScenes v7)."""

    def __init__(self, w_face, faces_sub=None, w_face_s=None, K=6,
                 tri_res=64, tri_ch=32, n_tri_layers=4, n_heads=12, **kw):
        super().__init__(w_face, faces_sub=faces_sub, w_face_s=w_face_s, K=K,
                         n_heads=n_heads, **kw)
        D = self.img_proj.out_features
        self.tri_res, self.tri_ch = tri_res, tri_ch
        self.tri_tokens = nn.Parameter(torch.randn(3 * tri_res * tri_res, D) * 0.02)
        self.tri_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(D, n_heads, D * 4, batch_first=True,
                                       norm_first=True, activation="gelu"), n_tri_layers)
        self.tri_to_feat = nn.Linear(D, tri_ch)
        self.tri_mlp = nn.Sequential(nn.Linear(3 * tri_ch, D), nn.SiLU(), nn.Linear(D, D))
        nn.init.zeros_(self.tri_mlp[-1].weight)              # zero -> neutral at init
        nn.init.zeros_(self.tri_mlp[-1].bias)
        # deeper (3-layer) MLP heads for capacity; zero-init final layer
        self.fur_head = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D),
                                      nn.SiLU(), nn.Linear(D, 21))
        self.body_rgb_head = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D),
                                           nn.SiLU(), nn.Linear(D, 14))
        for head in (self.fur_head, self.body_rgb_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self._sub2_M = None
        self._tri = self._bbox = self._canon_l1 = self._canon_l2 = None

    def _sub2(self, x):
        """level-1 (Vs) -> level-2 subdivide, batched. x [B,Vs,C] -> [B,Vs2,C]."""
        if self._sub2_M is None:
            from dog_lrm.smal_model import build_subdiv
            self._sub2_M = build_subdiv(self.faces_sub, 1, x.device)
        return torch.stack([torch.sparse.mm(self._sub2_M, x[b]) for b in range(x.shape[0])])

    def make_triplane(self, img_tok):
        B = img_tok.shape[0]
        q = self.tri_tokens[None].expand(B, -1, -1)          # [B,3*R*R,D]
        t = self.tri_to_feat(self.tri_decoder(q, img_tok))   # [B,3*R*R,C]
        R = self.tri_res
        return t.reshape(B, 3, R, R, self.tri_ch).permute(0, 1, 4, 2, 3).contiguous()

    def sample_triplane(self, pts_canon):
        """pts_canon [B,N,3] canonical -> [B,N,D] added to per-point features."""
        tri = self._tri
        lo, hi = self._bbox
        p = (pts_canon - lo) / (hi - lo).clamp_min(1e-6) * 2 - 1
        feats = []
        for pi, (a, c) in enumerate(((0, 1), (0, 2), (1, 2))):     # xy, xz, yz planes
            grid = torch.stack([p[..., a], p[..., c]], -1)[:, :, None, :]  # [B,N,1,2]
            s = F.grid_sample(tri[:, pi], grid, mode="bilinear",
                              align_corners=True, padding_mode="border")    # [B,C,N,1]
            feats.append(s[..., 0].permute(0, 2, 1))                        # [B,N,C]
        return self.tri_mlp(torch.cat(feats, -1))

    def _augment_root_feat(self, xr, anc, idx, ba):
        root_canon = self._interp(self._canon_l1, idx, ba)   # [B,N,3] canonical roots
        return xr + self.sample_triplane(root_canon)

    def forward(self, img, label, canon, anc, subdivide):
        x = self.features(img, label, canon, anc.get("face_crop"))   # [B,3889,D]
        img_tok = self.img_proj(self.encode_image(img))              # [B,G*G,D]
        self._tri = self.make_triplane(img_tok)
        lo, hi = canon.amin(1, keepdim=True), canon.amax(1, keepdim=True)
        pad = (hi - lo) * 0.05
        self._bbox = (lo - pad, hi + pad)
        self._canon_l1 = subdivide(canon)                            # [B,Vs,3]
        self._canon_l2 = self._sub2(self._canon_l1)                  # [B,Vs2,3]
        return self.from_features(subdivide(x), anc)

    def from_features(self, xs, anc):
        fur = self.fur_gaussians(xs, anc)                            # triplane via hook
        # ---- body on subdiv level-2 (62194), triplane-augmented ----
        xb = self._sub2(xs)                                          # [B,Vs2,D]
        xb = xb + self.sample_triplane(self._canon_l2)
        q = self.body_rgb_head(xb)
        roots_b = self._sub2(anc["roots"])
        t_b, b_b, n_b = self._tbn(self._sub2(anc["t"]), self._sub2(anc["n"]))
        wf = self._sub2(anc["w_face"][..., None])[..., 0] if anc["w_face"].dim() == 2 \
            else self._sub2(anc["w_face"])
        body_rgb = torch.sigmoid(q[..., 0:3])
        blend_b = torch.sigmoid(q[..., 3:4])
        if not getattr(self, "pure_color", False):                  # v10: body color = pure predicted too
            pf = getattr(self, "proj_floor", 0.0)                    # v11: floor projection where visible
            bw = blend_b.clamp(min=pf) if pf > 0 else blend_b
            if "samp_rgb_b" in anc:
                w = bw * anc["samp_vis_b"]
                body_rgb = w * anc["samp_rgb_b"] + (1 - w) * body_rgb
            elif "ref_rgb" in anc:
                samp, vis = self.sample_ref(roots_b, n_b, anc)
                w = bw * vis
                body_rgb = w * samp + (1 - w) * body_rgb
        if "d_rgb_b" in anc:
            body_rgb = body_rgb + anc["d_rgb_b"]
        body_rgb = body_rgb.clamp(0, 1)
        gate = q[..., 4]
        if "d_op_b" in anc:
            gate = gate + anc["d_op_b"]
        body_op = torch.sigmoid(gate + (-8.0 * (1 - wf) + 3.0 * wf))
        a_local = torch.tanh(q[..., 5:14]).view(*q.shape[:2], 3, 3) * 0.25
        if "d_sh_b" in anc:
            a_local = (a_local + anc["d_sh_b"]).clamp(-0.25, 0.25)
        sh_b = torch.cat([sh0_from_rgb(body_rgb)[..., None, :],
                          self._sh1_world(a_local, t_b, b_b, n_b)], dim=2)
        B, Vb, _ = xb.shape
        sc1 = torch.full((Vb,), 0.004, device=xs.device)             # finer mesh -> smaller
        q0 = torch.zeros(Vb, 4, device=xs.device)
        q0[:, 0] = 1
        body = []
        for bi in range(B):
            body.append(dict(means=roots_b[bi], quats=q0,
                             scales=(sc1 * anc["diag"][bi]).view(Vb, 1).expand(Vb, 3),
                             opacities=body_op[bi], sh=sh_b[bi], rgb=body_rgb[bi]))
        return fur, body


class DogLRMFurV8(DogLRMFurV7):
    """v8 (FUR_V8_PLAN): per-strand curl head + face/paw fur-kill + FREELY-OPTIMIZED body.
      - fur_head 21->23: +per-strand curl_amp/freq residual (base fur_gaussians reads idx 21,22).
      - body branch geometry is no longer frozen: body_rgb_head 14->24 predicts, per level-2 vert,
        a position offset / anisotropic scale / rotation -- ALL in the CANONICAL local TBN frame so
        D-SMAL skinning still carries them (animation-safe). Zero-init the new rows -> at init the
        body = spacing-sized (body_base_sc) spheres at the verts (overlapping -> no 斑点), then it
        learns to spread/tilt. nofur (face/paw) applied via anc in base fur_gaussians."""

    def __init__(self, w_face, faces_sub=None, w_face_s=None, K=6,
                 body_off_bound=0.05, body_base_sc=0.010, body_thin=0.1, **kw):
        super().__init__(w_face, faces_sub=faces_sub, w_face_s=w_face_s, K=K, **kw)
        D = self.img_proj.out_features
        self.body_off_bound, self.body_base_sc = body_off_bound, body_base_sc
        self.body_thin = body_thin                          # normal-axis (z) scale factor -> flat surfel
        self.fur_head = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D),
                                      nn.SiLU(), nn.Linear(D, 23))      # +curl_amp,+curl_freq
        self.body_rgb_head = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D),
                                           nn.SiLU(), nn.Linear(D, 24))  # +offset(3)+logscale(3)+quat(4)
        for head in (self.fur_head, self.body_rgb_head):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def from_features(self, xs, anc):
        from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
        fur = self.fur_gaussians(xs, anc)
        xb = self._sub2(xs)
        xb = xb + self.sample_triplane(self._canon_l2)
        q = self.body_rgb_head(xb)                                   # [B,Vb,24]
        roots_b = self._sub2(anc["roots"])
        t_b, b_b, n_b = self._tbn(self._sub2(anc["t"]), self._sub2(anc["n"]))
        wf = self._sub2(anc["w_face"][..., None])[..., 0] if anc["w_face"].dim() == 2 \
            else self._sub2(anc["w_face"])
        body_rgb = torch.sigmoid(q[..., 0:3])
        blend_b = torch.sigmoid(q[..., 3:4])
        if "samp_rgb_b" in anc:
            w = blend_b * anc["samp_vis_b"]; body_rgb = w * anc["samp_rgb_b"] + (1 - w) * body_rgb
        elif "ref_rgb" in anc:
            samp, vis = self.sample_ref(roots_b, n_b, anc)
            w = blend_b * vis; body_rgb = w * samp + (1 - w) * body_rgb
        if "d_rgb_b" in anc:
            body_rgb = body_rgb + anc["d_rgb_b"]
        body_rgb = body_rgb.clamp(0, 1)
        gate = q[..., 4]
        if "d_op_b" in anc:
            gate = gate + anc["d_op_b"]
        body_op = torch.sigmoid(gate + (-8.0 * (1 - wf) + 3.0 * wf))
        a_local = torch.tanh(q[..., 5:14]).view(*q.shape[:2], 3, 3) * 0.25
        if "d_sh_b" in anc:
            a_local = (a_local + anc["d_sh_b"]).clamp(-0.25, 0.25)
        sh_b = torch.cat([sh0_from_rgb(body_rgb)[..., None, :],
                          self._sh1_world(a_local, t_b, b_b, n_b)], dim=2)
        B, Vb, _ = xb.shape
        diag = anc["diag"].view(B, 1, 1)
        # ---- FREE body geometry in canonical local TBN frame (zero-init -> neutral) ----
        geo = q[..., 14:24]
        off = torch.tanh(geo[..., 0:3]) * self.body_off_bound * diag          # [B,Vb,3] in (t,b,n)
        means_b = (roots_b + off[..., 0:1] * t_b + off[..., 1:2] * b_b + off[..., 2:3] * n_b)
        sc = torch.exp(torch.tanh(geo[..., 3:6]) * 0.8)                      # [B,Vb,3] learnable
        sc = sc * sc.new_tensor([1.0, 1.0, self.body_thin])                  # thin NORMAL axis (local z=n_b) -> flat surfel
        scales_b = (self.body_base_sc * diag) * sc
        R_tbn = torch.stack([t_b, b_b, n_b], dim=-1)                          # [B,Vb,3,3], det +1
        q_base = matrix_to_quaternion(R_tbn)
        q_res = F.normalize(geo[..., 6:10] + q_base.new_tensor([1., 0, 0, 0]), dim=-1)
        quats_b = quaternion_multiply(q_base, q_res)                         # local frame (x)resid
        body = [dict(means=means_b[bi], quats=quats_b[bi], scales=scales_b[bi],
                     opacities=body_op[bi], sh=sh_b[bi], rgb=body_rgb[bi]) for bi in range(B)]
        return fur, body


class DogLRMFurV9(DogLRMFurV8):
    """v9 (FUR_V9_PLAN, Splatter-Image, SINGLE-image): add a pixel-aligned splat branch.
    A conv decoder on the DINO feature map predicts, per pixel of an RxR grid over the ref
    image, a gaussian whose POSITION is the ref-view ray unprojected at the body-surface depth
    (rendered from the body gaussians) + a small learned residual, COLOR is sampled directly
    from the input image (-> input view reconstructs sharply), small isotropic scale + learned
    opacity. Added to the body gaussian set. Zero-init head -> opacity≈0.5*... actually op_logit
    0 -> on; positions on surface -> renders correctly at all (same-pose) sup views. Animation
    binding (to nearest D-SMAL face) is a follow-up; training/eval are single fit-pose so static
    splats are correct. nofur/curl/free-body all inherited from v8."""

    def __init__(self, w_face, faces_sub=None, w_face_s=None, K=6,
                 splat_res=128, splat_base_sc=0.004, splat_dres=0.05, **kw):
        super().__init__(w_face, faces_sub=faces_sub, w_face_s=w_face_s, K=K, **kw)
        D = self.img_proj.out_features
        self.splat_res, self.splat_base_sc, self.splat_dres = splat_res, splat_base_sc, splat_dres
        self.splat_dec = nn.Sequential(nn.Conv2d(D, 128, 3, padding=1), nn.SiLU(),
                                       nn.Conv2d(128, 64, 3, padding=1), nn.SiLU(),
                                       nn.Conv2d(64, 3, 1))   # per-pixel: depth_resid, log_scale, op_logit
        nn.init.zeros_(self.splat_dec[-1].weight); nn.init.zeros_(self.splat_dec[-1].bias)
        self._splat_grid = None

    def forward(self, img, label, canon, anc, subdivide):
        x = self.features(img, label, canon, anc.get("face_crop"))
        img_tok = self.img_proj(self.encode_image(img))             # [B,g*g,D]
        self._tri = self.make_triplane(img_tok)
        g = int(round(img_tok.shape[1] ** 0.5))
        self._splat_grid = img_tok.transpose(1, 2).reshape(img_tok.shape[0], -1, g, g)  # [B,D,g,g]
        lo, hi = canon.amin(1, keepdim=True), canon.amax(1, keepdim=True)
        pad = (hi - lo) * 0.05
        self._bbox = (lo - pad, hi + pad)
        self._canon_l1 = subdivide(canon)
        self._canon_l2 = self._sub2(self._canon_l1)
        return self.from_features(subdivide(x), anc)

    def from_features(self, xs, anc):
        fur, body = super().from_features(xs, anc)                   # v8 fur + thin-surfel body
        if "ref_rgb" not in anc or self._splat_grid is None:
            return fur, body
        from dog_lrm.render import render_gaussians
        R = self.splat_res
        sp = self.splat_dec(F.interpolate(self._splat_grid, size=(R, R),
                                          mode="bilinear", align_corners=False))   # [B,3,R,R]
        B = sp.shape[0]
        for bi in range(B):
            c2w, K = anc["ref_c2w"][bi], anc["ref_K"][bi]
            img = anc["ref_rgb"][bi]                                 # [3,H,W]
            H, W = img.shape[1:]
            bd = body[bi]
            with torch.no_grad():                                    # body-surface depth at ref view
                _, alpha, depth = render_gaussians(bd["means"], bd["quats"], bd["scales"],
                                                   bd["opacities"], bd["sh"], c2w, K, W, H,
                                                   sh_degree=1, return_depth=True)
            # RxR pixel centers -> ref pixel coords
            ys = (torch.arange(R, device=sp.device) + 0.5) * H / R
            xs_ = (torch.arange(R, device=sp.device) + 0.5) * W / R
            vv, uu = torch.meshgrid(ys, xs_, indexing="ij")          # [R,R]
            gx = (uu / (W - 1)) * 2 - 1
            gy = (vv / (H - 1)) * 2 - 1
            grid = torch.stack([gx, gy], -1)[None]                   # [1,R,R,2]
            dep = F.grid_sample(depth.permute(2, 0, 1)[None], grid, align_corners=True)[0, 0]   # [R,R]
            al = F.grid_sample(alpha.permute(2, 0, 1)[None], grid, align_corners=True)[0, 0]
            col = F.grid_sample(img[None], grid, align_corners=True)[0].permute(1, 2, 0)        # [R,R,3]
            dres, lsc, oplg = sp[bi]                                  # each [R,R]
            z = (dep + torch.tanh(dres) * self.splat_dres * anc["diag"][bi]).clamp_min(1e-4)
            camx = (uu - K[0, 2]) / K[0, 0] * z
            camy = (vv - K[1, 2]) / K[1, 1] * z
            cam = torch.stack([camx, camy, z], -1).reshape(-1, 3)    # [R*R,3]
            world = cam @ c2w[:3, :3].T + c2w[:3, 3]                 # [R*R,3]
            valid = (al.reshape(-1) > 0.5) & (dep.reshape(-1) > 1e-4)
            op = (0.9 * torch.sigmoid(oplg.reshape(-1))) * valid.float()
            sc = (self.splat_base_sc * anc["diag"][bi]) * torch.exp(torch.tanh(lsc.reshape(-1)) * 0.8)
            scales = sc[:, None].expand(-1, 3)
            quats = torch.zeros(R * R, 4, device=sp.device); quats[:, 0] = 1
            rgb = col.reshape(-1, 3).clamp(0, 1)
            sh = torch.cat([sh0_from_rgb(rgb)[:, None, :],
                            torch.zeros(R * R, 3, 3, device=sp.device)], dim=1)
            body[bi] = {k: torch.cat([body[bi][k], v]) for k, v in
                        dict(means=world, quats=quats, scales=scales, opacities=op, sh=sh,
                             rgb=rgb).items()}
        return fur, body


class DogLRMFurV10(DogLRMFurV8):
    """v10 (user 2026-06-20): v8 geometry (thin-surfel free body + curl + nofur) but COLOR is
    100% MODEL-PREDICTED -- the ref-image sample/blend crutch and the per-strand `tone` mottle are
    dropped (they smeared spot patterns + desaturated far side). Anchors = rough geometry only;
    albedo+SH heads predict all appearance. Pure prediction is initially blurrier -> sharpness must
    come from capacity + the generative prior IN the FF loop (adversarial/distillation). NOT v9
    (splat also copied ref pixels). Warm-start v8 (color heads adapt from their partial v8 state)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pure_color = True


def load_fur_ckpt(model, path, device="cuda"):
    """Load an r4 state dict into an r5 model: zero-pad the widened head rows
    (8->19 fur, 3->14 body; zero rows = neutral residuals) and drop the stale
    S-dependent buffers. Also accepts r5 dicts unchanged."""
    sd = torch.load(path, map_location=device)
    for k in ("disk_r", "disk_a", "phase", "tone"):
        sd.pop(k, None)
    for k, newp in (("fur_head.2", model.fur_head[-1]),
                    ("body_rgb_head.2", model.body_rgb_head[-1]),
                    ("curl_mlp.2", model.curl_mlp[-1])):  # v6: 2->4 (offset_shell+len_scale zero-pad)
        w = sd.get(k + ".weight")
        if w is not None and w.shape[0] < newp.weight.shape[0]:
            for suf, tgt in ((".weight", newp.weight), (".bias", newp.bias)):
                old = sd[k + suf]
                pad = torch.zeros_like(tgt)
                pad[: old.shape[0]] = old
                sd[k + suf] = pad
    model.load_state_dict(sd, strict=False)
    return model
