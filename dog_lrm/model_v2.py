"""Dog-LRM v2: same overall design as dog_lrm/model.py (frozen DINOv2 tokens +
canonical point tokens -> joint transformer -> subdivide -> per-point refine ->
Gaussian heads with bounded offsets off the SMAL surface), with the backbone
swapped for the modernized MM-DiT in dit_v2.py and scaled 384/4L -> 512/8L.

Interface is identical to DogLRM so trainers/eval scripts can switch via a flag.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import fourier_embed
from .dit_v2 import MMDiTv2, CrossBlockV2, rope_2d


class DogLRMv2(nn.Module):
    def __init__(self, dim=512, n_layers=8, n_heads=8, n_freq=8, n_refine=2,
                 dino_name="facebook/dinov2-large", offset_max=0.15, base_scale=0.02,
                 gaussians_per_point=2, refine_chunk=16384):
        super().__init__()
        from transformers import AutoModel
        self.dino = AutoModel.from_pretrained(dino_name)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        self.offset_max, self.base_scale, self.K = offset_max, base_scale, gaussians_per_point
        self.head_dim = dim // n_heads
        self.refine_chunk = refine_chunk

        self.img_proj = nn.Linear(self.dino.config.hidden_size, dim)
        pe = 3 + 3 * 2 * n_freq
        self.pt_proj = nn.Sequential(nn.Linear(pe, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.transformer = MMDiTv2(dim, n_layers, n_heads)
        self.temb_proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # per-point refinement on the high-res subdivided surface (cross-attn only)
        self.pt_proj_fine = nn.Sequential(nn.Linear(pe, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.refine = nn.ModuleList([CrossBlockV2(dim, n_heads) for _ in range(n_refine)])
        # pixel-aligned appearance (plan A): shallow stride-4 CNN over a hi-res ref crop;
        # each anchor projects into the ref view and samples its own local feature + RGB.
        # proj_in zero-init -> inert at warm-start; swappable for an upsampled-DINO map
        # (FeatUp/JAFAR) later without touching the rest of the pipeline.
        self.ref_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1))
        self.proj_in = nn.Linear(64 + 3 + 1, dim)
        nn.init.zeros_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)

        K = self.K
        self.head_offset = nn.Linear(dim, K * 3)
        self.head_scale = nn.Linear(dim, K * 3)
        self.head_quat = nn.Linear(dim, K * 4)
        self.head_opacity = nn.Linear(dim, K * 1)
        self.head_rgb = nn.Linear(dim, K * 3)
        for h in (self.head_offset, self.head_scale, self.head_quat,
                  self.head_opacity, self.head_rgb):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._rope = None                                    # cached (cos,sin) per grid

    @torch.no_grad()
    def encode_image(self, img, res=224):
        x = F.interpolate((img - self.mean) / self.std, size=(res, res),
                          mode="bilinear", align_corners=False)
        return self.dino(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS

    def _refine(self, x, canon, proj, img_ctx, rope_cos, rope_sin):
        q = x + self.pt_proj_fine(fourier_embed(canon))
        if proj is not None:
            q = q + proj
        for blk in self.refine:
            q = blk(q, img_ctx, rope=(rope_cos, rope_sin))
        return q

    def _pixel_aligned(self, posed_pts, ref_hi, ref_K, ref_c2w, anchor_normals):
        """Project each anchor into the hi-res ref view, sample its own local feature
        + RGB, gate by visibility (front-facing + in-frustum). [B,N,3] -> [B,N,D]."""
        feat = self.ref_cnn(ref_hi)                                     # [B,C,h,w] stride 4
        w2c = torch.inverse(ref_c2w)
        Xc = torch.einsum("bij,bnj->bni", w2c[:, :3, :3], posed_pts) + w2c[:, None, :3, 3]
        z = Xc[..., 2]
        zs = z.clamp(min=1e-6)
        u = Xc[..., 0] / zs * ref_K[:, None, 0, 0] + ref_K[:, None, 0, 2]
        v = Xc[..., 1] / zs * ref_K[:, None, 1, 1] + ref_K[:, None, 1, 2]
        Wp, Hp = ref_hi.shape[-1], ref_hi.shape[-2]
        gx, gy = u / (Wp - 1) * 2 - 1, v / (Hp - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1)[:, None]                       # [B,1,N,2]
        sf = F.grid_sample(feat, grid, align_corners=True, padding_mode="zeros")[:, :, 0]
        sr = F.grid_sample(ref_hi, grid, align_corners=True, padding_mode="zeros")[:, :, 0]
        vis = (z > 0) & (gx.abs() <= 1) & (gy.abs() <= 1)
        if anchor_normals is not None:                                  # front-facing only
            vdir = F.normalize(posed_pts - ref_c2w[:, None, :3, 3], dim=-1)
            vis = vis & ((anchor_normals * vdir).sum(-1) < 0)
        vis = vis.float()[..., None]                                    # [B,N,1]
        samp = torch.cat([sf, sr], 1).permute(0, 2, 1) * vis            # occluded -> 0
        return self.proj_in(torch.cat([samp, vis], -1))                 # zero-init head

    def _rope_for(self, n_tok, device):
        g = int(n_tok ** 0.5)
        assert g * g == n_tok, f"image tokens {n_tok} not a square grid"
        if self._rope is None or self._rope[0].shape[0] != n_tok or self._rope[0].device != device:
            self._rope = rope_2d(g, g, self.head_dim, device)
        return self._rope

    def forward(self, img, canonical_pts, posed_pts, subdivide=None, scale_clip=None, return_feat=False,
                ref_hi=None, ref_K=None, ref_c2w=None, anchor_normals=None):
        img_tok = self.img_proj(self.encode_image(img))            # [B,M,D]
        rope = self._rope_for(img_tok.shape[1], img_tok.device)
        temb = self.temb_proj(img_tok.mean(dim=1))                 # [B,D] global cond for adaLN
        pt = self.pt_proj(fourier_embed(canonical_pts))            # [B,N,D]
        x, img_ctx = self.transformer(pt, img_tok, temb, rope=rope)  # joint attention
        canon = canonical_pts
        if subdivide is not None:                                  # densify output surface
            x = subdivide(x)                                       # [B,Vsub,D] (interpolated)
            canon = subdivide(canonical_pts)                       # [B,Vsub,3]
            posed_pts = subdivide(posed_pts)                       # [B,Vsub,3]
        # individuate each subdivided point by its own position, then pull appearance
        # from the (transformer-updated) image tokens. Cross-attn is independent per
        # query point, so chunk + checkpoint: exact math, activation memory bounded
        # per chunk (needed for ~250k anchors at n_subdiv=3).
        proj = None
        if ref_hi is not None:
            proj = self._pixel_aligned(posed_pts, ref_hi, ref_K, ref_c2w, anchor_normals)
        if self.refine_chunk and x.shape[1] > self.refine_chunk:
            from torch.utils.checkpoint import checkpoint
            ck = torch.is_grad_enabled()
            parts = []
            for s in range(0, x.shape[1], self.refine_chunk):
                a, c = x[:, s:s + self.refine_chunk], canon[:, s:s + self.refine_chunk]
                p = proj[:, s:s + self.refine_chunk] if proj is not None else None
                parts.append(checkpoint(self._refine, a, c, p, img_ctx, rope[0], rope[1],
                                        use_reentrant=False)
                             if ck else self._refine(a, c, p, img_ctx, rope[0], rope[1]))
            x = torch.cat(parts, dim=1)
        else:
            x = self._refine(x, canon, proj, img_ctx, rope[0], rope[1])
        B, N, _ = x.shape
        K = self.K

        offset = torch.tanh(self.head_offset(x)).view(B, N, K, 3) * self.offset_max
        scales = torch.exp(self.head_scale(x).view(B, N, K, 3).clamp(-6, 2)) * self.base_scale
        if scale_clip is not None:                                 # warmup: cap max scale
            scales = scales.clamp(max=scale_clip)
        q0 = torch.tensor([1.0, 0, 0, 0], device=x.device)
        quats = F.normalize(self.head_quat(x).view(B, N, K, 4) + q0, dim=-1)
        opacities = torch.sigmoid(self.head_opacity(x).view(B, N, K) + 2.0)
        rgb = torch.sigmoid(self.head_rgb(x).view(B, N, K, 3))
        means = posed_pts[:, :, None, :] + offset                  # [B,N,K,3]
        out = dict(means=means.reshape(B, N * K, 3),
                   scales=scales.reshape(B, N * K, 3),
                   quats=quats.reshape(B, N * K, 4),
                   opacities=opacities.reshape(B, N * K),
                   rgb=rgb.reshape(B, N * K, 3),
                   offset=offset.reshape(B, N * K, 3))
        if return_feat:
            out["feat"] = x                                        # [B,N,D] per-anchor feature (for Stage-2 fur head)
        return out
