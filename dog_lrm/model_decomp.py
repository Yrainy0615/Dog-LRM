"""Dog-LRM v2: face/body decomposed tokens (FUR_V2_PLAN P1a, body Gaussians only).

Image branch: DINO at 518 -> 37x37 feature grid, bilinearly upsampled x2 -> 74x74,
split into face/body token sets by the projected region label grid. Geometry branch:
canonical-vert tokens split by template w_face. Region-restricted cross-attention
(face<->face, body<->body), each memory gets a shared whole-dog pooled token for
consistent tone. Features are scattered back to the full vert set (soft-blended in
the 0.2<w<0.8 neck band) and decoded by the same Gaussian heads as v1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dog_lrm.model import fourier_embed

GRID = 74                       # upsampled image-token grid (37x37 DINO x2)
N_FACE_TOK, N_BODY_TOK = 256, 1024


def _sample_idx(mask_flat, n):
    """Indices of true cells, sampled/padded to exactly n (repeat if fewer)."""
    idx = mask_flat.nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return torch.zeros(n, dtype=torch.long, device=mask_flat.device)
    r = torch.randint(len(idx), (n,), device=mask_flat.device) if len(idx) < n else \
        torch.randperm(len(idx), device=mask_flat.device)[:n]
    return idx[r]


class DogLRMDecomp(nn.Module):
    def __init__(self, w_face, dim=384, n_layers=4, n_heads=6, n_freq=8,
                 dino_name="facebook/dinov2-large", offset_max=0.15, base_scale=0.02,
                 gaussians_per_point=2, dino_res=518):
        super().__init__()
        from transformers import AutoModel
        self.dino = AutoModel.from_pretrained(dino_name)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        self.offset_max, self.base_scale, self.K = offset_max, base_scale, gaussians_per_point
        self.dino_res = dino_res

        w = w_face.float()
        self.register_buffer("w_face", w)
        self.register_buffer("idx_f", (w > 0.2).nonzero(as_tuple=True)[0])   # face stream verts
        self.register_buffer("idx_b", (w < 0.8).nonzero(as_tuple=True)[0])   # body stream verts

        self.img_proj = nn.Linear(self.dino.config.hidden_size, dim)
        pe = 3 + 3 * 2 * n_freq
        self.pt_proj = nn.Sequential(nn.Linear(pe, dim), nn.SiLU(), nn.Linear(dim, dim))
        mk = lambda: nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, n_heads, dim * 4, batch_first=True,
                                       norm_first=True, activation="gelu"), n_layers)
        self.tf_face, self.tf_body = mk(), mk()

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

    @torch.no_grad()
    def encode_image(self, img):
        x = F.interpolate((img - self.mean) / self.std, size=(self.dino_res, self.dino_res),
                          mode="bilinear", align_corners=False)
        tok = self.dino(pixel_values=x).last_hidden_state[:, 1:]              # [B,37*37,C]
        g = self.dino_res // 14
        fm = tok.permute(0, 2, 1).reshape(tok.shape[0], -1, g, g)
        fm = F.interpolate(fm, size=(GRID, GRID), mode="bilinear", align_corners=False)
        return fm.flatten(2).permute(0, 2, 1)                                 # [B,GRID*GRID,C]

    def region_tokens(self, img, label, face_crop=None):
        """label [B,GRID,GRID] int: 0=bg 1=body 2=face -> per-region token sets.
        face_crop [B,3,H,W]: a high-res crop of the face re-encoded through DINO; when
        given, the face token set is drawn from it (dense face detail) instead of the few
        face patches in the full 518 image."""
        feat = self.img_proj(self.encode_image(img))                          # [B,G*G,D]
        lab = label.flatten(1)                                                # [B,G*G]
        fc = self.img_proj(self.encode_image(face_crop)) if face_crop is not None else None
        all_true = torch.ones(GRID * GRID, dtype=torch.bool, device=feat.device)
        toks_f, toks_b = [], []
        for b in range(feat.shape[0]):
            dog = lab[b] > 0
            glob = feat[b][dog].mean(0, keepdim=True) if dog.any() else feat[b].mean(0, keepdim=True)
            b_idx = _sample_idx(lab[b] == 1, N_BODY_TOK)
            if fc is not None:                                                # dense crop tokens
                f_idx = _sample_idx(all_true, N_FACE_TOK)
                toks_f.append(torch.cat([fc[b][f_idx], glob], 0))
            else:
                toks_f.append(torch.cat([feat[b][_sample_idx(lab[b] == 2, N_FACE_TOK)], glob], 0))
            toks_b.append(torch.cat([feat[b][b_idx], glob], 0))
        return torch.stack(toks_f), torch.stack(toks_b)

    def features(self, img, label, canonical_pts, face_crop=None):
        """Region-routed per-vertex features [B,V,D] (shared by all decoder heads)."""
        mem_f, mem_b = self.region_tokens(img, label, face_crop)
        pt = self.pt_proj(fourier_embed(canonical_pts))                       # [B,V,D]
        xf = self.tf_face(pt[:, self.idx_f], mem_f)
        xb = self.tf_body(pt[:, self.idx_b], mem_b)
        B, V, D = pt.shape
        x = pt.new_zeros(B, V, D)
        wf = self.w_face.clamp(0, 1)
        x[:, self.idx_f] += xf * wf[self.idx_f, None]                         # soft blend in
        x[:, self.idx_b] += xb * (1 - wf)[self.idx_b, None]                   # the neck band
        return x

    def forward(self, img, label, canonical_pts, posed_pts, subdivide=None):
        x = self.features(img, label, canonical_pts)

        if subdivide is not None:
            x = subdivide(x)
            posed_pts = subdivide(posed_pts)
        B, N, _ = x.shape
        K = self.K
        offset = torch.tanh(self.head_offset(x)).view(B, N, K, 3) * self.offset_max
        scales = torch.exp(self.head_scale(x).view(B, N, K, 3).clamp(-6, 2)) * self.base_scale
        q0 = torch.tensor([1.0, 0, 0, 0], device=x.device)
        quats = F.normalize(self.head_quat(x).view(B, N, K, 4) + q0, dim=-1)
        opacities = torch.sigmoid(self.head_opacity(x).view(B, N, K) + 2.0)
        rgb = torch.sigmoid(self.head_rgb(x).view(B, N, K, 3))
        means = posed_pts[:, :, None, :] + offset
        return dict(means=means.reshape(B, N * K, 3),
                    scales=scales.reshape(B, N * K, 3),
                    quats=quats.reshape(B, N * K, 4),
                    opacities=opacities.reshape(B, N * K),
                    rgb=rgb.reshape(B, N * K, 3))
