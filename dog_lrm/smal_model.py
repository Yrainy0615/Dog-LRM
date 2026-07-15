"""SMAL body model wrapper (BARC SMBLD dog) for Dog-LRM.

Thin wrapper around BARC's SMAL layer exposing canonical (rest-pose) and posed vertices.
v1 anchors Gaussians at posed vertices, so we don't need the per-vertex LBS transform yet.
"""
import json
import os
import sys

import torch
import torch.nn as nn


def _load_barc_smal(device):
    barc_src = os.path.join(os.path.dirname(__file__), "..", "third_party",
                            "barc_release", "src")
    sys.path.insert(0, os.path.abspath(barc_src))
    from smal_pytorch.smal_model.smal_torch_new import SMAL
    return SMAL().to(device)


def build_subdiv(faces, n_subdiv, device):
    """Sparse [Vsub, V] operator: keeps old verts, adds edge-midpoints (mean of
    endpoints). Same operator applied to vertex positions and to features."""
    faces = torch.as_tensor(faces).long().cpu()
    V = int(faces.max()) + 1
    M = None
    for _ in range(n_subdiv):
        e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
        e = torch.sort(e, dim=1).values
        e = torch.unique(e, dim=0)                      # [E,2] unique undirected edges
        E = e.shape[0]
        i = torch.cat([torch.arange(V), V + torch.arange(E), V + torch.arange(E)])
        j = torch.cat([torch.arange(V), e[:, 0], e[:, 1]])
        val = torch.cat([torch.ones(V), torch.full((E,), 0.5), torch.full((E,), 0.5)])
        level = torch.sparse_coo_tensor(torch.stack([i, j]), val, (V + E, V)).coalesce()
        M = level if M is None else torch.sparse.mm(level, M.to_dense()).to_sparse()
        # advance faces to next level (4-way split) to chain another subdivision
        eidx = {tuple(t.tolist()): V + k for k, t in enumerate(e)}
        def mid(a, b):
            return eidx[tuple(sorted((int(a), int(b))))]
        nf = []
        for f in faces:
            a, b, c = f.tolist()
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        faces = torch.tensor(nf)
        V = V + E
    return M.to(device)


def build_surface_sampler(faces, verts, n_samples, vert_weight=None, seed=0, device="cuda"):
    """Sparse [N, V] barycentric operator putting N random points on the mesh surface,
    sampling probability ∝ face area × mean(vert_weight over corners). Same linear
    contract as build_subdiv, so it applies to positions and features alike. Breaks
    the regular subdivision lattice (anti moiré/dot-pattern); fixed seed → identical
    anchors across ranks and at inference."""
    f = torch.as_tensor(faces).long().cpu()
    v = torch.as_tensor(verts).float().cpu()
    tri = v[f]                                                      # [F,3,3]
    w = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=-1) / 2
    if vert_weight is not None:
        w = w * torch.as_tensor(vert_weight).float().cpu()[f].mean(1)
    g = torch.Generator().manual_seed(seed)
    fi = torch.multinomial(w / w.sum(), n_samples, replacement=True, generator=g)
    r = torch.rand(n_samples, 2, generator=g)
    r1 = r[:, 0].sqrt()                                             # uniform in-triangle
    b = torch.stack([1 - r1, r1 * (1 - r[:, 1]), r1 * r[:, 1]], 1)  # [N,3] barycentric
    i = torch.arange(n_samples).repeat_interleave(3)
    j = f[fi].reshape(-1)
    M = torch.sparse_coo_tensor(torch.stack([i, j]), b.reshape(-1),
                                (n_samples, v.shape[0])).coalesce()
    return M.to(device), fi.to(device)


def subdivided_faces(faces, n_subdiv):
    """Face list matching build_subdiv's vertex layout (old verts, then edge
    midpoints in sorted-unique edge order). Returns [F*4^n, 3] long."""
    f = torch.as_tensor(faces).long().cpu()
    V = int(f.max()) + 1
    for _ in range(n_subdiv):
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


class SMALModel(nn.Module):
    def __init__(self, device="cuda", n_subdiv=1):
        super().__init__()
        self.smal = _load_barc_smal(device)
        self.num_betas = self.smal.num_betas
        self.device = device
        self.subdiv_M = self._build_subdiv(n_subdiv, device) if n_subdiv > 0 else None

    @property
    def faces(self):
        return self.smal.faces

    def _build_subdiv(self, n_subdiv, device):
        return build_subdiv(self.smal.faces, n_subdiv, device)

    def subdivide(self, x):
        """x [B,N,C] -> [B,Vsub,C] via the precomputed operator."""
        if self.subdiv_M is None:
            return x
        return torch.stack([torch.sparse.mm(self.subdiv_M, x[b]) for b in range(x.shape[0])])

    def _verts(self, betas, limbs, theta):
        B = betas.shape[0]
        z = torch.zeros(B, 3, device=betas.device)
        V, _, _ = self.smal(beta=betas, betas_limbs=limbs, theta=theta, trans=z, get_skin=True)
        return V

    def canonical_verts(self, betas, limbs):
        """Rest-pose, shape-deformed mesh (Gaussian-anchor scaffold)."""
        B = betas.shape[0]
        return self._verts(betas, limbs, torch.zeros(B, 35, 3, device=betas.device))

    def posed_verts(self, betas, limbs, theta, trans, scale):
        """World-frame posed mesh: scale * SMAL(theta) + trans (matches fit_smal)."""
        B = betas.shape[0]
        V = self._verts(betas, limbs, theta)
        return scale.view(B, 1, 1) * V + trans.view(B, 1, 3)

    def posed_joints(self, betas, limbs, theta, trans, scale):
        """World-frame regressed 35 joints, same convention as posed_verts (for IK)."""
        B = betas.shape[0]
        z = torch.zeros(B, 3, device=betas.device)
        _, J, _ = self.smal(beta=betas, betas_limbs=limbs, theta=theta, trans=z,
                            get_skin=True, keyp_conf="red")
        return scale.view(B, 1, 1) * J[:, :35] + trans.view(B, 1, 3)


def load_pseudo_gt(scene_dir, out_subdir, num_betas, device):
    """Read fit_smal's smal_params.json -> batched tensors (theta = [global; body])."""
    d = json.load(open(os.path.join(scene_dir, out_subdir, "smal_params.json")))
    betas = torch.zeros(1, num_betas, device=device)
    b = torch.tensor(d["betas"], device=device).float()
    betas[0, :b.numel()] = b[:num_betas]
    limbs = torch.tensor(d["betas_limbs"], device=device).float()[None]
    go = torch.tensor(d["global_orient"], device=device).float().view(1, 1, 3)
    bp = torch.tensor(d["body_pose"], device=device).float().view(1, 34, 3)
    theta = torch.cat([go, bp], dim=1)  # [1,35,3]
    trans = torch.tensor(d["trans"], device=device).float()[None]
    scale = torch.tensor([d["scale"]], device=device).float()
    return dict(betas=betas, limbs=limbs, theta=theta, trans=trans, scale=scale)
