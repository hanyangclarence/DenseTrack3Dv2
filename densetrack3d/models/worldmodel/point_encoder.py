"""Object-cloud state encoder (spec §4).

Turns the single present-frame cloud ``(B, P+N, 3)`` -- already globally centered and
isotropically scaled by the Dataset (§2.1) -- into ``M`` position-aware scene tokens
``S (B, M, C)`` that the backbone cross-attends into. Optionally also emits a per-query
Feature-Propagation feature (``emit_query_feat=True``), an ablation that is OFF by default
because a position-aware ``S`` + scene cross-attention already supplies each query's local
surface context (§4 point 4, §6).

Dependency-free PointNet++ set abstraction: clouds are tiny (P+N ~= 528), so the classic
CUDA ball-query kernel is unnecessary -- ``torch.cdist`` grouping is fast enough and dodges
the CUDA-extension build pain this env fights (memory ``env-cuda-build-setup``). The encoder
sits behind ``sa_cfg`` so a compiled backend can replace it later without touching callers.

``PosEnc`` is the SHARED positional code (§4/§6): one instance, owned by ``IntentModel``, is
passed to BOTH this encoder (for ``S``'s centroid positions) and ``init_query_tokens`` (for the
query seed), so cross-attention Q.K matches query <-> centroid positions with no learned
change-of-basis. Radii in ``sa_cfg`` live in the NORMALIZED coordinate space (a typical object
has ~0.75 mean radius from its centroid), NOT metres.
"""
import torch
import torch.nn as nn

from densetrack3d.models.embeddings import get_3d_embedding


class PosEnc(nn.Module):
    """Shared xyz -> C positional code: fixed sinusoid (get_3d_embedding) -> Linear.

    ``scale`` lifts unit-scale normalized coords into the sinusoid's useful frequency band
    (§11 gate 2b); ``emb_C`` sets the frequency count. get_3d_embedding(xyz, emb_C,
    cat_coords=False) returns ``(B, N, 3*emb_C)``.
    """

    def __init__(self, emb_C=128, C=384, scale=1.0):
        super().__init__()
        self.emb_C = emb_C
        self.scale = scale
        self.proj = nn.Linear(3 * emb_C, C)

    def forward(self, xyz):                                   # xyz (B, N, 3) -> (B, N, C)
        pe = get_3d_embedding(xyz * self.scale, self.emb_C, cat_coords=False)
        return self.proj(pe)


def _square_dists(a, b):
    """Pairwise squared Euclidean distances. a (B, Na, 3), b (B, Nb, 3) -> (B, Na, Nb)."""
    return torch.cdist(a, b, p=2.0) ** 2


def farthest_point_sample(xyz, npoint):
    """FPS over xyz (B, N, 3) -> centroid indices (B, npoint) (long)."""
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    # deterministic seed point (index 0); avoids Math.random-style nondeterminism
    far = torch.zeros(B, dtype=torch.long, device=device)
    batch = torch.arange(B, device=device)
    for i in range(npoint):
        idx[:, i] = far
        centroid = xyz[batch, far, :].unsqueeze(1)            # (B, 1, 3)
        d = ((xyz - centroid) ** 2).sum(-1)                   # (B, N)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx


def _gather_points(x, idx):
    """Gather along dim=1. x (B, N, C), idx (B, K) -> (B, K, C)."""
    B, _, C = x.shape
    K = idx.shape[1]
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(B, K, C))


def ball_query_group(radius, nsample, xyz, new_xyz):
    """PointNet++ ball query: group the ``nsample`` nearest points WITHIN ``radius``.

    xyz (B, N, 3) all points; new_xyz (B, S, 3) centroids -> neighbour indices (B, S, nsample).
    NOT kNN: points beyond ``radius`` are never grouped (kNN would stretch a group across a
    removed patch, breaking occlusion-robustness); under-full balls pad with the centroid's
    nearest in-ball point. FPS centroids are a subset of ``xyz``, so every ball is non-empty
    (the centroid sits in it at distance 0). Radii are in NORMALIZED coords (§4).
    """
    dd = radius * radius
    dists = _square_dists(new_xyz, xyz)                       # (B, S, N)
    in_ball = dists <= dd
    # push out-of-ball points behind all in-ball points, preserving nearest-first order
    ranked = torch.where(in_ball, dists, dists + 1e9)
    knn_idx = ranked.argsort(dim=-1)[:, :, :nsample]          # (B, S, nsample)
    # pad under-full balls: replace any out-of-ball pick with the nearest in-ball point (col 0)
    picked_in_ball = torch.gather(in_ball, 2, knn_idx)
    first = knn_idx[:, :, :1].expand_as(knn_idx)
    return torch.where(picked_in_ball, knn_idx, first)


class SetAbstraction(nn.Module):
    """One PointNet++ SA level: FPS -> ball/kNN group -> shared MLP -> max-pool."""

    def __init__(self, npoint, radius, nsample, in_c, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        layers = []
        last = in_c + 3                                       # +3 for relative xyz
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feat):
        # xyz (B, N, 3); feat (B, N, in_c) or None
        cidx = farthest_point_sample(xyz, self.npoint)        # (B, npoint)
        new_xyz = _gather_points(xyz, cidx)                   # (B, npoint, 3)
        gidx = ball_query_group(self.radius, self.nsample, xyz, new_xyz)   # (B, npoint, nsample)

        B, S, K = gidx.shape
        flat = gidx.reshape(B, S * K)
        grouped_xyz = _gather_points(xyz, flat).reshape(B, S, K, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)      # relative to centroid
        if feat is not None:
            grouped_feat = _gather_points(feat, flat).reshape(B, S, K, -1)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)
        else:
            grouped = grouped_xyz
        grouped = grouped.permute(0, 3, 1, 2)                 # (B, C_in, S, K)
        new_feat = self.mlp(grouped).max(dim=-1)[0]           # (B, C_out, S)
        new_feat = new_feat.permute(0, 2, 1)                  # (B, S, C_out)
        return new_xyz, new_feat


class FeaturePropagation(nn.Module):
    """3-NN inverse-distance interpolation of centroid features back to query xyz."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(in_c, out_c), nn.ReLU(inplace=True))

    def forward(self, query_xyz, centroid_xyz, centroid_feat):
        # query_xyz (B, Nq, 3); centroid_xyz (B, M, 3); centroid_feat (B, M, c)
        dists = _square_dists(query_xyz, centroid_xyz)        # (B, Nq, M)
        d, idx = dists.sort(dim=-1)
        d, idx = d[:, :, :3], idx[:, :, :3]                   # 3 nearest
        w = 1.0 / (d.sqrt() + 1e-8)
        w = w / w.sum(dim=-1, keepdim=True)                   # (B, Nq, 3)
        B, Nq, _ = idx.shape
        c = centroid_feat.shape[-1]
        gathered = _gather_points(centroid_feat, idx.reshape(B, Nq * 3)).reshape(B, Nq, 3, c)
        interp = (gathered * w.unsqueeze(-1)).sum(dim=2)      # (B, Nq, c)
        return self.lin(interp)


class PointEncoder(nn.Module):
    """Cloud (B, P+N, 3) normalized -> scene tokens S (B, M, C) [+ optional query_feat]."""

    def __init__(self, pos_enc, out_dim=384,
                 sa_cfg=((256, 0.2, 32, (32, 32, 64)),
                         (128, 0.4, 32, (64, 64, 128)),
                         (64, 0.8, 32, (128, 128, 256))),
                 emit_query_feat=False):
        super().__init__()
        self.pos_enc = pos_enc                                # SHARED with query init (§4/§6)
        self.emit_query_feat = emit_query_feat
        self.sa_layers = nn.ModuleList()
        in_c = 0
        last_c = 0
        for npoint, radius, nsample, mlp in sa_cfg:
            self.sa_layers.append(SetAbstraction(npoint, radius, nsample, in_c, mlp))
            in_c = mlp[-1]
            last_c = mlp[-1]
        self.proj = nn.Linear(last_c, out_dim)
        if emit_query_feat:
            self.fp = FeaturePropagation(last_c, last_c)
            self.q_proj = nn.Linear(last_c, out_dim)

    def forward(self, cloud, x0_norm):
        # cloud (B, P+N, 3) normalized; x0_norm (B, N, 3) == cloud[:, -N:]
        xyz, feat = cloud, None
        for sa in self.sa_layers:
            xyz, feat = sa(xyz, feat)
        cxyz, cfeat = xyz, feat                               # (B, M, 3), (B, M, last_c)
        S = self.proj(cfeat) + self.pos_enc(cxyz)             # (B, M, C) position-aware
        query_feat = None
        if self.emit_query_feat:
            query_feat = self.q_proj(self.fp(x0_norm, cxyz, cfeat))   # (B, N, C)
        return S, query_feat
