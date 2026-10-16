"""Metric-learning losses. All expect L2-normalised embeddings and are computed in float32."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero_like_graph(z):
    return z.sum() * 0.0


def supcon_loss(z, y, temperature=0.1):
    """Supervised contrastive loss (Khosla et al., 2020), single-view batch.

    For anchor i: -1/|P(i)| * sum_{p in P(i)} log( exp(z_i.z_p/t) / sum_{a != i} exp(z_i.z_a/t) ).
    Anchors without a positive in the batch are ignored.
    """
    z = z.float()
    n = z.shape[0]
    logits = (z @ z.t()) / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_mask = (y.view(-1, 1) == y.view(1, -1)) & ~self_mask
    n_pos = pos_mask.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return _zero_like_graph(z)
    log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    loss_i = -log_prob.sum(1)[valid] / n_pos[valid]
    return loss_i.mean()


def pairwise_euclidean(z):
    """Euclidean distance between L2-normalised rows: sqrt(2 - 2 cos)."""
    sim = (z @ z.t()).clamp(-1.0, 1.0)
    return torch.sqrt((2.0 - 2.0 * sim).clamp_min(1e-12))


def batch_hard_triplet_loss(z, y, margin=0.2, soft=False):
    """Batch-hard triplet loss (Hermans et al., 2017) on L2-normalised embeddings."""
    z = z.float()
    d = pairwise_euclidean(z)
    n = z.shape[0]
    same = y.view(-1, 1) == y.view(1, -1)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = same & ~eye
    neg = ~same
    valid = pos.any(1) & neg.any(1)
    if not valid.any():
        return _zero_like_graph(z)
    d_ap = torch.where(pos, d, torch.full_like(d, -1.0)).max(1).values
    d_an = torch.where(neg, d, torch.full_like(d, 1e6)).min(1).values
    diff = (d_ap - d_an)[valid]
    loss = F.softplus(diff) if soft else F.relu(diff + margin)
    return loss.mean()


def prototypical_loss(z_support, y_support, z_query, y_query, temperature=0.1):
    """Prototypical loss with cosine similarity (Snell et al., 2017, cosine variant).

    Labels are remapped to 0..n_way-1 using the classes present in the support set.
    Returns (loss, accuracy).
    """
    z_support, z_query = z_support.float(), z_query.float()
    classes, y_s = torch.unique(y_support, return_inverse=True)
    lookup = {int(c): i for i, c in enumerate(classes.tolist())}
    y_q = torch.tensor([lookup[int(c)] for c in y_query.tolist()], device=z_query.device)
    protos = torch.stack([z_support[y_s == i].mean(0) for i in range(len(classes))])
    protos = F.normalize(protos, dim=1)
    logits = (z_query @ protos.t()) / temperature
    loss = F.cross_entropy(logits, y_q)
    acc = (logits.argmax(1) == y_q).float().mean()
    return loss, acc


def batch_pairs(y, rng_generator=None, neg_per_pos=1):
    """All positive pairs (i<j, same class) + an equal number (x neg_per_pos) of random negatives."""
    n = y.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=y.device)
    same = y[iu[0]] == y[iu[1]]
    pos = iu[:, same]
    negs = iu[:, ~same]
    k = min(negs.shape[1], max(1, pos.shape[1] * neg_per_pos))
    perm = torch.randperm(negs.shape[1], generator=rng_generator, device="cpu")[:k].to(y.device)
    negs = negs[:, perm]
    idx = torch.cat([pos, negs], dim=1)
    target = torch.cat([torch.ones(pos.shape[1]), torch.zeros(negs.shape[1])]).to(y.device)
    return idx[0], idx[1], target


def pair_bce_loss(pair_head, z, y, neg_per_pos=1, generator=None):
    """Classic Siamese verification loss (Koch et al., 2015): BCE on sigmoid(w.|z_i - z_j| + b)."""
    z = z.float()
    i, j, t = batch_pairs(y, generator, neg_per_pos)
    if t.numel() == 0:
        return _zero_like_graph(z), torch.tensor(0.0)
    logits = pair_head((z[i] - z[j]).abs()).squeeze(1).float()
    loss = F.binary_cross_entropy_with_logits(logits, t)
    acc = ((logits > 0).float() == t).float().mean()
    return loss, acc


def contrastive_pair_loss(z, y, margin=0.5, neg_per_pos=1, generator=None):
    """Hadsell et al. (2006) contrastive loss on pair distances."""
    z = z.float()
    i, j, t = batch_pairs(y, generator, neg_per_pos)
    if t.numel() == 0:
        return _zero_like_graph(z)
    d = torch.sqrt(((z[i] - z[j]) ** 2).sum(1).clamp_min(1e-12))
    loss = t * d.pow(2) + (1 - t) * F.relu(margin - d).pow(2)
    return 0.5 * loss.mean()
