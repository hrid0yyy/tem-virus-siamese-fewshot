"""Nearest-prototype classification and all few-shot evaluation protocols.

Everything here works on pre-computed, L2-normalised embeddings held in an EmbeddingBank.
"""

from __future__ import annotations

import numpy as np

from .constants import BASE_CLASSES
from .metrics import (aggregate_by_group, auroc, classification_scores, fpr_at_tpr, harmonic_mean,
                      mean_ci95, stratified_cluster_bootstrap, subset_accuracy,
                      subset_mean_class_accuracy)
from .samplers import GroupIndex, draw_image_balanced, sample_episode


class EmbeddingBank:
    """Embeddings for a list of PatchRecords plus their metadata (split, class, micrograph)."""

    def __init__(self, records, embeddings: dict):
        self.records = list(records)
        self.emb = embeddings
        self.split = np.array([r.split for r in self.records])
        self.cls = np.array([r.cls for r in self.records])
        self.group = np.array([r.group for r in self.records])
        n = len(self.records)
        for k, v in embeddings.items():
            if len(v) != n:
                raise ValueError(f"embedding '{k}' has {len(v)} rows, expected {n}")

    def mask(self, splits, classes):
        splits = [splits] if isinstance(splits, str) else list(splits)
        return np.isin(self.split, splits) & np.isin(self.cls, list(classes))

    def labels(self, mask, classes):
        lut = {c: i for i, c in enumerate(classes)}
        return np.array([lut[c] for c in self.cls[mask]], dtype=np.int64)


def l2n(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def prototypes(emb, labels, n_classes):
    protos = np.zeros((n_classes, emb.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = labels == c
        if not m.any():
            raise ValueError(f"no support samples for class index {c}")
        protos[c] = emb[m].mean(0)
    return l2n(protos)


def ncm_scores(emb, protos):
    """Cosine similarity of each (normalised) embedding to each prototype."""
    return l2n(emb) @ protos.T


def _report(y_true, scores, groups, class_names, n_boot, seed, base_ids=None, novel_ids=None):
    """Per-patch + per-micrograph metrics with bootstrap CI and optional base/novel breakdown."""
    n_classes = len(class_names)
    y_pred = scores.argmax(1)
    res = classification_scores(y_true, y_pred, n_classes)
    res["class_names"] = list(class_names)
    if n_boot > 0:
        res["bootstrap_ci95"] = stratified_cluster_bootstrap(y_true, y_pred, groups, n_classes,
                                                             n_boot=n_boot, seed=seed)
    gnames, gscores, gtrue = aggregate_by_group(scores, groups, y_true)
    gres = classification_scores(gtrue, gscores.argmax(1), n_classes)
    res["image_level"] = {k: gres[k] for k in ("accuracy", "macro_f1", "macro_precision",
                                                "macro_recall", "n", "confusion")}
    if base_ids is not None and novel_ids is not None:
        for tag, yt, yp in (("patch", y_true, y_pred), ("image", gtrue, gscores.argmax(1))):
            b = subset_accuracy(yt, yp, base_ids)
            n = subset_accuracy(yt, yp, novel_ids)
            bm = subset_mean_class_accuracy(yt, yp, base_ids)
            nm = subset_mean_class_accuracy(yt, yp, novel_ids)
            res[f"gfsl_{tag}"] = {"base_acc": b, "novel_acc": n, "hm_acc": harmonic_mean(b, n),
                                  "base_mean_class_acc": bm, "novel_mean_class_acc": nm,
                                  "hm_mean_class_acc": harmonic_mean(bm, nm)}
    return res


# --------------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------------
def closed_set_ncm(bank, key, classes, support_splits=("train",), query_split="test",
                   n_boot=1000, seed=0):
    """Nearest-prototype classification over `classes` (prototypes from support split(s))."""
    sm = bank.mask(support_splits, classes)
    qm = bank.mask(query_split, classes)
    protos = prototypes(bank.emb[key][sm], bank.labels(sm, classes), len(classes))
    scores = ncm_scores(bank.emb[key][qm], protos)
    return _report(bank.labels(qm, classes), scores, bank.group[qm], classes, n_boot, seed,
                   **_base_novel_ids(classes))


def _base_novel_ids(classes):
    """base/novel index lists when the class list mixes both (enables the GFSL breakdown)."""
    base = [i for i, c in enumerate(classes) if c in BASE_CLASSES]
    novel = [i for i, c in enumerate(classes) if c not in BASE_CLASSES]
    return {"base_ids": base, "novel_ids": novel} if base and novel else {}


def closed_set_logits(bank, classes, query_split="test", n_boot=1000, seed=0):
    """Classifier evaluation from stored logits (label space == model training classes == classes)."""
    qm = bank.mask(query_split, classes)
    scores = bank.emb["logits"][qm]
    if scores.shape[1] != len(classes):
        raise ValueError("logit dimension does not match the class list")
    return _report(bank.labels(qm, classes), scores, bank.group[qm], classes, n_boot, seed,
                   **_base_novel_ids(classes))


def generalized_fewshot(bank, key, base, novel, support_splits=("train",), query_split="test",
                        n_boot=1000, seed=0):
    """GFSL: prototypes for base + novel classes, test on all of them (label space = base+novel)."""
    classes = list(base) + list(novel)
    sm = bank.mask(support_splits, classes)
    qm = bank.mask(query_split, classes)
    protos = prototypes(bank.emb[key][sm], bank.labels(sm, classes), len(classes))
    scores = ncm_scores(bank.emb[key][qm], protos)
    y = bank.labels(qm, classes)
    base_ids = list(range(len(base)))
    novel_ids = list(range(len(base), len(classes)))
    res = _report(y, scores, bank.group[qm], classes, n_boot, seed, base_ids, novel_ids)
    # Novel-only N-way (prototypes of novel classes only, novel queries only)
    nm = np.isin(y, novel_ids)
    novel_scores = scores[nm][:, novel_ids]
    res["novel_only_acc"] = float((novel_scores.argmax(1) + len(base) == y[nm]).mean())
    bmask = np.isin(y, base_ids)
    res["base_only_acc"] = float((scores[bmask][:, base_ids].argmax(1) == y[bmask]).mean())
    return res


def kshot_curve(bank, key, base, novel, ks=(1, 3, 5, 10), repeats=100, support_split="train",
                query_split="test", seed=0):
    """Novel classes get K support patches (from K different micrographs where possible);
    base classes keep all their training patches. Repeated `repeats` times per K."""
    classes = list(base) + list(novel)
    nb = len(base)
    emb = bank.emb[key]
    bm = bank.mask(support_split, base)
    base_protos = prototypes(emb[bm], bank.labels(bm, base), nb)
    qm = bank.mask(query_split, classes)
    q_emb, q_y = emb[qm], bank.labels(qm, classes)
    novel_ids = np.arange(nb, len(classes))
    # Novel support pools: micrograph -> indices (into the full bank)
    pools = {}
    for j, c in enumerate(novel):
        m = np.where(bank.mask(support_split, [c]))[0]
        g2i = {}
        for i in m:
            g2i.setdefault(bank.group[i], []).append(int(i))
        pools[j] = g2i
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        stats = {"base_acc": [], "novel_acc": [], "hm_acc": [], "novel_only_acc": [],
                 "novel_mean_class_acc": [], "support_actual_min": []}
        for _ in range(repeats):
            nprotos, actual = [], []
            for j in range(len(novel)):
                g2i = pools[j]
                order = list(rng.permutation(list(g2i.keys())))
                idx = draw_image_balanced(rng, g2i, order, k, allow_repeat=False)
                actual.append(len(idx))
                nprotos.append(emb[idx].mean(0))
            protos = np.concatenate([base_protos, l2n(np.stack(nprotos))], 0)
            scores = ncm_scores(q_emb, protos)
            pred = scores.argmax(1)
            b = subset_accuracy(q_y, pred, range(nb))
            n = subset_accuracy(q_y, pred, novel_ids)
            nmask = np.isin(q_y, novel_ids)
            n_only = float((scores[nmask][:, novel_ids].argmax(1) + nb == q_y[nmask]).mean())
            stats["base_acc"].append(b)
            stats["novel_acc"].append(n)
            stats["hm_acc"].append(harmonic_mean(b, n))
            stats["novel_only_acc"].append(n_only)
            stats["novel_mean_class_acc"].append(subset_mean_class_accuracy(q_y, pred, novel_ids))
            stats["support_actual_min"].append(min(actual))
        out[str(k)] = {m: mean_ci95(v) for m, v in stats.items()}
    return out


def episodic_eval(bank, key, classes, pool_splits=("train", "validation", "test"), n_way=None,
                  k_shot=5, n_query=15, episodes=600, seed=0, return_per_episode=False):
    """Image-disjoint N-way K-shot episodes on classes the encoder never trained on.

    Pools all `pool_splits` of `classes` (allowed only for classes unseen during training)
    and always separates support/query micrographs.
    """
    m = bank.mask(pool_splits, classes)
    idx_all = np.where(m)[0]
    y_all = bank.labels(m, classes)
    g_all = bank.group[m]
    gi = GroupIndex(y_all, list(g_all))
    emb = bank.emb[key][idx_all]
    n_way = len(classes) if n_way is None else n_way
    rng = np.random.default_rng(seed)
    accs, mca = [], []
    for _ in range(episodes):
        chosen = sorted(int(c) for c in rng.choice(gi.classes, size=n_way, replace=False))
        s_idx, s_cls, q_idx, q_cls = sample_episode(rng, gi, chosen, k_shot, n_query,
                                                    image_disjoint=True, allow_repeat_support=False,
                                                    allow_repeat_query=False)
        lut = {c: i for i, c in enumerate(chosen)}
        ys = np.array([lut[c] for c in s_cls])
        yq = np.array([lut[c] for c in q_cls])
        protos = prototypes(emb[s_idx], ys, n_way)
        pred = ncm_scores(emb[q_idx], protos).argmax(1)
        accs.append(float((pred == yq).mean()))
        mca.append(subset_mean_class_accuracy(yq, pred, range(n_way)))
    out = {"n_way": n_way, "k_shot": k_shot, "n_query": n_query, "episodes": episodes,
           "acc": mean_ci95(accs), "mean_class_acc": mean_ci95(mca)}
    if return_per_episode:
        out["per_episode_acc"] = accs
    return out


def open_set(bank, key, known, unknown, support_split="train", query_split="test"):
    """Unknown-virus detection: score = max cosine similarity to the known-class prototypes."""
    sm = bank.mask(support_split, known)
    protos = prototypes(bank.emb[key][sm], bank.labels(sm, known), len(known))
    km = bank.mask(query_split, known)
    um = bank.mask(query_split, unknown)
    s_known = ncm_scores(bank.emb[key][km], protos)
    s_unknown = ncm_scores(bank.emb[key][um], protos)
    k_score, u_score = s_known.max(1), s_unknown.max(1)
    res = {"patch": {"auroc": auroc(k_score, u_score), "fpr_at_95tpr": fpr_at_tpr(k_score, u_score),
                     "n_known": int(km.sum()), "n_unknown": int(um.sum())}}

    def group_max(scores, groups):
        out = {}
        for s, g in zip(scores, groups):
            out.setdefault(g, []).append(s)
        return np.array([np.mean(v, 0).max() for v in out.values()])

    gk, gu = group_max(s_known, bank.group[km]), group_max(s_unknown, bank.group[um])
    res["image"] = {"auroc": auroc(gk, gu), "fpr_at_95tpr": fpr_at_tpr(gk, gu),
                    "n_known": int(len(gk)), "n_unknown": int(len(gu))}
    return res
