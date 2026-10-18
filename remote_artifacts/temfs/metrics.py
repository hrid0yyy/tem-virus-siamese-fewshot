"""Evaluation metrics.

Accuracy and macro precision/recall/F1 follow sklearn.classification_report (zero_division=0),
which is what the dataset authors used (per-patch, each class weighted equally).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def scores_from_confusion(cm, present_only=True):
    """Per-class and macro precision/recall/F1 + accuracy from a confusion matrix.

    present_only: macro averages over classes with support > 0 in y_true (sklearn behaviour when
    `labels` is not given and the class never occurs in y_true or y_pred).
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    support = cm.sum(1)
    predicted = cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    if present_only:
        mask = (support > 0) | (predicted > 0)
    else:
        mask = np.ones_like(support, dtype=bool)
    total = cm.sum()
    return {
        "accuracy": float(tp.sum() / total) if total > 0 else 0.0,
        "macro_precision": float(precision[mask].mean()) if mask.any() else 0.0,
        "macro_recall": float(recall[mask].mean()) if mask.any() else 0.0,
        "macro_f1": float(f1[mask].mean()) if mask.any() else 0.0,
        "balanced_accuracy": float(recall[support > 0].mean()) if (support > 0).any() else 0.0,
        "per_class": {"precision": precision.tolist(), "recall": recall.tolist(),
                      "f1": f1.tolist(), "support": support.astype(int).tolist()},
        "n": int(total),
    }


def classification_scores(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    out = scores_from_confusion(cm)
    out["confusion"] = cm.tolist()
    return out


def subset_accuracy(y_true, y_pred, class_subset):
    """Accuracy restricted to samples whose TRUE class is in class_subset (predictions over all classes)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = np.isin(y_true, list(class_subset))
    return float((y_true[m] == y_pred[m]).mean()) if m.any() else float("nan")


def subset_mean_class_accuracy(y_true, y_pred, class_subset):
    """Mean per-class recall over class_subset (robust to imbalance)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs = [float((y_pred[y_true == c] == c).mean()) for c in class_subset if (y_true == c).any()]
    return float(np.mean(accs)) if accs else float("nan")


def harmonic_mean(a, b):
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    if a + b == 0:
        return 0.0
    return float(2 * a * b / (a + b))


def aggregate_by_group(scores, groups, y_true):
    """Image-level decision: average the per-patch class scores of each micrograph.

    Returns (group_names, mean_scores[G, C], y_true_group[G]).
    """
    scores = np.asarray(scores)
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    names = sorted(idx)
    agg = np.stack([scores[idx[g]].mean(0) for g in names])
    yt = []
    for g in names:
        labels = set(int(y_true[i]) for i in idx[g])
        if len(labels) != 1:
            raise ValueError(f"group {g} has several labels {labels}")
        yt.append(labels.pop())
    return names, agg, np.array(yt)


def stratified_cluster_bootstrap(y_true, y_pred, groups, n_classes, n_boot=1000, seed=0,
                                 metrics=("accuracy", "macro_f1")):
    """95% CI by resampling micrographs with replacement inside each true class.

    Patches of one micrograph are correlated, so micrographs (not patches) are resampled.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    cls_groups = defaultdict(lambda: defaultdict(list))
    for i, (t, g) in enumerate(zip(y_true.tolist(), groups)):
        cls_groups[t][g].append(i)
    cls_groups = {c: list(d.values()) for c, d in cls_groups.items()}
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        sel = []
        for c, glist in cls_groups.items():
            picks = rng.integers(0, len(glist), size=len(glist))
            for p in picks:
                sel.extend(glist[p])
        sel = np.asarray(sel)
        s = scores_from_confusion(confusion_matrix(y_true[sel], y_pred[sel], n_classes))
        for m in metrics:
            vals[m].append(s[m])
    return {m: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5)),
                "std": float(np.std(v))} for m, v in vals.items()}


def mean_ci95(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "std": float("nan"), "n": 0}
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return {"mean": float(v.mean()), "ci95": 1.96 * std / math.sqrt(v.size), "std": std, "n": int(v.size)}


def auroc(scores_pos, scores_neg):
    """AUROC where higher score => positive. Mann-Whitney U with tie handling."""
    sp, sn = np.asarray(scores_pos, dtype=np.float64), np.asarray(scores_neg, dtype=np.float64)
    if sp.size == 0 or sn.size == 0:
        return float("nan")
    allv = np.concatenate([sp, sn])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[:sp.size].sum()
    return float((r_pos - sp.size * (sp.size + 1) / 2.0) / (sp.size * sn.size))


def fpr_at_tpr(scores_pos, scores_neg, tpr=0.95):
    """False-positive rate on negatives at the threshold that keeps `tpr` of positives."""
    sp, sn = np.asarray(scores_pos), np.asarray(scores_neg)
    if sp.size == 0 or sn.size == 0:
        return float("nan")
    thr = np.quantile(sp, 1.0 - tpr)
    return float((sn >= thr).mean())
