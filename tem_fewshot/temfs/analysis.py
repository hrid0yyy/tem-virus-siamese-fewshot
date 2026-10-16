"""Inference-time analyses that need no retraining (all tuning on the validation split).

* CL2N feature transform (SimpleShot, Wang et al. 2019): subtract the mean feature of the base-class
  TRAIN patches, then L2-normalise. Uses no validation/test/novel data.
* Calibrated stacking (Chao et al. 2016, generalised zero-/few-shot learning): subtract a constant
  gamma from the base-class scores to correct the bias towards seen classes. gamma is chosen on the
  VALIDATION split (prototypes from train, queries from validation) and then applied once to test.
* Paired comparison of two models on identical test patches: stratified micrograph (cluster)
  bootstrap of the accuracy / macro-F1 difference and an exact McNemar test.
"""

from __future__ import annotations

import math

import numpy as np

from .fewshot import l2n, ncm_scores, prototypes
from .metrics import scores_from_confusion, confusion_matrix


def cl2n_transform(emb, mean):
    return l2n(emb - mean[None, :])


def base_train_mean(bank, key, base):
    m = bank.mask("train", base)
    return bank.emb[key][m].mean(0)


def gfsl_scores(bank, emb, base, novel, query_split, support_splits=("train",)):
    """Cosine scores of query patches against base+novel prototypes. Returns (scores, y, groups, mask)."""
    classes = list(base) + list(novel)
    sm = bank.mask(support_splits, classes)
    qm = bank.mask(query_split, classes)
    protos = prototypes(emb[sm], bank.labels(sm, classes), len(classes))
    return ncm_scores(emb[qm], protos), bank.labels(qm, classes), bank.group[qm], qm


def stacked_predict(scores, n_base, gamma):
    s = scores.copy()
    s[:, :n_base] -= gamma
    return s.argmax(1)


def choose_gamma(scores_val, y_val, n_base, n_classes, grid, criterion="macro_f1"):
    """Pick gamma on validation; ties -> smallest |gamma| (closest to no calibration)."""
    best, best_g, curve = -1.0, 0.0, []
    for g in grid:
        s = scores_from_confusion(confusion_matrix(y_val, stacked_predict(scores_val, n_base, g), n_classes))
        v = s[criterion]
        curve.append((float(g), float(v)))
        if v > best + 1e-12 or (abs(v - best) <= 1e-12 and abs(g) < abs(best_g)):
            best, best_g = v, float(g)
    return best_g, best, curve


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test on paired correctness vectors. Returns (b, c, p)."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int((a & ~b_).sum())  # A right, B wrong
    c = int((~a & b_).sum())  # A wrong, B right
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b, c, float(min(1.0, 2 * p))


def paired_cluster_bootstrap(y_true, pred_a, pred_b, groups, n_classes, n_boot=2000, seed=0):
    """CI of (A - B) for accuracy and macro-F1, resampling micrographs within each true class."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    by_cls = {}
    for i, (t, g) in enumerate(zip(y_true.tolist(), groups)):
        by_cls.setdefault(t, {}).setdefault(g, []).append(i)
    by_cls = {c: list(d.values()) for c, d in by_cls.items()}

    def metrics(idx):
        sa = scores_from_confusion(confusion_matrix(y_true[idx], pred_a[idx], n_classes))
        sb = scores_from_confusion(confusion_matrix(y_true[idx], pred_b[idx], n_classes))
        return sa["accuracy"] - sb["accuracy"], sa["macro_f1"] - sb["macro_f1"]

    full = metrics(np.arange(len(y_true)))
    da, df = [], []
    for _ in range(n_boot):
        sel = []
        for glist in by_cls.values():
            for p in rng.integers(0, len(glist), size=len(glist)):
                sel.extend(glist[p])
        a, f = metrics(np.asarray(sel))
        da.append(a)
        df.append(f)
    da, df = np.array(da), np.array(df)
    return {
        "acc_diff": float(full[0]), "acc_diff_ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
        "acc_diff_p_le0": float((da <= 0).mean()),
        "macro_f1_diff": float(full[1]), "macro_f1_diff_ci95": [float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))],
        "macro_f1_diff_p_le0": float((df <= 0).mean()), "n_boot": n_boot,
    }
