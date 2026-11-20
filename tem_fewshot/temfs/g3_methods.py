"""G3 decision-rule ladder on FROZEN embeddings (see prereg/G3_frozen_feature_study_2026-11-20.md).

Every classifier has the interface of temfs.gfsl_methods:
    fit(Xs, ys, n_classes, base_ids=..., novel_ids=..., **params) -> model
    predict(model, Xq) -> scores (n_query, n_classes), higher = more likely
Each rung changes one factor relative to the previous one. The paper's four classifiers (ncm, maha,
logreg = lr_std_bal, dc) are re-used unchanged from gfsl_methods, so the published grids are not touched.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .fewshot import l2n
from .gfsl_methods import METHODS as PAPER_METHODS
from .gfsl_methods import macro_f1

GAMMA_GRID = np.round(np.arange(-0.10, 0.2001, 0.005), 4)  # identical to analyze.py
C_GRID = (0.001, 0.01, 0.1, 1.0)  # identical to the paper's logistic-regression grid


def _protos(Xs, ys, n_classes):
    for c in range(n_classes):
        if not np.any(ys == c):
            raise ValueError("every class needs training samples")
    return np.stack([Xs[ys == c].mean(0) for c in range(n_classes)])


# R2: CL2N nearest class mean (centre on the BASE-class training mean, then L2)
def fit_cl2n(Xs, ys, n_classes, base_ids, **_):
    mu = Xs[np.isin(ys, base_ids)].mean(0)
    return {"mu": mu, "protos": l2n(_protos(l2n(Xs - mu), ys, n_classes))}


def predict_cl2n(m, Xq):
    return l2n(Xq - m["mu"]) @ m["protos"].T


# R3: standardised nearest class mean (z-score per feature on the fit data, then cosine)
def fit_stdncm(Xs, ys, n_classes, **_):
    sc = StandardScaler().fit(Xs)
    return {"sc": sc, "protos": l2n(_protos(sc.transform(Xs), ys, n_classes))}


def predict_stdncm(m, Xq):
    return l2n(m["sc"].transform(Xq)) @ m["protos"].T


# R4: LDA with a shared, Ledoit-Wolf-shrunk covariance and uniform class priors (as NCM, no prior bias)
def fit_lda(Xs, ys, n_classes, **_):
    _protos(Xs, ys, n_classes)  # presence check
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.full(n_classes, 1.0 / n_classes))
    lda.fit(Xs, ys)
    return {"lda": lda}


def predict_lda(m, Xq):
    return m["lda"].decision_function(Xq)


# R6a-d: logistic regression with/without standardisation and with/without class-balanced weights
def _make_lr(standardise, balanced):
    def fit(Xs, ys, n_classes, C=1.0, **_):
        sc = StandardScaler().fit(Xs) if standardise else None
        Z = sc.transform(Xs) if sc is not None else Xs
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced" if balanced else None)
        clf.fit(Z, ys)
        if list(clf.classes_) != list(range(n_classes)):
            raise ValueError("every class needs training samples")
        return {"sc": sc, "clf": clf}

    def predict(m, Xq):
        Z = m["sc"].transform(Xq) if m["sc"] is not None else Xq
        return m["clf"].decision_function(Z)

    return fit, predict


# R7: nearest class mean + calibrated stacking (subtract gamma from the base-class cosine scores)
def fit_stack(Xs, ys, n_classes, base_ids, gamma=0.0, **_):
    return {"protos": l2n(_protos(Xs, ys, n_classes)), "base_ids": np.asarray(base_ids), "gamma": float(gamma)}


def predict_stack(m, Xq):
    s = l2n(Xq) @ m["protos"].T
    s[:, m["base_ids"]] -= m["gamma"]
    return s


_lr = {k: _make_lr(*v) for k, v in {"lr_raw": (False, False), "lr_std": (True, False),
                                     "lr_bal": (False, True), "lr_std_bal": (True, True)}.items()}

# name -> (fit, predict, grid). Order = ladder order (R1 ... R7, then DC for the paper's pool).
LADDER = {
    "ncm": PAPER_METHODS["ncm"],                                     # R1
    "cl2n": (fit_cl2n, predict_cl2n, [{}]),                          # R2
    "stdncm": (fit_stdncm, predict_stdncm, [{}]),                    # R3
    "lda": (fit_lda, predict_lda, [{}]),                             # R4
    "maha": PAPER_METHODS["maha"],                                   # R5
    "lr_raw": (*_lr["lr_raw"], [{"C": c} for c in C_GRID]),          # R6a
    "lr_std": (*_lr["lr_std"], [{"C": c} for c in C_GRID]),          # R6b
    "lr_bal": (*_lr["lr_bal"], [{"C": c} for c in C_GRID]),          # R6c
    "lr_std_bal": PAPER_METHODS["logreg"],                           # R6d (= paper's logreg)
    "stack": (fit_stack, predict_stack, [{"gamma": float(g)} for g in GAMMA_GRID]),  # R7
    "dc": PAPER_METHODS["dc"],
}
PRIMARY_POOL = ("ncm", "maha", "lr_std_bal", "dc")  # the paper's four classifiers; selection rule fixed in the paper
KSHOT_METHODS = PRIMARY_POOL                         # Q4: per-K re-selection for the paper's pool


def choose(curve):
    """curve: list of (params, score) in grid order. Highest score; ties (1e-12) -> first setting in grid order,
    as gfsl_methods.select_on_validation (grids are ordered simplest first), except for calibrated stacking,
    where ties -> gamma closest to 0, as analysis.choose_gamma."""
    best = max(s for _, s in curve)
    tied = [p for p, s in curve if s >= best - 1e-12]
    if "gamma" in tied[0]:
        return min(tied, key=lambda p: abs(p["gamma"])), best
    return tied[0], best


def select_22class_val(name, Xtr, ytr, Xva, yva, n_classes, base_ids, novel_ids):
    """Primary rule: fit on train, score 22-class validation macro-F1 (rare validation patches included)."""
    fit, pred, grid = LADDER[name]
    curve = []
    for p in grid:
        m = fit(Xtr, ytr, n_classes, base_ids=base_ids, novel_ids=novel_ids, **p)
        curve.append((p, macro_f1(yva, pred(m, Xva).argmax(1), n_classes)))
    params, score = choose(curve)
    return params, score, curve


def make_pseudo_episodes(y, split, groups, base_ids, n_draws, n_pseudo, mic_counts, k=None, seed=0):
    """Pseudo-rare episodes on the BASE classes only (no rare-class label is ever touched).

    Each episode: n_pseudo base classes become pseudo-rare. If k is None, each keeps the training patches
    of m micrographs, m drawn from mic_counts; otherwise it keeps k patches from k different micrographs.
    The other base classes keep all training patches. Returns a list of dicts with local labels:
    pseudo-base classes first (ids 0..nb-1), then pseudo-rare.
    """
    rng = np.random.default_rng(seed)
    tr = split == "train"
    eps = []
    for _ in range(n_draws):
        pr = [int(c) for c in rng.choice(base_ids, size=n_pseudo, replace=False)]
        pb = [int(c) for c in base_ids if c not in pr]
        order = pb + pr
        idx = list(np.where(tr & np.isin(y, pb))[0])
        for c in pr:
            g2i = {}
            for i in np.where(tr & (y == c))[0]:
                g2i.setdefault(groups[i], []).append(int(i))
            mics = list(rng.permutation(sorted(g2i)))
            if k is None:
                m = int(min(rng.choice(mic_counts), len(mics)))
                for g in mics[:m]:
                    idx += g2i[g]
            else:
                for g in mics[:min(k, len(mics))]:
                    idx.append(int(rng.choice(g2i[g])))
        idx = np.array(sorted(idx))
        q = np.where((split == "validation") & np.isin(y, base_ids))[0]
        lut = {c: i for i, c in enumerate(order)}
        eps.append({"train_idx": idx, "query_idx": q, "lut": lut, "n_base": len(pb), "pseudo_rare": pr})
    return eps


def select_base_only(name, X, y, episodes):
    """Mean 14-class validation macro-F1 over pseudo-rare episodes; highest mean wins (ties -> simplest)."""
    fit, pred, grid = LADDER[name]
    sums = np.zeros(len(grid))
    for ep in episodes:
        lut = ep["lut"]
        ytr = np.array([lut[int(c)] for c in y[ep["train_idx"]]])
        yq = np.array([lut[int(c)] for c in y[ep["query_idx"]]])
        n = len(lut)
        b_ids, n_ids = list(range(ep["n_base"])), list(range(ep["n_base"], n))
        Xtr, Xq = X[ep["train_idx"]], X[ep["query_idx"]]
        for j, p in enumerate(grid):
            m = fit(Xtr, ytr, n, base_ids=b_ids, novel_ids=n_ids, **p)
            sums[j] += macro_f1(yq, pred(m, Xq).argmax(1), n)
    curve = [(p, float(s / len(episodes))) for p, s in zip(grid, sums)]
    params, score = choose(curve)
    return params, score, curve
