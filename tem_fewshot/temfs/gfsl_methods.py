"""Classifiers on FROZEN embeddings for generalised few-shot (14 base + 8 rare classes).

All hyper-parameters are selected on the VALIDATION split (prototypes / training data from the
official TRAIN split, queries from VALIDATION); the selected setting is then applied once to TEST.

Methods
  ncm   - nearest class mean, cosine (the current default).
  maha  - class-specific Mahalanobis distance with shrinkage, in the style of Simple CNAPS
          (Bateni et al., CVPR 2020): Q_c = l_c*S_c + (1-l_c)*S_all + beta*s*I, l_c = n_c/(n_c+1),
          computed in a PCA space fitted on the BASE-class training features only.
  logreg- multinomial logistic regression on the frozen features (Tian et al., ECCV 2020),
          class-balanced weights.
  dc    - distribution calibration (Yang et al., ICLR 2021): Tukey power transform, then for each
          rare-class support feature borrow mean/covariance from its k nearest BASE classes, sample
          synthetic features, and train the logistic regression on real + synthetic features.

Only the linear classifier / statistics are fitted; the encoder is never retrained.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .fewshot import l2n
from .metrics import confusion_matrix, scores_from_confusion


# ----------------------------------------------------------------------------- helpers
def macro_f1(y, pred, n):
    return scores_from_confusion(confusion_matrix(y, pred, n))["macro_f1"]


def _ncomp(pca_dim, X):
    """PCA size: requested dimension, capped by the number of features and samples."""
    return int(min(pca_dim, X.shape[1], X.shape[0] - 1))


def tukey(x, lam=0.5):
    """Tukey ladder-of-powers transform; features must be non-negative (ReLU+GAP features are)."""
    x = np.clip(x, 0, None)
    return np.power(x, lam) if lam != 0 else np.log(x + 1e-6)


# ----------------------------------------------------------------------------- NCM
def fit_ncm(Xs, ys, n_classes, **_):
    protos = np.stack([Xs[ys == c].mean(0) for c in range(n_classes)])
    return {"protos": l2n(protos)}


def predict_ncm(model, Xq):
    return l2n(Xq) @ model["protos"].T


# ----------------------------------------------------------------------------- Mahalanobis
def fit_maha(Xs, ys, n_classes, base_ids, pca_dim=128, beta=1.0, **_):
    base_mask = np.isin(ys, base_ids)
    pca = PCA(n_components=_ncomp(pca_dim, Xs[base_mask]), random_state=0).fit(Xs[base_mask])
    Z = pca.transform(Xs)
    d = Z.shape[1]
    S_all = np.cov(Z, rowvar=False)
    s = np.trace(S_all) / d  # scale of the identity term
    means, invs = [], []
    for c in range(n_classes):
        Zc = Z[ys == c]
        n = len(Zc)
        Sc = np.cov(Zc, rowvar=False) if n > 1 else np.zeros((d, d))
        lam = n / (n + 1.0)
        Q = lam * Sc + (1 - lam) * S_all + beta * s * np.eye(d)
        means.append(Zc.mean(0))
        invs.append(np.linalg.inv(Q))
    return {"pca": pca, "means": np.stack(means), "invs": invs}


def predict_maha(model, Xq):
    Z = model["pca"].transform(Xq)
    out = np.empty((len(Z), len(model["means"])))
    for c, (m, Qi) in enumerate(zip(model["means"], model["invs"])):
        D = Z - m
        out[:, c] = -np.einsum("ij,jk,ik->i", D, Qi, D)  # higher = closer
    return out


# ----------------------------------------------------------------------------- logistic regression
def fit_logreg(Xs, ys, n_classes, C=1.0, sample_weight=None, **_):
    sc = StandardScaler().fit(Xs)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(Xs), ys, sample_weight=sample_weight)
    if list(clf.classes_) != list(range(n_classes)):
        raise ValueError("every class needs training samples")
    return {"scaler": sc, "clf": clf}


def predict_logreg(model, Xq):
    return model["clf"].decision_function(model["scaler"].transform(Xq))


# ----------------------------------------------------------------------------- distribution calibration
def fit_dc(Xs, ys, n_classes, base_ids, novel_ids, k=2, alpha=0.2, n_gen=500, pca_dim=128, C=1.0,
           seed=0, **_):
    rng = np.random.default_rng(seed)
    T = tukey(Xs)
    base_mask = np.isin(ys, base_ids)
    pca = PCA(n_components=_ncomp(pca_dim, T[base_mask]), random_state=0).fit(T[base_mask])
    Z = pca.transform(T)
    d = Z.shape[1]
    b_means = np.stack([Z[ys == b].mean(0) for b in base_ids])
    b_covs = np.stack([np.cov(Z[ys == b], rowvar=False) for b in base_ids])
    gen_X, gen_y = [], []
    for c in novel_ids:
        Zc = Z[ys == c]
        per = max(1, int(np.ceil(n_gen / len(Zc))))  # ~n_gen synthetic features per rare class
        for z in Zc:
            dist = np.linalg.norm(b_means - z, axis=1)
            nn = np.argsort(dist)[:k]
            mu = (b_means[nn].sum(0) + z) / (k + 1)
            cov = b_covs[nn].mean(0) + alpha * np.eye(d)
            gen_X.append(rng.multivariate_normal(mu, cov, size=per, method="cholesky"))
            gen_y += [c] * per
    Xtr = np.concatenate([Z] + gen_X) if gen_X else Z
    ytr = np.concatenate([ys, np.array(gen_y, dtype=ys.dtype)]) if gen_y else ys
    lr = fit_logreg(Xtr, ytr, n_classes, C=C)
    return {"tukey": True, "pca": pca, "lr": lr}


def predict_dc(model, Xq):
    return predict_logreg(model["lr"], model["pca"].transform(tukey(Xq)))


METHODS = {
    "ncm": (fit_ncm, predict_ncm, [{}]),
    "maha": (fit_maha, predict_maha, [{"pca_dim": p, "beta": b} for p in (64, 128, 256) for b in (0.1, 1.0, 10.0)]),
    "logreg": (fit_logreg, predict_logreg, [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)]),
    "dc": (fit_dc, predict_dc, [{"k": k, "alpha": a, "C": c, "pca_dim": 128}
                                for k in (1, 2) for a in (0.1, 0.3) for c in (0.01, 0.1)]),
}


def select_on_validation(name, Xtr, ytr, Xva, yva, n_classes, base_ids, novel_ids, log=print):
    """Grid-search one method on validation macro-F1; ties -> first (simplest) setting."""
    fit, pred, grid = METHODS[name]
    best, best_p, curve = -1.0, None, []
    for params in grid:
        m = fit(Xtr, ytr, n_classes, base_ids=base_ids, novel_ids=novel_ids, **params)
        f = macro_f1(yva, pred(m, Xva).argmax(1), n_classes)
        curve.append((params, f))
        if f > best + 1e-12:
            best, best_p = f, params
    log(f"  [{name}] validation macro-F1 by setting: " +
        ", ".join(f"{p}:{f:.4f}" for p, f in curve) + f"  -> selected {best_p}")
    return best_p, best, curve


def pair_separability(X_tr, y_tr, X_va, y_va, a, b):
    """Binary logistic regression trained on TRAIN patches of classes a,b; AUC/accuracy on VALIDATION."""
    from sklearn.metrics import roc_auc_score
    mtr, mva = np.isin(y_tr, [a, b]), np.isin(y_va, [a, b])
    sc = StandardScaler().fit(X_tr[mtr])
    clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(X_tr[mtr]), y_tr[mtr] == a)
    s = clf.decision_function(sc.transform(X_va[mva]))
    yt = y_va[mva] == a
    # also how a plain 2-class nearest-mean rule does
    ma, mb = l2n(X_tr[y_tr == a].mean(0, keepdims=True)), l2n(X_tr[y_tr == b].mean(0, keepdims=True))
    ncm_pred = (l2n(X_va[mva]) @ ma.T > l2n(X_va[mva]) @ mb.T).ravel()
    return {"auc_logreg": float(roc_auc_score(yt, s)), "acc_logreg": float(((s > 0) == yt).mean()),
            "acc_ncm": float((ncm_pred == yt).mean()), "n_val_a": int(yt.sum()), "n_val_b": int((~yt).sum())}
