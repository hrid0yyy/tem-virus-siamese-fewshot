"""G3 frozen-feature study, pre-registered in prereg/G3_frozen_feature_study_2026-11-20.md.

Per feature set (trained checkpoint or frozen ImageNet backbone):
  0. embed all 10,604 patches once (backbone 'feat', with and without D4 TTA) and cache them (reused by G6-G10);
  Q1/Q2  every ladder classifier, hyper-parameters chosen by 22-class validation macro-F1 (primary rule), test once;
  Q3     the same classifiers with hyper-parameters chosen on pseudo-rare BASE-class episodes only, test once;
  Q4     per-K re-selection (pseudo-rare episodes with K patches) vs the primary setting, same support draws;
  Q5     separability of the confusable pairs: leave-one-micrograph-out on train+validation (primary) and
         train -> test (descriptive), AUROC with micrograph-cluster bootstrap CIs.

    set TEMFS_FP32=1
    python g3_study.py --data-root data --out runs/g3 --features runs/A/best.pt ... imagenet:densenet201
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import summarise  # noqa: E402
from gfsl_improve import PAIRS  # noqa: E402
from temfs.constants import BASE_CLASSES, NOVEL_CLASSES, SPLITS  # noqa: E402
from temfs.data import select_records  # noqa: E402
from temfs.g3_methods import (KSHOT_METHODS, LADDER, PRIMARY_POOL, choose, make_pseudo_episodes,  # noqa: E402
                              select_22class_val)
from temfs.gfsl_methods import macro_f1  # noqa: E402
from temfs.metrics import harmonic_mean, mean_ci95, subset_accuracy, subset_mean_class_accuracy  # noqa: E402
from temfs.samplers import draw_image_balanced  # noqa: E402
from temfs.utils import Logger, environment_info, get_device, inference_fp32, save_json, seed_everything  # noqa: E402

RARE_TRAIN_MICS = (5, 8, 13, 19, 20, 20, 22, 25)  # training micrographs of the 8 rare classes (official split)
CLASSES = list(BASE_CLASSES) + list(NOVEL_CLASSES)
BASE_IDS, NOVEL_IDS = list(range(len(BASE_CLASSES))), list(range(len(BASE_CLASSES), len(CLASSES)))


# ----------------------------------------------------------------------------- embeddings
def feature_name(spec):
    return spec.split(":", 1)[1] + "_imagenet_frozen" if spec.startswith("imagenet:") else Path(spec).parent.name


def embed_cached(spec, out_dir, data_root, device, log, state):
    f = out_dir / f"emb_{feature_name(spec)}.npz"
    if f.exists():
        log(f"  cached embeddings: {f}")
        return dict(np.load(f, allow_pickle=False))
    from temfs.pipeline import build_bank, frozen_imagenet_model, load_checkpoint, prepare_data
    if "records" not in state:
        state["records"], state["store"] = prepare_data(data_root, log)
    if spec.startswith("imagenet:"):
        model, base = frozen_imagenet_model(spec.split(":", 1)[1], device), list(BASE_CLASSES)
    else:
        model, cfg, _ = load_checkpoint(spec, device)
        base = list(cfg["train_classes"])
    if base != list(BASE_CLASSES):
        raise ValueError(f"{spec}: training classes {base} differ from the canonical base classes")
    recs = select_records(state["records"], SPLITS, CLASSES)
    t0 = time.time()
    b1 = build_bank(model, recs, state["store"], device, tta=True, keys=("feat",))
    b0 = build_bank(model, recs, state["store"], device, tta=False, keys=("feat",))
    del model
    d = {"X_tta": b1.emb["feat"].astype(np.float32), "X_notta": b0.emb["feat"].astype(np.float32),
         "cls": b1.cls.astype(str), "split": b1.split.astype(str), "group": b1.group.astype(str),
         "path": np.array([r.path for r in recs], dtype=str), "fp32": np.array(inference_fp32())}
    np.savez_compressed(f, **d)
    log(f"  embedded {len(recs)} patches (TTA and no-TTA) in {time.time() - t0:.0f}s -> {f}")
    return d


# ----------------------------------------------------------------------------- selection helpers (parallel)
def _episode_scores(name, X, y, ep):
    fit, pred, grid = LADDER[name]
    lut = ep["lut"]
    ytr = np.array([lut[int(c)] for c in y[ep["train_idx"]]])
    yq = np.array([lut[int(c)] for c in y[ep["query_idx"]]])
    n = len(lut)
    b_ids, n_ids = list(range(ep["n_base"])), list(range(ep["n_base"], n))
    Xtr, Xq = X[ep["train_idx"]], X[ep["query_idx"]]
    return [macro_f1(yq, pred(fit(Xtr, ytr, n, base_ids=b_ids, novel_ids=n_ids, **p), Xq).argmax(1), n)
            for p in grid]


def select_base_only_par(name, X, y, episodes, n_jobs):
    rows = Parallel(n_jobs=n_jobs)(delayed(_episode_scores)(name, X, y, ep) for ep in episodes)
    curve = [(p, float(np.mean([r[j] for r in rows]))) for j, p in enumerate(LADDER[name][2])]
    params, score = choose(curve)
    return params, score, curve


def kshot_draws(y, split, groups, k, repeats, seed):
    """Support draws exactly as gfsl_improve.kshot_eval (rare classes: k patches from k micrographs)."""
    rng = np.random.default_rng(seed)
    tr = split == "train"
    base_idx = np.where(tr & np.isin(y, BASE_IDS))[0]
    pools = {}
    for c in NOVEL_IDS:
        g2i = {}
        for i in np.where(tr & (y == c))[0]:
            g2i.setdefault(groups[i], []).append(int(i))
        pools[c] = g2i
    draws = []
    for _ in range(repeats):
        idx = list(base_idx)
        for c in NOVEL_IDS:
            order = list(rng.permutation(list(pools[c].keys())))
            idx += draw_image_balanced(rng, pools[c], order, k, allow_repeat=False)
        draws.append(np.array(idx))
    return draws


def _kshot_one(name, params, X, y, te, idx):
    fit, pred, _ = LADDER[name]
    m = fit(X[idx], y[idx], len(CLASSES), base_ids=BASE_IDS, novel_ids=NOVEL_IDS, **params)
    p = pred(m, X[te]).argmax(1)
    yt = y[te]
    b, n = subset_accuracy(yt, p, BASE_IDS), subset_mean_class_accuracy(yt, p, NOVEL_IDS)
    return {"novel_mean_class_acc": n, "base_acc": b, "hm": harmonic_mean(b, n),
            "macro_f1": macro_f1(yt, p, len(CLASSES))}


def kshot_eval_par(name, params, X, y, te, draws, n_jobs):
    rows = Parallel(n_jobs=n_jobs)(delayed(_kshot_one)(name, params, X, y, te, idx) for idx in draws)
    keys = rows[0].keys()
    return {k: mean_ci95([r[k] for r in rows]) for k in keys} | {"per_draw": {k: [float(r[k]) for r in rows]
                                                                            for k in keys}}


# ----------------------------------------------------------------------------- separability
def _pair_lr(Xa, ya):
    sc = StandardScaler().fit(Xa)
    clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(Xa), ya)
    return sc, clf


def cluster_auc_ci(scores, yt, groups, n_boot, seed):
    """95 % CI of AUROC, resampling micrographs with replacement within each class."""
    rng = np.random.default_rng(seed)
    gidx = {}
    for i, (g, t) in enumerate(zip(groups, yt)):
        gidx.setdefault((bool(t), g), []).append(i)
    pos = [v for (t, _), v in gidx.items() if t]
    neg = [v for (t, _), v in gidx.items() if not t]
    vals = []
    for _ in range(n_boot):
        ii = [i for j in rng.integers(0, len(pos), len(pos)) for i in pos[j]] + \
             [i for j in rng.integers(0, len(neg), len(neg)) for i in neg[j]]
        vals.append(roc_auc_score(yt[ii], scores[ii]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def separability(X, y, split, groups, a, b, n_boot, seed):
    tv = np.isin(split, ["train", "validation"]) & np.isin(y, [a, b])
    idx = np.where(tv)[0]
    oof = np.empty(len(idx))
    g_tv = groups[idx]
    for g in np.unique(g_tv):  # leave one micrograph out (micrographs belong to one class and one split)
        hold = g_tv == g
        sc, clf = _pair_lr(X[idx[~hold]], y[idx[~hold]] == a)
        oof[hold] = clf.decision_function(sc.transform(X[idx[hold]]))
    yt = y[idx] == a
    out = {"lomo_auc": float(roc_auc_score(yt, oof)), "lomo_ci95": cluster_auc_ci(oof, yt, g_tv, n_boot, seed),
           "lomo_n": [int(yt.sum()), int((~yt).sum())], "lomo_n_micrographs": int(len(np.unique(g_tv)))}
    trm = (split == "train") & np.isin(y, [a, b])
    tem = (split == "test") & np.isin(y, [a, b])
    sc, clf = _pair_lr(X[trm], y[trm] == a)
    s = clf.decision_function(sc.transform(X[tem]))
    yt2 = y[tem] == a
    out |= {"test_auc": float(roc_auc_score(yt2, s)), "test_ci95": cluster_auc_ci(s, yt2, groups[tem], n_boot, seed),
            "test_n": [int(yt2.sum()), int((~yt2).sum())]}
    return out


# ----------------------------------------------------------------------------- main
def run_feature_set(d, name, a, log, out_dir):
    X = d["X_tta"].astype(np.float64)
    lut = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([lut[c] for c in d["cls"]])
    split, groups = d["split"], d["group"]
    tr, va, te = split == "train", split == "validation", split == "test"
    nc = len(CLASSES)
    R = {"feature_set": name, "classes": CLASSES, "primary": {}, "base_only": {}, "kshot": {}, "separability": {}}
    preds = {"test_idx": np.where(te)[0], "y": y[te], "groups": groups[te], "classes": np.array(CLASSES)}

    def test_summary(name_, params):
        fit, pred, _ = LADDER[name_]
        m = fit(X[tr], y[tr], nc, base_ids=BASE_IDS, novel_ids=NOVEL_IDS, **params)
        p = pred(m, X[te]).argmax(1)
        s = summarise(y[te], p, groups[te], nc, BASE_IDS, NOVEL_IDS, n_boot=a.n_boot, seed=a.seed)
        return p, s

    # Q1/Q2 primary rule
    for cl in LADDER:
        t0 = time.time()
        params, vf, curve = select_22class_val(cl, X[tr], y[tr], X[va], y[va], nc, BASE_IDS, NOVEL_IDS)
        p, s = test_summary(cl, params)
        preds[f"primary_{cl}"] = p
        R["primary"][cl] = {"params": params, "val_macro_f1": vf, "val_curve": [[str(q), f] for q, f in curve],
                            "test": s}
        log(f"  [primary {cl:10s}] {params} val={vf:.4f} | TEST macroF1={s['macro_f1']:.4f} acc={s['accuracy']:.4f} "
            f"base={s['base_acc']:.4f} rare mean-class={s['novel_mean_class_acc']:.4f} ({time.time() - t0:.0f}s)")
    # paper rule: highest validation macro-F1 among the four classifiers (ties -> earlier in the pool)
    R["primary_reported"] = max(PRIMARY_POOL, key=lambda cl: (R["primary"][cl]["val_macro_f1"],
                                                              -PRIMARY_POOL.index(cl)))
    log(f"  primary rule selects: {R['primary_reported']}")

    # Q3 base-only selection
    eps = make_pseudo_episodes(y, split, groups, BASE_IDS, a.pseudo_draws, 4, RARE_TRAIN_MICS, k=None, seed=a.seed)
    for cl in LADDER:
        t0 = time.time()
        params, sc_, curve = select_base_only_par(cl, X, y, eps, a.n_jobs)
        p, s = test_summary(cl, params)
        preds[f"baseonly_{cl}"] = p
        R["base_only"][cl] = {"params": params, "pseudo_val_macro_f1": sc_, "curve": [[str(q), f] for q, f in curve],
                              "test": s}
        log(f"  [base-only {cl:10s}] {params} pseudo-val={sc_:.4f} | TEST macroF1={s['macro_f1']:.4f} "
            f"rare mean-class={s['novel_mean_class_acc']:.4f} ({time.time() - t0:.0f}s)")
    R["base_only_reported"] = max(PRIMARY_POOL, key=lambda cl: (R["base_only"][cl]["pseudo_val_macro_f1"],
                                                                -PRIMARY_POOL.index(cl)))
    log(f"  base-only rule selects: {R['base_only_reported']}")

    # Q4 per-K selection vs primary setting, identical support draws for every feature set and classifier
    for k in a.kshot:
        draws = kshot_draws(y, split, groups, k, a.kshot_repeats, a.seed + k)
        eps_k = make_pseudo_episodes(y, split, groups, BASE_IDS, a.pseudo_draws, 4, RARE_TRAIN_MICS, k=k,
                                     seed=a.seed + 1000 + k)
        RK = R["kshot"][str(k)] = {}
        for cl in KSHOT_METHODS:
            t0 = time.time()
            pk, sk, ck = select_base_only_par(cl, X, y, eps_k, a.n_jobs)
            prim = R["primary"][cl]["params"]
            r_k = kshot_eval_par(cl, pk, X, y, te, draws, a.n_jobs)
            r_p = r_k if pk == prim else kshot_eval_par(cl, prim, X, y, te, draws, a.n_jobs)
            RK[cl] = {"perK_params": pk, "pseudo_val": sk, "perK": r_k, "primary_params": prim, "primary": r_p}
            log(f"  [K={k:2d} {cl:10s}] per-K {pk}: rare={r_k['novel_mean_class_acc']['mean']:.3f} | primary {prim}: "
                f"rare={r_p['novel_mean_class_acc']['mean']:.3f} ({time.time() - t0:.0f}s)")

    # Q5 separability
    for pa, pb in PAIRS:
        r = separability(X, y, split, groups, lut[pa], lut[pb], a.n_boot, a.seed)
        R["separability"][f"{pa}|{pb}"] = r
        log(f"  [sep {pa} vs {pb}] LOMO(train+val) AUC={r['lomo_auc']:.3f} {r['lomo_ci95']} | "
            f"train->test AUC={r['test_auc']:.3f} {r['test_ci95']} (n test {r['test_n']})")

    np.savez_compressed(out_dir / f"predictions_{name}.npz", **preds)
    return R


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--features", nargs="+", required=True, help="checkpoint paths and/or imagenet:<backbone>")
    ap.add_argument("--out", default="runs/g3")
    ap.add_argument("--kshot", default="1,3,5,10")
    ap.add_argument("--kshot-repeats", type=int, default=30)
    ap.add_argument("--pseudo-draws", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    a.kshot = [int(k) for k in a.kshot.split(",") if k]
    seed_everything(a.seed)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")
    if not inference_fp32():
        log("WARNING: TEMFS_FP32 is not 1 - embeddings will not match the paper's fp32 numbers")
    device = get_device(a.device)
    state = {}
    meta = {"prereg": "prereg/G3_frozen_feature_study_2026-11-20.md", "args": {k: v for k, v in vars(a).items()},
            "environment": environment_info(), "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    for spec in a.features:
        name = feature_name(spec)
        jf = out_dir / f"g3_{name}.json"
        if jf.exists():
            log(f"== {name}: done already ({jf}), skipped")
            continue
        log(f"== {name} ({spec})")
        t0 = time.time()
        d = embed_cached(spec, out_dir, a.data_root, device, log, state)
        R = run_feature_set(d, name, a, log, out_dir)
        R["meta"] = meta | {"spec": spec, "seconds": round(time.time() - t0)}
        save_json(R, jf)
        log(f"== {name} done in {time.time() - t0:.0f}s -> {jf}")


if __name__ == "__main__":
    main()
