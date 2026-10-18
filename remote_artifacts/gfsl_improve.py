"""Rare-class improvement study on FROZEN features (no encoder retraining).

For each checkpoint:
  1. embed all patches of the 22 classes (backbone 'feat', D4 TTA),
  2. separability diagnostic on VALIDATION for the confusable pairs,
  3. for each method (ncm, maha, logreg, dc): choose hyper-parameters on VALIDATION
     (fit on official train, score validation macro-F1), then evaluate ONCE on TEST,
  4. K-shot check (K = 1, 5 rare-class training patches from different micrographs, repeated).

    python gfsl_improve.py --data-root data --checkpoints runs/A/best.pt runs/B/best.pt --out runs/gfsl_improve
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import summarise  # noqa: E402
from temfs.constants import NOVEL_CLASSES, SPLITS  # noqa: E402
from temfs.data import select_records  # noqa: E402
from temfs.gfsl_methods import METHODS, macro_f1, pair_separability, select_on_validation  # noqa: E402
from temfs.metrics import harmonic_mean, mean_ci95, subset_accuracy, subset_mean_class_accuracy  # noqa: E402
from temfs.pipeline import build_bank, load_checkpoint, prepare_data  # noqa: E402
from temfs.samplers import draw_image_balanced  # noqa: E402
from temfs.utils import Logger, environment_info, get_device, save_json, seed_everything  # noqa: E402

PAIRS = [("Dengue", "Sapovirus"), ("Dengue", "TBE"), ("Dengue", "Astrovirus"), ("Sapovirus", "Norovirus"),
         ("Sapovirus", "Astrovirus"), ("Guanarito", "Lassa"), ("Guanarito", "Machupo"), ("LCM", "Machupo")]
WEAK = ["Dengue", "Sapovirus", "Guanarito", "LCM", "Machupo", "Astrovirus", "Adenovirus", "Lassa"]


def kshot_eval(fit, pred, params, X, y, groups, split, classes, base_ids, novel_ids, ks, repeats, seed):
    """Rare classes get K train patches (K micrographs); base classes keep all train patches."""
    rng = np.random.default_rng(seed)
    tr, te = split == "train", split == "test"
    base_idx = np.where(tr & np.isin(y, base_ids))[0]
    pools = {}
    for c in novel_ids:
        g2i = {}
        for i in np.where(tr & (y == c))[0]:
            g2i.setdefault(groups[i], []).append(int(i))
        pools[c] = g2i
    out = {}
    for k in ks:
        st = {"novel_mean_class_acc": [], "base_acc": [], "hm": [], "macro_f1": []}
        for _ in range(repeats):
            idx = list(base_idx)
            for c in novel_ids:
                order = list(rng.permutation(list(pools[c].keys())))
                idx += draw_image_balanced(rng, pools[c], order, k, allow_repeat=False)
            idx = np.array(idx)
            m = fit(X[idx], y[idx], len(classes), base_ids=base_ids, novel_ids=novel_ids, **params)
            p = pred(m, X[te]).argmax(1)
            yt = y[te]
            b, n = subset_accuracy(yt, p, base_ids), subset_mean_class_accuracy(yt, p, novel_ids)
            st["base_acc"].append(b)
            st["novel_mean_class_acc"].append(n)
            st["hm"].append(harmonic_mean(b, n))
            st["macro_f1"].append(macro_f1(yt, p, len(classes)))
        out[str(k)] = {m_: mean_ci95(v) for m_, v in st.items()}
        out[str(k)]["per_draw"] = {m_: [float(x) for x in v] for m_, v in st.items()}  # same draws for every model
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", default="runs/gfsl_improve")
    ap.add_argument("--methods", default="ncm,maha,logreg,dc")
    ap.add_argument("--kshot", default="1,5")
    ap.add_argument("--kshot-repeats", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args(argv)

    seed_everything(a.seed)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")
    device = get_device(a.device)
    methods = [m for m in a.methods.split(",") if m]
    ks = [int(k) for k in a.kshot.split(",") if k]
    records, store = prepare_data(a.data_root, log, skip_verify=a.skip_verify)
    results = {"methods": methods, "selection": "validation macro-F1 (train -> validation), applied once to test",
               "environment": environment_info(), "checkpoints": {}}

    for ck in a.checkpoints:
        t0 = time.time()
        model, cfg, _ = load_checkpoint(ck, device)
        base = list(cfg["train_classes"])
        novel = [c for c in NOVEL_CLASSES if c not in base]
        classes = base + novel
        base_ids, novel_ids = list(range(len(base))), list(range(len(base), len(classes)))
        recs = select_records(records, SPLITS, classes)
        bank = build_bank(model, recs, store, device, tta=a.tta, keys=("feat",))
        del model
        X = bank.emb["feat"]
        lut = {c: i for i, c in enumerate(classes)}
        y = np.array([lut[c] for c in bank.cls])
        split, groups = bank.split, bank.group
        tr, va, te = split == "train", split == "validation", split == "test"
        log(f"== {ck}: embedded {len(X)} patches in {time.time() - t0:.0f}s; base={len(base)} rare={len(novel)}")
        R = {"diagnostic_validation": {}, "methods": {}}
        preds = {"test_idx": np.where(te)[0], "y": y[te], "groups": np.array(groups[te], dtype=str),
                 "classes": np.array(classes, dtype=str)}
        for p, q in PAIRS:
            if p in lut and q in lut:
                d = pair_separability(X[tr], y[tr], X[va], y[va], lut[p], lut[q])
                R["diagnostic_validation"][f"{p}|{q}"] = d
                log(f"  separability (val) {p} vs {q}: logreg AUC={d['auc_logreg']:.3f} acc={d['acc_logreg']:.3f} | "
                    f"2-class NCM acc={d['acc_ncm']:.3f}  (n={d['n_val_a']}/{d['n_val_b']})")
        for name in methods:
            fit, pred, _ = METHODS[name]
            params, val_f1, curve = select_on_validation(name, X[tr], y[tr], X[va], y[va], len(classes),
                                                         base_ids, novel_ids, log=log)
            m = fit(X[tr], y[tr], len(classes), base_ids=base_ids, novel_ids=novel_ids, **params)
            p_te = pred(m, X[te]).argmax(1)
            preds[f"{name}_pred"] = p_te
            s = summarise(y[te], p_te, groups[te], len(classes), base_ids, novel_ids, n_boot=a.n_boot, seed=a.seed)
            s["weak_f1"] = {c: s["per_class_f1"][lut[c]] for c in WEAK if c in lut}
            s["weak_recall"] = {c: s["per_class_recall"][lut[c]] for c in WEAK if c in lut}
            ksr = kshot_eval(fit, pred, params, X, y, groups, split, classes, base_ids, novel_ids, ks,
                             a.kshot_repeats, a.seed)
            R["methods"][name] = {"selected_params": params, "val_macro_f1": val_f1,
                                  "val_curve": [[str(p_), f] for p_, f in curve], "test": s, "kshot": ksr}
            log(f"  [{name}] TEST acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f} base={s['base_acc']:.4f} "
                f"rare mean-class={s['novel_mean_class_acc']:.4f} HM={s['hm_mean_class_acc']:.4f} | weak F1: " +
                ", ".join(f"{c}={v:.2f}" for c, v in s["weak_f1"].items()) + " | K-shot rare acc: " +
                ", ".join(f"K={k}:{v['novel_mean_class_acc']['mean']:.3f}" for k, v in ksr.items()))
        results["checkpoints"][ck] = R
        np.savez_compressed(out_dir / f"predictions_{Path(ck).parent.name}.npz", **preds)
        save_json(results, out_dir / "gfsl_improve.json")
    log(f"done -> {out_dir / 'gfsl_improve.json'}")
    return results


if __name__ == "__main__":
    main()
