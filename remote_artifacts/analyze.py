"""Inference-time variants + per-patch predictions for paired statistics (no retraining).

For a checkpoint (Siamese or classifier) this:
  1. embeds every patch of all 22 classes (D4 TTA by default),
  2. evaluates embedding x transform in {feat, head} x {l2, cl2n}:
       - closed 14-class nearest-prototype on the official TEST split,
       - 22-class GFSL on TEST, raw (gamma=0) and with calibrated stacking where gamma is chosen
         on the VALIDATION split only (prototypes from train, queries from validation),
  3. for classifiers also stores the logit predictions,
  4. writes analysis.json, analysis_summary.txt and predictions.npz (for compare.py).

    python analyze.py --data-root data --checkpoint runs/<run>/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temfs.analysis import (base_train_mean, choose_gamma, cl2n_transform, gfsl_scores,  # noqa: E402
                            stacked_predict)
from temfs.constants import NOVEL_CLASSES, SPLITS  # noqa: E402
from temfs.data import select_records  # noqa: E402
from temfs.fewshot import ncm_scores, prototypes  # noqa: E402
from temfs.metrics import (classification_scores, harmonic_mean, stratified_cluster_bootstrap,  # noqa: E402
                           subset_accuracy, subset_mean_class_accuracy)
from temfs.pipeline import build_bank, load_checkpoint, prepare_data  # noqa: E402
from temfs.utils import Logger, get_device, save_json, seed_everything  # noqa: E402

GAMMA_GRID = np.round(np.arange(-0.10, 0.2001, 0.005), 4)


def summarise(y, pred, groups, n_classes, base_ids=None, novel_ids=None, n_boot=1000, seed=0):
    s = classification_scores(y, pred, n_classes)
    out = {k: s[k] for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "balanced_accuracy")}
    out["per_class_f1"] = s["per_class"]["f1"]
    out["per_class_recall"] = s["per_class"]["recall"]
    out["per_class_precision"] = s["per_class"]["precision"]
    if n_boot:
        out["ci95"] = stratified_cluster_bootstrap(y, pred, groups, n_classes, n_boot=n_boot, seed=seed)
    if base_ids is not None:
        b, n = subset_accuracy(y, pred, base_ids), subset_accuracy(y, pred, novel_ids)
        bm, nm = subset_mean_class_accuracy(y, pred, base_ids), subset_mean_class_accuracy(y, pred, novel_ids)
        out.update(base_acc=b, novel_acc=n, hm_acc=harmonic_mean(b, n),
                   base_mean_class_acc=bm, novel_mean_class_acc=nm, hm_mean_class_acc=harmonic_mean(bm, nm))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "analysis_log.txt")
    model, cfg, ckpt = load_checkpoint(args.checkpoint, device)
    base = list(cfg["train_classes"])
    novel = [c for c in NOVEL_CLASSES if c not in base]
    log(f"Checkpoint {args.checkpoint} | model_type={cfg['model_type']} | base={len(base)} novel={len(novel)}")

    records, store = prepare_data(args.data_root, log, skip_verify=args.skip_verify)
    recs = select_records(records, SPLITS, base + novel)
    t0 = time.time()
    bank = build_bank(model, recs, store, device, tta=args.tta, batch_size=args.batch_size)
    log(f"Embedded {len(recs)} patches in {time.time() - t0:.0f}s (TTA={args.tta}); keys={list(bank.emb)}")

    results = {"checkpoint": str(args.checkpoint), "config": {k: v for k, v in cfg.items() if k != "environment"},
               "base": base, "novel": novel, "tta": args.tta, "gamma_grid": GAMMA_GRID.tolist(), "variants": {}}
    preds = {"paths": np.array([r.path for r in recs])}
    nb = len(base)

    # ---------- classifier logits (closed set over the training classes) ----------
    if "logits" in bank.emb:
        qm = bank.mask("test", base)
        y = bank.labels(qm, base)
        pred = bank.emb["logits"][qm].argmax(1)
        kw = {}
        if any(c in NOVEL_CLASSES for c in base):
            kw = {"base_ids": [i for i, c in enumerate(base) if c not in NOVEL_CLASSES],
                  "novel_ids": [i for i, c in enumerate(base) if c in NOVEL_CLASSES]}
        results["variants"]["logits_closed"] = summarise(y, pred, bank.group[qm], nb, n_boot=args.n_boot,
                                                         seed=args.seed, **kw)
        preds["logits_closed_idx"] = np.where(qm)[0]
        preds["logits_closed_pred"] = pred
        preds["logits_closed_y"] = y
        log(f"[logits] closed test acc={results['variants']['logits_closed']['accuracy']:.4f} "
            f"macroF1={results['variants']['logits_closed']['macro_f1']:.4f}")

    # ---------- nearest-prototype variants ----------
    for key in [k for k in ("feat", "head") if k in bank.emb]:
        mu = base_train_mean(bank, key, base)
        for tname in ("l2", "cl2n"):
            emb = bank.emb[key] if tname == "l2" else cl2n_transform(bank.emb[key], mu)
            name = f"{key}_{tname}"
            v = {}
            # closed set over the training classes
            sm, qm = bank.mask("train", base), bank.mask("test", base)
            protos = prototypes(emb[sm], bank.labels(sm, base), nb)
            y = bank.labels(qm, base)
            pred = ncm_scores(emb[qm], protos).argmax(1)
            v["closed_test"] = summarise(y, pred, bank.group[qm], nb, n_boot=args.n_boot, seed=args.seed)
            preds[f"{name}_closed_idx"], preds[f"{name}_closed_pred"], preds[f"{name}_closed_y"] = np.where(qm)[0], pred, y
            if novel:
                classes = base + novel
                nc = len(classes)
                b_ids, n_ids = list(range(nb)), list(range(nb, nc))
                s_val, y_val, _, _ = gfsl_scores(bank, emb, base, novel, "validation")
                g_star, g_val, curve = choose_gamma(s_val, y_val, nb, nc, GAMMA_GRID)
                v["gamma_selected_on_val"] = g_star
                v["val_macro_f1_at_gamma"] = g_val
                v["val_macro_f1_at_0"] = dict(curve)[0.0]
                v["gamma_curve_val"] = curve
                s_te, y_te, g_te, qm_te = gfsl_scores(bank, emb, base, novel, "test")
                for tag, g in (("raw", 0.0), ("calibrated", g_star)):
                    p = stacked_predict(s_te, nb, g)
                    v[f"gfsl_test_{tag}"] = summarise(y_te, p, g_te, nc, b_ids, n_ids, n_boot=args.n_boot,
                                                      seed=args.seed)
                    preds[f"{name}_gfsl_{tag}_pred"] = p
                preds[f"{name}_gfsl_idx"], preds[f"{name}_gfsl_y"] = np.where(qm_te)[0], y_te
            results["variants"][name] = v
            c = v["closed_test"]
            msg = f"[{name}] closed acc={c['accuracy']:.4f} F1={c['macro_f1']:.4f}"
            if novel:
                r_, k_ = v["gfsl_test_raw"], v["gfsl_test_calibrated"]
                msg += (f" | GFSL raw acc={r_['accuracy']:.4f} F1={r_['macro_f1']:.4f} HM={r_['hm_mean_class_acc']:.4f}"
                        f" | calibrated(g={v['gamma_selected_on_val']:+.3f} from val) acc={k_['accuracy']:.4f} "
                        f"F1={k_['macro_f1']:.4f} HM={k_['hm_mean_class_acc']:.4f}")
            log(msg)

    save_json(results, out_dir / "analysis.json")
    np.savez_compressed(out_dir / "predictions.npz", **preds)
    lines = [f"checkpoint: {args.checkpoint}", f"TTA={args.tta}", ""]
    for name, v in results["variants"].items():
        if name == "logits_closed":
            lines.append(f"logits closed : acc={v['accuracy']:.4f} macroF1={v['macro_f1']:.4f} "
                         f"CI acc=[{v['ci95']['accuracy']['low']:.4f},{v['ci95']['accuracy']['high']:.4f}]")
            continue
        c = v["closed_test"]
        lines.append(f"{name:10s} closed14 : acc={c['accuracy']:.4f} macroF1={c['macro_f1']:.4f} "
                     f"CI acc=[{c['ci95']['accuracy']['low']:.4f},{c['ci95']['accuracy']['high']:.4f}]")
        for tag in ("raw", "calibrated"):
            k = f"gfsl_test_{tag}"
            if k in v:
                g = v[k]
                gam = f" gamma={v['gamma_selected_on_val']:+.3f}(val)" if tag == "calibrated" else ""
                lines.append(f"{'':10s} gfsl22 {tag:10s}: acc={g['accuracy']:.4f} macroF1={g['macro_f1']:.4f} "
                             f"base={g['base_acc']:.4f} novel={g['novel_acc']:.4f} "
                             f"mean-class base={g['base_mean_class_acc']:.4f} novel={g['novel_mean_class_acc']:.4f} "
                             f"HM={g['hm_mean_class_acc']:.4f}{gam}")
    (out_dir / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    log(f"Wrote {out_dir}")
    return results


if __name__ == "__main__":
    main()
