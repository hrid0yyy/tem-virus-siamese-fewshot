"""Evaluate a trained checkpoint (or a frozen ImageNet backbone) with all protocols.

Protocols (official splits unless stated):
  closed   - E1: nearest-prototype on the training classes, official test split
             (+ classifier logits for CE baselines). Directly comparable with the paper (93.1 % / 0.921).
  gfsl     - E2: generalised few-shot, training (base) classes + novel classes, prototypes from the
             official train split, test on the official test split. Reports base/novel accuracy + HM.
  kshot    - E3: novel classes get only K support patches (K different micrographs), repeated.
  episodic - E4: image-disjoint N-way K-shot episodes pooled over ALL splits of classes the encoder
             never saw (novel classes, or the hidden classes of F2).
  openset  - E5: unknown-virus detection, known = training classes, unknown = novel classes.

    python evaluate.py --data-root data --checkpoint runs/<run>/best.pt
    python evaluate.py --data-root data --frozen-backbone densenet201          # ImageNet baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temfs.constants import SPLITS, resolve_classes  # noqa: E402
from temfs.data import select_records  # noqa: E402
from temfs.pipeline import (build_bank, evaluate_bank, frozen_imagenet_model, load_checkpoint,  # noqa: E402
                            prepare_data)
from temfs.utils import Logger, environment_info, get_device, save_json, seed_everything, write_confusion_csv  # noqa: E402

ALL_PROTOCOLS = ("closed", "gfsl", "kshot", "episodic", "openset")


def int_list(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="best.pt / last.pt from train.py")
    src.add_argument("--frozen-backbone", help="evaluate an untrained ImageNet backbone (feat embeddings)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default=None, help="default: <checkpoint dir>/eval_<time> or reports/")
    ap.add_argument("--base-classes", default="base14", help="only for --frozen-backbone")
    ap.add_argument("--novel-classes", default="novel8")
    ap.add_argument("--protocols", default=",".join(ALL_PROTOCOLS))
    ap.add_argument("--embeddings", default="auto", help="head,feat | head | feat | auto")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--novel-support-splits", default="train",
                    help="splits used for novel prototypes in gfsl (default: official train split)")
    ap.add_argument("--kshot-list", default="1,3,5,10")
    ap.add_argument("--kshot-repeats", type=int, default=100)
    ap.add_argument("--episodic-classes", default=None, help="default: novel classes")
    ap.add_argument("--episodic-way", type=int, default=None, help="default: all episodic classes")
    ap.add_argument("--episodic-k", default="1,5")
    ap.add_argument("--n-query", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-preload", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    return ap


def run(args, log=None):
    seed_everything(args.seed)
    device = get_device(args.device)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.checkpoint:
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / f"eval_{stamp}"
    else:
        out_dir = Path(args.out_dir or "reports") / f"eval_frozen_{args.frozen_backbone}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = log or Logger(out_dir / "eval_log.txt")
    log(f"Environment: {environment_info()}")

    if args.checkpoint:
        model, cfg, ckpt = load_checkpoint(args.checkpoint, device)
        base = cfg["train_classes"]
        model_type = cfg["model_type"]
        log(f"Checkpoint {args.checkpoint} (epoch {ckpt.get('epoch')}, {cfg.get('select_metric')}="
            f"{ckpt.get('best_metric')}) | model_type={model_type} | trained on {len(base)} classes")
        avail = ["head", "feat"] if model_type == "metric" else ["feat"]
    else:
        model = frozen_imagenet_model(args.frozen_backbone, device)
        base = resolve_classes(args.base_classes)
        model_type = "frozen"
        avail = ["feat"]
        cfg = {"model_type": "frozen", "backbone": args.frozen_backbone, "train_classes": base}
    keys = avail if args.embeddings == "auto" else [k for k in args.embeddings.split(",") if k]
    bad = [k for k in keys if k not in avail]
    if bad:
        raise ValueError(f"embeddings {bad} not available for this model (available: {avail})")

    novel = [c for c in resolve_classes(args.novel_classes) if c not in base]
    exclude_hidden = cfg.get("exclude_classes") or []
    episodic_classes = resolve_classes(args.episodic_classes) if args.episodic_classes else (novel or exclude_hidden)
    protocols = [p for p in args.protocols.split(",") if p]
    unknown = [p for p in protocols if p not in ALL_PROTOCOLS]
    if unknown:
        raise ValueError(f"unknown protocols {unknown}")

    records, store = prepare_data(args.data_root, log, preload=not args.no_preload, skip_verify=args.skip_verify)
    needed = sorted(set(base) | set(novel) | set(episodic_classes))
    recs = select_records(records, SPLITS, needed)
    t0 = time.time()
    bank = build_bank(model, recs, store, device, tta=args.tta, batch_size=args.batch_size,
                      num_workers=args.num_workers)
    log(f"Embedded {len(recs)} patches in {time.time() - t0:.0f}s (TTA={args.tta}) keys={list(bank.emb)}")

    opts = {"n_boot": args.n_boot, "seed": args.seed, "kshot_list": int_list(args.kshot_list),
            "kshot_repeats": args.kshot_repeats, "episodic_classes": episodic_classes,
            "episodic_way": args.episodic_way, "episodic_k": int_list(args.episodic_k),
            "n_query": args.n_query, "episodes": args.episodes,
            "novel_support_splits": [s for s in args.novel_support_splits.split(",") if s]}
    results = evaluate_bank(bank, base, novel, keys, protocols, opts, log,
                            model_type="classifier" if model_type == "classifier" else "metric")
    results["config"] = {k: v for k, v in cfg.items() if k != "environment"}
    results["eval_args"] = vars(args)
    results["environment"] = environment_info()
    save_json(results, out_dir / "results.json")

    # Confusion matrices as CSV for quick inspection
    for key in keys:
        for proto in ("closed", "gfsl"):
            r = results.get(key, {}).get(proto)
            if r:
                write_confusion_csv(out_dir / f"confusion_{key}_{proto}.csv", r["confusion"], r["class_names"])
    if "logits" in results:
        r = results["logits"]["closed"]
        write_confusion_csv(out_dir / "confusion_logits_closed.csv", r["confusion"], r["class_names"])
    write_summary(results, keys, out_dir / "summary.txt")
    log(f"Results written to {out_dir}")
    return results, out_dir


def _fmt(x):
    return "  n/a " if x is None else f"{x:.4f}"


def write_summary(results, keys, path):
    lines = [f"Base (training) classes: {len(results['base_classes'])} | novel: {results['novel_classes']}", ""]
    if "logits" in results:
        c = results["logits"]["closed"]
        ci = c.get("bootstrap_ci95", {})
        lines.append(f"[classifier logits] closed test: acc={_fmt(c['accuracy'])} macroF1={_fmt(c['macro_f1'])}"
                     + (f"  acc95%CI=[{ci['accuracy']['low']:.4f},{ci['accuracy']['high']:.4f}]" if ci else ""))
        if "gfsl_patch" in c:
            g = c["gfsl_patch"]
            lines.append(f"    base acc={_fmt(g['base_acc'])} novel acc={_fmt(g['novel_acc'])} HM={_fmt(g['hm_acc'])}")
    for key in keys:
        r = results.get(key, {})
        lines.append(f"=== embedding: {key} ===")
        if "closed" in r:
            c = r["closed"]
            ci = c.get("bootstrap_ci95", {})
            lines.append(f"closed (E1) test : acc={_fmt(c['accuracy'])} macroF1={_fmt(c['macro_f1'])} "
                         f"image-acc={_fmt(c['image_level']['accuracy'])}"
                         + (f"  acc95%CI=[{ci['accuracy']['low']:.4f},{ci['accuracy']['high']:.4f}]" if ci else ""))
        if "gfsl" in r:
            g = r["gfsl"]
            gp = g["gfsl_patch"]
            lines.append(f"GFSL (E2) test   : all acc={_fmt(g['accuracy'])} macroF1={_fmt(g['macro_f1'])} | "
                         f"base={_fmt(gp['base_acc'])} novel={_fmt(gp['novel_acc'])} HM={_fmt(gp['hm_acc'])} | "
                         f"novel-only={_fmt(g['novel_only_acc'])}")
            names = g["class_names"]
            rec = g["per_class"]["recall"]
            lines.append("    per-class recall: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(names, rec)))
        if "kshot" in r:
            for k, v in r["kshot"].items():
                lines.append(f"K-shot (E3) K={k:>2s}: novel={v['novel_acc']['mean']:.4f}+-{v['novel_acc']['ci95']:.4f} "
                             f"base={v['base_acc']['mean']:.4f} HM={v['hm_acc']['mean']:.4f} "
                             f"novel-only={v['novel_only_acc']['mean']:.4f}")
        if "episodic" in r:
            for k, v in r["episodic"].items():
                lines.append(f"episodic (E4) {v['n_way']}-way {k}-shot: acc={v['acc']['mean']:.4f}"
                             f"+-{v['acc']['ci95']:.4f} ({v['episodes']} episodes)")
        if "openset" in r:
            o = r["openset"]
            lines.append(f"open-set (E5): patch AUROC={_fmt(o['patch']['auroc'])} FPR@95TPR={_fmt(o['patch']['fpr_at_95tpr'])}"
                         f" | image AUROC={_fmt(o['image']['auroc'])}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
