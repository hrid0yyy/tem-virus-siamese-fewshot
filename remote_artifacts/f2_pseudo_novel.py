"""F2 / F3 - pseudo-novel simulation: the key go/no-go test for the Siamese approach.

1. Hide 4 of the 14 base classes completely (all splits) and train the Siamese encoder on the
   remaining 10 (official train split; model selection on the official validation split of the 10).
2. On the 4 hidden classes (never seen in training) run image-disjoint 4-way K-shot episodes
   (support and query patches always from different micrographs), pooling all their splits.
3. Compare with the frozen ImageNet backbone on the *same* episodes (paired comparison).

GO   = trained encoder beats frozen ImageNet features on unseen classes (K=5) with a paired 95% CI > 0.
WEAK = trained encoder better on average, but the paired CI includes 0.
NO-GO= trained encoder not better -> the learned metric does not transfer to unseen classes.

    python f2_pseudo_novel.py --data-root data                     # F2 (preset f2)
    python f2_pseudo_novel.py --data-root data --hidden preset:f3  # F3 (hidden classes with look-alikes)
    python f2_pseudo_novel.py --data-root data --checkpoint runs/<run>/best.pt   # re-evaluate only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temfs.constants import BASE_CLASSES, SPLITS, resolve_classes  # noqa: E402
from temfs.data import select_records  # noqa: E402
from temfs.fewshot import episodic_eval  # noqa: E402
from temfs.metrics import mean_ci95  # noqa: E402
from temfs.pipeline import (build_bank, frozen_imagenet_model, load_checkpoint, make_config,  # noqa: E402
                            prepare_data, run_training)
from temfs.utils import Logger, get_device, save_json, seed_everything  # noqa: E402


def int_list(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--hidden", default="preset:f2", help="preset:f2 | preset:f3 | comma list of base classes")
    ap.add_argument("--checkpoint", default=None, help="skip training and evaluate this checkpoint")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--backbone", default="densenet201")
    ap.add_argument("--loss", default="supcon", choices=["supcon", "triplet", "proto", "pair", "contrastive"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--lr-backbone", type=float, default=None)
    ap.add_argument("--lr-head", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=None)
    ap.add_argument("--k-list", default="1,5,10")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--n-query", type=int, default=15)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--no-preload", action="store_true")
    args = ap.parse_args(argv)

    hidden = resolve_classes(args.hidden)
    not_base = [c for c in hidden if c not in BASE_CLASSES]
    if not_base:
        raise ValueError(f"hidden classes must be base classes: {not_base}")
    if len(hidden) < 2:
        raise ValueError("need at least 2 hidden classes")
    seed_everything(args.seed)
    device = get_device(args.device)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        run_dir = ckpt_path.parent
    else:
        cfg = make_config("metric", {
            "train_classes": "base14", "exclude_classes": ",".join(hidden), "backbone": args.backbone,
            "loss": args.loss, "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
            "P": args.P, "K": args.K, "lr_backbone": args.lr_backbone, "lr_head": args.lr_head,
            "max_steps": args.max_steps, "pretrained": args.pretrained, "seed": args.seed,
            "device": args.device, "skip_verify": args.skip_verify, "preload": not args.no_preload,
            "patience": 0,
        })
        tag = "f3" if args.hidden == "preset:f3" else "f2"
        run_name = f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}_pseudonovel_{args.loss}_{args.backbone}"
        run_dir, ckpt_path = run_training(cfg, args.data_root, args.out_dir, run_name)

    log = Logger(Path(run_dir) / "f2_eval_log.txt")
    model, cfg, ckpt = load_checkpoint(ckpt_path, device)
    overlap = [c for c in hidden if c in cfg["train_classes"]]
    if overlap:
        raise ValueError(f"checkpoint was trained on hidden classes {overlap}; not a valid F2 evaluation")
    log(f"Hidden (pseudo-novel) classes: {hidden}")
    log(f"Encoder trained on: {cfg['train_classes']}")

    records, store = prepare_data(args.data_root, log, preload=not args.no_preload, skip_verify=args.skip_verify)
    recs = select_records(records, SPLITS, hidden)
    log(f"{len(recs)} patches of hidden classes (all splits pooled, image-disjoint episodes)")
    bank_t = build_bank(model, recs, store, device, tta=args.tta)
    del model
    frozen = frozen_imagenet_model(cfg["backbone"], device)
    bank_f = build_bank(frozen, recs, store, device, tta=args.tta, keys=("feat",))

    results = {"hidden": hidden, "trained_on": cfg["train_classes"], "checkpoint": str(ckpt_path),
               "k_list": int_list(args.k_list), "episodes": args.episodes, "n_query": args.n_query,
               "tta": args.tta, "by_k": {}}
    n_way = len(hidden)
    for k in int_list(args.k_list):
        row = {}
        evals = {"trained_head": (bank_t, "head"), "trained_feat": (bank_t, "feat"),
                 "frozen_imagenet": (bank_f, "feat")}
        per_ep = {}
        for name, (bank, key) in evals.items():
            e = episodic_eval(bank, key, hidden, n_way=n_way, k_shot=k, n_query=args.n_query,
                              episodes=args.episodes, seed=args.seed, return_per_episode=True)
            per_ep[name] = np.array(e.pop("per_episode_acc"))
            row[name] = e
        for name in ("trained_head", "trained_feat"):
            row[f"paired_diff_{name}_vs_frozen"] = mean_ci95(per_ep[name] - per_ep["frozen_imagenet"])
        results["by_k"][str(k)] = row
        log(f"{n_way}-way {k}-shot | trained(head)={row['trained_head']['acc']['mean']:.4f} "
            f"trained(feat)={row['trained_feat']['acc']['mean']:.4f} "
            f"frozen={row['frozen_imagenet']['acc']['mean']:.4f} | paired diff head-frozen="
            f"{row['paired_diff_trained_head_vs_frozen']['mean']:+.4f}+-{row['paired_diff_trained_head_vs_frozen']['ci95']:.4f}")

    k_ref = "5" if "5" in results["by_k"] else list(results["by_k"])[-1]
    ref = results["by_k"][k_ref]
    best_name = max(("trained_head", "trained_feat"), key=lambda n: ref[n]["acc"]["mean"])
    d = ref[f"paired_diff_{best_name}_vs_frozen"]
    if d["mean"] - d["ci95"] > 0:
        verdict = "GO"
    elif d["mean"] > 0:
        verdict = "WEAK GO (better on average, paired 95% CI includes 0)"
    else:
        verdict = "NO-GO (learned metric does not beat frozen ImageNet features on unseen classes)"
    results["verdict"] = {"k": int(k_ref), "best_embedding": best_name, "paired_diff": d, "decision": verdict}
    out = Path(run_dir) / "f2_results.json"
    save_json(results, out)
    log("")
    log(f"F2 DECISION @ {n_way}-way {k_ref}-shot using {best_name}: {verdict}")
    log(f"  paired accuracy difference vs frozen ImageNet = {d['mean']:+.4f} +- {d['ci95']:.4f} (95% CI)")
    log(f"Saved {out}")
    return results


if __name__ == "__main__":
    main()
