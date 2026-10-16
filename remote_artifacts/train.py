"""Train a Siamese/metric encoder or a cross-entropy classifier baseline.

Examples
--------
Siamese encoder (supervised contrastive, DenseNet201) on the 14 base classes  [E1/E2 main model]:
    python train.py --data-root data --model-type metric --loss supcon

Classic pairwise Siamese network (Koch et al. 2015) ablation:
    python train.py --data-root data --model-type metric --loss pair

Batch-hard triplet / prototypical-episode ablations:
    python train.py --data-root data --model-type metric --loss triplet
    python train.py --data-root data --model-type metric --loss proto

Paper replication (DenseNet201 + Flatten-1024-1024 head, Adam 1e-5, batch 12, 50 epochs, augmented_train):
    python train.py --data-root data --model-type classifier

22-class cross-entropy baseline (all classes, class-balanced sampling)  [E2 baseline]:
    python train.py --data-root data --model-type classifier --train-classes all22 \
        --train-source train --sampler balanced --augment --max-shift 8

Feasibility F2 (hide 4 base classes from training) is easier with f2_pseudo_novel.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temfs.models import BACKBONES  # noqa: E402
from temfs.pipeline import make_config, run_training  # noqa: E402


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "true", "yes", "y"):
        return True
    if v.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("run")
    g.add_argument("--data-root", required=True)
    g.add_argument("--out-dir", default="runs")
    g.add_argument("--run-name", default=None)
    g.add_argument("--model-type", choices=["metric", "classifier"], default="metric")
    g.add_argument("--train-classes", default=None, help="base14 | all22 | novel8 | comma list (default base14)")
    g.add_argument("--exclude-classes", default=None,
                   help="classes removed from training (e.g. preset:f2 or 'Astrovirus,Influenza')")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--device", default=None, help="auto | cuda | cpu")
    g.add_argument("--deterministic", action="store_true", default=None)
    g.add_argument("--num-workers", type=int, default=None)
    g.add_argument("--no-preload", dest="preload", action="store_false", default=None)
    g.add_argument("--skip-verify", action="store_true", default=None)
    g.add_argument("--max-steps", type=int, default=None, help="debug: limit steps per epoch")
    g.add_argument("--resume", action="store_true", default=None,
                   help="continue runs/<run-name> from last.pt (skips if already finished)")
    g.add_argument("--debug-stop-after-epoch", type=int, default=None, help=argparse.SUPPRESS)

    g = ap.add_argument_group("model")
    g.add_argument("--backbone", choices=BACKBONES, default=None)
    g.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=None)
    g.add_argument("--memory-efficient", type=str2bool, default=None,
                   help="DenseNet gradient checkpointing (saves GPU memory). default: metric=true, classifier=false")
    g.add_argument("--emb-dim", type=int, default=None)
    g.add_argument("--hidden-dim", type=int, default=None)
    g.add_argument("--ce-head", choices=["paper", "linear", "cosine"], default=None)
    g.add_argument("--dropout", type=float, default=None)

    g = ap.add_argument_group("metric learning")
    g.add_argument("--loss", choices=["supcon", "triplet", "proto", "pair", "contrastive"], default=None)
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--margin", type=float, default=None)
    g.add_argument("--soft-margin", action="store_true", default=None)
    g.add_argument("--aux-ce-weight", type=float, default=None, help="add cosine-classifier CE (Baseline++) term")
    g.add_argument("--aux-scale", type=float, default=None)
    g.add_argument("--P", type=int, default=None, help="classes per batch")
    g.add_argument("--K", type=int, default=None, help="samples per class per batch")
    g.add_argument("--steps-per-epoch", type=int, default=None)
    g.add_argument("--no-image-balanced", dest="image_balanced", action="store_false", default=None)
    g.add_argument("--n-way", type=int, default=None)
    g.add_argument("--k-shot", type=int, default=None)
    g.add_argument("--n-query", type=int, default=None)
    g.add_argument("--val-support-max", type=int, default=None,
                   help="max train patches per class used as validation prototypes")

    g = ap.add_argument_group("classifier")
    g.add_argument("--train-source", choices=["train", "augmented_train"], default=None)
    g.add_argument("--sampler", choices=["shuffle", "balanced"], default=None)
    g.add_argument("--class-weights", choices=["none", "balanced"], default=None)
    g.add_argument("--label-smoothing", type=float, default=None)
    g.add_argument("--batch-size", type=int, default=None)

    g = ap.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--lr", type=float, default=None, help="sets both --lr-backbone and --lr-head")
    g.add_argument("--lr-backbone", type=float, default=None)
    g.add_argument("--lr-head", type=float, default=None)
    g.add_argument("--weight-decay", type=float, default=None)
    g.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default=None)
    g.add_argument("--schedule", choices=["cosine", "constant"], default=None)
    g.add_argument("--warmup-epochs", type=float, default=None)
    g.add_argument("--grad-clip", type=float, default=None)
    g.add_argument("--augment", dest="augment", action="store_true", default=None)
    g.add_argument("--no-augment", dest="augment", action="store_false")
    g.add_argument("--max-shift", type=int, default=None, help="random translation (pixels) during training")
    g.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    g.add_argument("--select-metric", default=None,
                   help="metric: val_head_macro_f1 | val_head_acc | val_feat_macro_f1 ...; classifier: val_acc | val_macro_f1")
    g.add_argument("--patience", type=int, default=None, help="early stopping (0 = off)")
    g.add_argument("--val-every", type=int, default=None)
    g.add_argument("--eval-batch-size", type=int, default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    ov = {k: v for k, v in vars(args).items() if k not in ("data_root", "out_dir", "run_name", "model_type", "lr")}
    if args.lr is not None:
        ov["lr_backbone"] = ov["lr_backbone"] if ov["lr_backbone"] is not None else args.lr
        ov["lr_head"] = ov["lr_head"] if ov["lr_head"] is not None else args.lr
    if ov.get("train_classes") is None:
        ov["train_classes"] = "base14"
    cfg = make_config(args.model_type, ov)
    run_dir, best = run_training(cfg, args.data_root, args.out_dir, args.run_name)
    print(f"\nEvaluate with:\n  python evaluate.py --data-root {args.data_root} --checkpoint {best}")
    return run_dir, best


if __name__ == "__main__":
    main()
