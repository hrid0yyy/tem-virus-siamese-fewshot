"""F1 - frozen-feature probe (no training).

Uses an ImageNet-pretrained backbone as-is and asks: do its features already separate the
14 base classes? Two read-outs on the OFFICIAL test split:
  * nearest-prototype (NCM, cosine)   - prototypes = mean feature of the official train split
  * logistic-regression linear probe  - trained on the official train split
Chance level is 1/14 = 7.1 %.

    python f1_frozen_probe.py --data-root data
    python f1_frozen_probe.py --data-root data --backbone resnet50 --tta
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temfs.constants import SPLITS, resolve_classes  # noqa: E402
from temfs.data import select_records  # noqa: E402
from temfs.fewshot import closed_set_ncm  # noqa: E402
from temfs.metrics import classification_scores  # noqa: E402
from temfs.pipeline import build_bank, frozen_imagenet_model, prepare_data  # noqa: E402
from temfs.utils import Logger, environment_info, get_device, save_json, seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--backbone", default="densenet201")
    ap.add_argument("--classes", default="base14")
    ap.add_argument("--tta", action="store_true", help="average features over the 8 D4 views")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-preload", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = Logger(out_dir / f"f1_{args.backbone}_{stamp}.log")
    device = get_device(args.device)
    log(f"Environment: {environment_info()}")
    classes = resolve_classes(args.classes)
    records, store = prepare_data(args.data_root, log, preload=not args.no_preload, skip_verify=args.skip_verify)
    recs = select_records(records, SPLITS, classes)
    model = frozen_imagenet_model(args.backbone, device)
    t0 = time.time()
    bank = build_bank(model, recs, store, device, tta=args.tta, batch_size=args.batch_size, keys=("feat",))
    log(f"Extracted {len(recs)} frozen {args.backbone} features in {time.time() - t0:.0f}s (TTA={args.tta})")

    res = {"backbone": args.backbone, "classes": classes, "tta": args.tta, "chance": 1.0 / len(classes)}
    for split in ("validation", "test"):
        r = closed_set_ncm(bank, "feat", classes, query_split=split, n_boot=1000 if split == "test" else 0,
                           seed=args.seed)
        res[f"ncm_{split}"] = r
        log(f"NCM  {split:10s}: acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} "
            f"| image-level acc={r['image_level']['accuracy']:.4f}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tr = bank.mask("train", classes)
    xtr, ytr = bank.emb["feat"][tr], bank.labels(tr, classes)
    scaler = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(scaler.transform(xtr), ytr)
    for split in ("validation", "test"):
        m = bank.mask(split, classes)
        pred = clf.predict(scaler.transform(bank.emb["feat"][m]))
        r = classification_scores(bank.labels(m, classes), pred, len(classes))
        res[f"linear_probe_{split}"] = r
        log(f"LinProbe {split:10s}: acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f}")

    ncm, lp = res["ncm_test"]["accuracy"], res["linear_probe_test"]["accuracy"]
    log("")
    log(f"Chance = {res['chance']:.3f} | NCM test acc = {ncm:.3f} | linear probe test acc = {lp:.3f} "
        f"| NCM/probe ratio = {ncm / max(lp, 1e-9):.2f}")
    log("Reading: NCM far above chance means the pretrained feature space already carries class "
        "structure (nearest-prototype classification is viable). A large NCM-vs-probe gap means the "
        "classes are linearly separable but not clustered - metric fine-tuning should help.")
    per_class = dict(zip(classes, res["ncm_test"]["per_class"]["recall"]))
    log("NCM per-class test recall: " + ", ".join(f"{c}={v:.2f}" for c, v in per_class.items()))
    save_json(res, out_dir / f"f1_{args.backbone}_{stamp}.json")
    log(f"Saved {out_dir / f'f1_{args.backbone}_{stamp}.json'}")


if __name__ == "__main__":
    main()
