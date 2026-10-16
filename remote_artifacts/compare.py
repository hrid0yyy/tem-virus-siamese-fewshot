"""Paired statistical comparison of two models on the identical official test patches.

Uses predictions.npz written by analyze.py. Reports the accuracy / macro-F1 difference with a
95% CI from a stratified micrograph (cluster) bootstrap, and an exact McNemar test.

    python compare.py --a runs/A/analysis/predictions.npz --a-key feat_l2_closed \
                      --b runs/B/analysis/predictions.npz --b-key logits_closed
Keys: <emb>_<l2|cl2n>_closed, <emb>_<l2|cl2n>_gfsl_raw, <emb>_<l2|cl2n>_gfsl_calibrated, logits_closed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temfs.analysis import mcnemar_exact, paired_cluster_bootstrap  # noqa: E402
from temfs.utils import save_json  # noqa: E402


def load(npz_path, key):
    d = np.load(npz_path, allow_pickle=False)
    paths = d["paths"]
    if key.endswith("_closed"):
        idx, pred, y = d[f"{key}_idx"], d[f"{key}_pred"], d[f"{key}_y"]
    else:  # gfsl_raw / gfsl_calibrated
        stem = key.rsplit("_gfsl_", 1)[0]
        idx, pred, y = d[f"{stem}_gfsl_idx"], d[f"{key}_pred"], d[f"{stem}_gfsl_y"]
    return paths[idx], pred, y


def group_of(path):
    """Micrograph key from a patch path: <split>/<class>/<image_id>."""
    p = Path(str(path))
    split_cls = [x for x in p.parts[-4:-1] if x != "_EXCLUDED"]
    return f"{split_cls[-2]}/{split_cls[-1]}/{p.name.split('_')[0]}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-key", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-key", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    pa, ya_pred, ya = load(args.a, args.a_key)
    pb, yb_pred, yb = load(args.b, args.b_key)
    order_a = {str(p): i for i, p in enumerate(pa)}
    common = [str(p) for p in pb if str(p) in order_a]
    if len(common) != len(pa) or len(common) != len(pb):
        raise ValueError(f"patch sets differ: A={len(pa)} B={len(pb)} common={len(common)}")
    ia = np.array([order_a[p] for p in common])
    ib = np.arange(len(pb))
    y = ya[ia]
    if not np.array_equal(y, yb[ib]):
        raise ValueError("label spaces differ between A and B")
    pred_a, pred_b = ya_pred[ia], yb_pred[ib]
    groups = [group_of(p) for p in common]
    n_classes = int(max(y.max(), pred_a.max(), pred_b.max()) + 1)
    bc, cb, p = mcnemar_exact(pred_a == y, pred_b == y)
    boot = paired_cluster_bootstrap(y, pred_a, pred_b, groups, n_classes, n_boot=args.n_boot)
    res = {"a": [args.a, args.a_key], "b": [args.b, args.b_key], "n_patches": len(y),
           "n_micrographs": len(set(groups)), "acc_a": float((pred_a == y).mean()),
           "acc_b": float((pred_b == y).mean()), "mcnemar_a_right_b_wrong": bc,
           "mcnemar_a_wrong_b_right": cb, "mcnemar_p": p, **boot}
    print(f"A={args.a_key} acc={res['acc_a']:.4f} | B={args.b_key} acc={res['acc_b']:.4f} | n={len(y)} patches, "
          f"{res['n_micrographs']} micrographs")
    print(f"acc diff A-B = {boot['acc_diff']:+.4f}  95% CI [{boot['acc_diff_ci95'][0]:+.4f}, {boot['acc_diff_ci95'][1]:+.4f}]"
          f"  (bootstrap P(diff<=0)={boot['acc_diff_p_le0']:.3f})")
    print(f"macro-F1 diff = {boot['macro_f1_diff']:+.4f}  95% CI [{boot['macro_f1_diff_ci95'][0]:+.4f}, "
          f"{boot['macro_f1_diff_ci95'][1]:+.4f}]  (P(diff<=0)={boot['macro_f1_diff_p_le0']:.3f})")
    print(f"McNemar: A right/B wrong={bc}, A wrong/B right={cb}, exact p={p:.4g}")
    if args.out:
        save_json(res, args.out)
    return res


if __name__ == "__main__":
    main()
