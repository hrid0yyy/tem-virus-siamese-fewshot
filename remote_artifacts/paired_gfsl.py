"""Paired statistics for the rare-class (frozen-feature) study written by gfsl_improve.py.

Uses predictions_<run>.npz (identical official test patches for every run) and the per-draw K-shot
values in gfsl_improve.json (identical support draws for every run and method, same seed).

  1. within a checkpoint: each classifier vs NCM (paired micrograph cluster bootstrap + exact McNemar);
  2. across checkpoints: Siamese run vs baseline run with the same classifier;
  3. K-shot: per-draw paired difference of rare mean-class accuracy (same support draws).

    python paired_gfsl.py --dir runs/gfsl_fp32/A runs/gfsl_fp32/B ... --siamese A ... --baseline B ... \n                          --out runs/paired_gfsl.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temfs.analysis import mcnemar_exact, paired_cluster_bootstrap  # noqa: E402
from temfs.metrics import mean_ci95  # noqa: E402
from temfs.utils import save_json  # noqa: E402


def load_preds(dirs, run):
    hits = [Path(d) / f"predictions_{run}.npz" for d in dirs if (Path(d) / f"predictions_{run}.npz").exists()]
    if len(hits) != 1:
        raise FileNotFoundError(f"predictions_{run}.npz found {len(hits)} times in {dirs}")
    z = np.load(hits[0], allow_pickle=False)
    return {k: z[k] for k in z.files}


def paired(y, pa, pb, groups, n_classes, n_boot, seed):
    b, c, p = mcnemar_exact(pa == y, pb == y)
    res = paired_cluster_bootstrap(y, pa, pb, list(groups), n_classes, n_boot=n_boot, seed=seed)
    res.update({"acc_a": float((pa == y).mean()), "acc_b": float((pb == y).mean()),
                "mcnemar_a_right_b_wrong": b, "mcnemar_a_wrong_b_right": c, "mcnemar_p": p})
    return res


def fmt(r):
    return (f"acc {100 * r['acc_diff']:+.2f} pts [{100 * r['acc_diff_ci95'][0]:+.2f}, {100 * r['acc_diff_ci95'][1]:+.2f}] | "
            f"macro-F1 {r['macro_f1_diff']:+.3f} [{r['macro_f1_diff_ci95'][0]:+.3f}, {r['macro_f1_diff_ci95'][1]:+.3f}] | "
            f"McNemar {r['mcnemar_a_right_b_wrong']}/{r['mcnemar_a_wrong_b_right']} p={r['mcnemar_p']:.3g}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", nargs="+", required=True, help="output folder(s) of gfsl_improve.py")
    ap.add_argument("--siamese", nargs="+", required=True, help="run names (folder names of the checkpoints)")
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--methods", default="ncm,maha,logreg,dc")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    methods = [m for m in a.methods.split(",") if m]
    G, envs, settings = {}, [], set()
    for d in a.dir:
        J = json.load(open(Path(d) / "gfsl_improve.json", encoding="utf-8"))
        envs.append(J.get("environment"))
        for k, v in J["checkpoints"].items():
            name = Path(k.replace("\\", "/")).parent.name
            if name in G:
                raise ValueError(f"{name} appears in more than one folder")
            G[name] = v
            settings.add(tuple(len(v["methods"]["ncm"]["kshot"][kk]["per_draw"]["novel_mean_class_acc"])
                               for kk in sorted(v["methods"]["ncm"]["kshot"], key=int)))
    if len(settings) != 1:  # paired K-shot needs the same number of draws (same seed -> same draws)
        raise ValueError(f"K-shot draw settings differ between folders: {settings}")
    runs = a.siamese + a.baseline
    P = {r: load_preds(a.dir, r) for r in runs}

    ref = P[runs[0]]
    for r in runs[1:]:  # every run must have been evaluated on the identical patches with identical labels
        if not (np.array_equal(P[r]["test_idx"], ref["test_idx"]) and np.array_equal(P[r]["y"], ref["y"])
                and np.array_equal(P[r]["classes"], ref["classes"])):
            raise ValueError(f"{r}: test patches / labels differ from {runs[0]}")
    y, groups, nc = ref["y"], ref["groups"], len(ref["classes"])
    out = {"n_patches": int(len(y)), "n_micrographs": int(len(set(groups.tolist()))), "n_boot": a.n_boot,
           "environments": envs, "within_checkpoint": {}, "siamese_vs_baseline": {}, "kshot": {}}

    print("== 1. each classifier vs NCM, same checkpoint (A = classifier, B = NCM)")
    for r in runs:
        for m in methods:
            if m == "ncm":
                continue
            res = paired(y, P[r][f"{m}_pred"], P[r]["ncm_pred"], groups, nc, a.n_boot, a.seed)
            out["within_checkpoint"][f"{r}|{m}_vs_ncm"] = res
            print(f"  {r:40s} {m:6s}: {fmt(res)}")

    print("== 2. Siamese vs baseline, same classifier (A = Siamese, B = baseline)")
    for s in a.siamese:
        for b in a.baseline:
            for m in methods:
                res = paired(y, P[s][f"{m}_pred"], P[b][f"{m}_pred"], groups, nc, a.n_boot, a.seed)
                out["siamese_vs_baseline"][f"{s}|{b}|{m}"] = res
                print(f"  {s} vs {b} [{m}]: {fmt(res)}")

    print("== 3. K-shot, rare mean-class accuracy, paired over identical support draws (A - B)")
    for k in sorted(G[runs[0]]["methods"]["ncm"]["kshot"].keys(), key=int):
        for m in methods:
            for s in a.siamese:
                for b in a.baseline:
                    va = np.array(G[s]["methods"][m]["kshot"][k]["per_draw"]["novel_mean_class_acc"])
                    vb = np.array(G[b]["methods"][m]["kshot"][k]["per_draw"]["novel_mean_class_acc"])
                    d = va - vb
                    res = {"mean_a": float(va.mean()), "mean_b": float(vb.mean()), "diff": mean_ci95(d),
                           "a_better": int((d > 0).sum()), "b_better": int((d < 0).sum()), "n_draws": int(len(d))}
                    out["kshot"][f"K{k}|{m}|{s}|{b}"] = res
                    print(f"  K={k} [{m}] {s} vs {b}: {100 * res['mean_a']:.1f} vs {100 * res['mean_b']:.1f} | "
                          f"diff {100 * res['diff']['mean']:+.1f} ± {100 * res['diff']['ci95']:.1f} | "
                          f"A better in {res['a_better']}/{res['n_draws']} draws")
            # DC vs NCM within each run (does calibration help at small K?)
            if m == "dc":
                for r in runs:
                    va = np.array(G[r]["methods"]["dc"]["kshot"][k]["per_draw"]["novel_mean_class_acc"])
                    vb = np.array(G[r]["methods"]["ncm"]["kshot"][k]["per_draw"]["novel_mean_class_acc"])
                    d = va - vb
                    out["kshot"][f"K{k}|dc_vs_ncm|{r}"] = {"diff": mean_ci95(d), "a_better": int((d > 0).sum()),
                                                            "n_draws": int(len(d))}
                    print(f"  K={k} dc vs ncm, {r}: diff {100 * d.mean():+.1f} ± {100 * mean_ci95(d)['ci95']:.1f} "
                          f"(dc better in {(d > 0).sum()}/{len(d)})")
    save_json(out, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
