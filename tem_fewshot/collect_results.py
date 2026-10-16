"""Collect every result file of the project into one JSON (input for the final report).

    python collect_results.py --runs runs --reports reports --out report_data.json

Missing files are recorded as null, never guessed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def jload(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def latest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def strip_closed(r):
    """Keep the scalar metrics of an evaluate.py block (drop confusion matrices)."""
    if not r:
        return None
    keep = {k: r[k] for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "balanced_accuracy", "n")
            if k in r}
    if "bootstrap_ci95" in r:
        keep["ci95"] = r["bootstrap_ci95"]
    if "image_level" in r:
        keep["image_level"] = {k: r["image_level"][k] for k in ("accuracy", "macro_f1", "n")}
    for k in ("gfsl_patch", "gfsl_image", "novel_only_acc", "base_only_acc", "class_names"):
        if k in r:
            keep[k] = r[k]
    if "per_class" in r:
        keep["per_class"] = r["per_class"]
    if "confusion" in r:
        keep["confusion"] = r["confusion"]
    return keep


def run_block(run_dir: Path):
    cfg = jload(run_dir / "config.json")
    summ = jload(run_dir / "train_summary.json")
    ev = jload(run_dir / "eval_full" / "results.json")
    an = jload(run_dir / "analysis" / "analysis.json")
    out = {"exists": run_dir.exists(), "finished": summ is not None}
    if cfg:
        out["config"] = {k: cfg.get(k) for k in (
            "model_type", "backbone", "loss", "ce_head", "train_classes", "train_source", "epochs", "lr_backbone",
            "lr_head", "optimizer", "batch_size", "P", "K", "temperature", "aux_ce_weight", "select_metric",
            "patience", "seed", "augment", "max_shift", "sampler", "class_weights")}
    if summ:
        hist = summ.get("history", [])
        out["train"] = {"best_epoch": summ.get("best_epoch"), "best_metric": summ.get("best_metric"),
                        "select_metric": summ.get("select_metric"), "epochs_run": len(hist),
                        "total_train_time_s": round(sum(h.get("train_time_s", 0) for h in hist), 1)}
    if ev:
        e = {}
        for key in ("head", "feat"):
            if key in ev:
                blk = {}
                for proto in ("closed", "gfsl"):
                    if proto in ev[key]:
                        blk[proto] = strip_closed(ev[key][proto])
                for proto in ("kshot", "episodic", "openset"):
                    if proto in ev[key]:
                        blk[proto] = ev[key][proto]
                e[key] = blk
        if "logits" in ev:
            e["logits"] = {"closed": strip_closed(ev["logits"]["closed"])}
        e["tta"] = ev.get("eval_args", {}).get("tta")
        out["eval"] = e
    if an:
        out["analysis"] = {"tta": an.get("tta"), "variants": an.get("variants")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="report_data.json")
    a = ap.parse_args()
    runs, reps = Path(a.runs), Path(a.reports)
    data = {"f0": None, "f1": None, "f2": None, "f3": None, "f3_gfsl_trained": None, "f3_gfsl_frozen": None,
            "runs": {}, "compare": {}}
    f0 = jload(reps / "f0_report.json")
    if f0:
        data["f0"] = {"problems": f0.get("problems"), "n_cross_split_duplicates": len(f0.get("cross_split_duplicates", [])),
                      "n_bad_images": len(f0.get("bad_images", [])), "counts": f0.get("counts")}
    f1 = latest(str(reps / "f1_densenet201_*.json"))
    if f1:
        j = jload(f1)
        data["f1"] = {"file": f1, "chance": j["chance"], "tta": j["tta"],
                      **{k: {m: j[k][m] for m in ("accuracy", "macro_f1")} for k in
                         ("ncm_validation", "ncm_test", "linear_probe_validation", "linear_probe_test")},
                      "ncm_test_per_class_recall": dict(zip(j["classes"], j["ncm_test"]["per_class"]["recall"]))}
    for tag, pat in (("f2", "*_f2_pseudonovel_*"), ("f3", "*_f3_pseudonovel_*")):
        d = latest(str(runs / pat))
        if d:
            data[tag] = jload(Path(d) / "f2_results.json")
            if data[tag] is not None:
                data[tag]["run_dir"] = d
                data[tag]["train"] = run_block(Path(d)).get("train")
            if tag == "f3":
                g = jload(Path(d) / "eval_gfsl" / "results.json")
                if g:
                    data["f3_gfsl_trained"] = {k: {p: (strip_closed(g[k][p]) if p == "gfsl" else g[k][p])
                                                   for p in ("gfsl", "kshot", "openset") if p in g[k]}
                                               for k in ("head", "feat") if k in g}
    fr = latest(str(reps / "f3_frozen" / "*" / "results.json"))
    if fr:
        g = jload(fr)
        data["f3_gfsl_frozen"] = {"feat": {p: (strip_closed(g["feat"][p]) if p == "gfsl" else g["feat"][p])
                                           for p in ("gfsl", "kshot", "openset") if p in g["feat"]}}
    for name in ("E1_siamese_supcon_densenet201", "E1_siamese_supcon_densenet201_seed1",
                 "E1_siamese_supcon_densenet201_seed2", "E1_paper_densenet201", "E2_ce22_densenet201",
                 "E6_supcon_auxce05_densenet201", "E6_triplet_densenet201"):
        data["runs"][name] = run_block(runs / name)
    for f in sorted(runs.glob("cmp_*.json")):
        data["compare"][f.stem] = jload(f)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print(f"wrote {a.out}")
    for name, r in data["runs"].items():
        print(f"  {name:42s} finished={r['finished']} eval={'eval' in r} analysis={'analysis' in r}")
    print("  compares:", list(data["compare"]))


if __name__ == "__main__":
    main()
