"""Turn report_data.json into Markdown tables (every number comes from the result files).

    python make_report_tables.py report_data.json > report_tables.md
"""

from __future__ import annotations

import json
import statistics
import sys

NOVEL = ["Dengue", "Guanarito", "LCM", "Machupo", "Pseudocowpox", "Sapovirus", "TBE", "WestNile"]


def pct(x, d=2):
    return "n/a" if x is None else f"{100 * x:.{d}f}"


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def ci(c, key="accuracy", scale=100, d=1):
    if not c or key not in c:
        return ""
    return f"[{c[key]['low'] * scale:.{d}f}, {c[key]['high'] * scale:.{d}f}]"


def get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    D = json.load(open(sys.argv[1], encoding="utf-8"))
    R = D["runs"]
    out = []

    # ---------------- F1
    f1 = D["f1"]
    out.append("## T1. F1 - frozen ImageNet DenseNet201 (no training), 14 classes")
    out.append(table(["Read-out", "Validation acc (%)", "Test acc (%)", "Test macro-F1"], [
        ["Nearest prototype (cosine)", pct(f1["ncm_validation"]["accuracy"], 1), pct(f1["ncm_test"]["accuracy"], 1),
         f3(f1["ncm_test"]["macro_f1"])],
        ["Logistic-regression probe", pct(f1["linear_probe_validation"]["accuracy"], 1),
         pct(f1["linear_probe_test"]["accuracy"], 1), f3(f1["linear_probe_test"]["macro_f1"])],
        ["Chance", "", pct(f1["chance"], 1), ""]]))

    # ---------------- F2 / F3 episodic
    for tag in ("f2", "f3"):
        f = D[tag]
        if not f:
            continue
        rows = []
        for k, v in f["by_k"].items():
            d = v["paired_diff_trained_feat_vs_frozen"]
            rows.append([f"{len(f['hidden'])}-way {k}-shot", pct(v["frozen_imagenet"]["acc"]["mean"], 1),
                         pct(v["trained_feat"]["acc"]["mean"], 1), pct(v["trained_head"]["acc"]["mean"], 1),
                         f"{100 * d['mean']:+.1f} ± {100 * d['ci95']:.1f}"])
        out.append(f"\n## T2{'a' if tag == 'f2' else 'b'}. {tag.upper()} - unseen classes {', '.join(f['hidden'])} "
                   f"({f['episodes']} image-disjoint episodes, {f['n_query']} queries/class)")
        out.append(table(["Task", "Frozen ImageNet (%)", "Trained backbone feat (%)", "Trained 128-d head (%)",
                          "Paired diff feat-frozen (pts, 95% CI)"], rows))
        out.append(f"Decision: **{f['verdict']['decision']}** (pre-defined rule at K={f['verdict']['k']}). "
                   f"Checkpoint selected on `{f['train']['select_metric']}`.")

    # ---------------- F3 GFSL
    t, fz = D.get("f3_gfsl_trained"), D.get("f3_gfsl_frozen")
    if t and fz:
        rows = []
        for name, blk in (("Frozen ImageNet (feat)", fz["feat"]), ("Trained (feat)", t["feat"]), ("Trained (head)", t["head"])):
            g = blk["gfsl"]["gfsl_patch"]
            rows.append([name, pct(g["base_acc"], 1), pct(g["novel_acc"], 1), f3(g["hm_acc"]),
                         pct(blk["kshot"]["5"]["novel_acc"]["mean"], 1), f3(blk["openset"]["patch"]["auroc"])])
        out.append("\n## T3. F3 generalised test - 10 trained classes + 4 hidden look-alike classes, official test split")
        out.append(table(["Embedding", "Trained-class acc (%)", "Hidden-class acc (%)", "HM",
                          "Hidden acc, 5 examples (%)", "Open-set AUROC"], rows))

    # ---------------- E1 14-class
    rows = []
    s0 = R["E1_siamese_supcon_densenet201"]
    seeds = [R[n] for n in ("E1_siamese_supcon_densenet201", "E1_siamese_supcon_densenet201_seed1",
                            "E1_siamese_supcon_densenet201_seed2") if R[n].get("eval")]
    p = R["E1_paper_densenet201"]
    rows.append(["Paper, published (Matuszewski & Sintorn 2021)", "93.10", "0.921", "", "no TTA"])
    if p.get("eval"):
        c = p["eval"]["logits"]["closed"]
        rows.append(["Paper model re-trained here (same evaluation)", pct(c["accuracy"]), f3(c["macro_f1"]),
                     ci(c.get("ci95")), f"best epoch {p['train']['best_epoch']}"])
    for i, s in enumerate(seeds):
        for key, label in (("feat", "backbone feat (pre-specified)"), ("head", "128-d head")):
            c = s["eval"][key]["closed"]
            rows.append([f"Siamese SupCon seed {s['config']['seed']}, {label}", pct(c["accuracy"]), f3(c["macro_f1"]),
                         ci(c.get("ci95")), f"best epoch {s['train']['best_epoch']}"])
    out.append("\n## T4. E1 - 14 classes, official test split (1,900 patches); 8-view TTA unless stated")
    out.append(table(["Model", "Accuracy (%)", "Macro-F1", "Acc 95% CI (micrograph bootstrap)", "Note"], rows))
    if len(seeds) > 1:
        for key in ("feat", "head"):
            a = [s["eval"][key]["closed"]["accuracy"] for s in seeds]
            f = [s["eval"][key]["closed"]["macro_f1"] for s in seeds]
            out.append(f"- Siamese {key}, {len(seeds)} seeds: accuracy {100 * statistics.mean(a):.2f} ± "
                       f"{100 * statistics.stdev(a):.2f} %, macro-F1 {statistics.mean(f):.3f} ± {statistics.stdev(f):.3f} (mean ± SD)")

    # ---------------- paired comparisons
    if D["compare"]:
        rows = []
        for name, c in D["compare"].items():
            rows.append([name, pct(c["acc_a"]), pct(c["acc_b"]), f"{100 * c['acc_diff']:+.2f}",
                         f"[{100 * c['acc_diff_ci95'][0]:+.2f}, {100 * c['acc_diff_ci95'][1]:+.2f}]",
                         f"{c['macro_f1_diff']:+.3f}", f"{c['mcnemar_p']:.3g}", c["n_patches"], c["n_micrographs"]])
        out.append("\n## T5. Paired comparisons on identical test patches")
        out.append(table(["Comparison (A vs B)", "Acc A (%)", "Acc B (%)", "Diff (pts)", "95% CI cluster bootstrap",
                          "Macro-F1 diff", "McNemar p", "Patches", "Micrographs"], rows))

    # ---------------- E2 22-class
    rows = []
    ce = R["E2_ce22_densenet201"]
    if ce.get("eval"):
        c = ce["eval"]["logits"]["closed"]
        g = c["gfsl_patch"]
        rows.append(["CE trained on all 22 classes (rare classes seen in training)", pct(c["accuracy"]), f3(c["macro_f1"]),
                     ci(c.get("ci95")), pct(g["base_acc"], 1), pct(g["novel_mean_class_acc"], 1)])
    for n, label in (("E1_paper_densenet201", "Paper model features, rare classes as prototypes"),):
        if R[n].get("eval"):
            g = R[n]["eval"]["feat"]["gfsl"]
            rows.append([label, pct(g["accuracy"]), f3(g["macro_f1"]), ci(g.get("ci95")),
                         pct(g["gfsl_patch"]["base_acc"], 1), pct(g["gfsl_patch"]["novel_mean_class_acc"], 1)])
    for s in seeds:
        for key in ("feat", "head"):
            g = s["eval"][key]["gfsl"]
            rows.append([f"Siamese seed {s['config']['seed']} ({key}), rare classes as prototypes", pct(g["accuracy"]),
                         f3(g["macro_f1"]), ci(g.get("ci95")), pct(g["gfsl_patch"]["base_acc"], 1),
                         pct(g["gfsl_patch"]["novel_mean_class_acc"], 1)])
    out.append("\n## T6. E2 - all 22 classes, official test split (2,057 patches)")
    out.append(table(["Model", "Accuracy (%)", "Macro-F1", "Acc 95% CI", "14-class acc (%)", "Rare mean-class acc (%)"], rows))
    if len(seeds) > 1:
        a = [s["eval"]["feat"]["gfsl"]["accuracy"] for s in seeds]
        f = [s["eval"]["feat"]["gfsl"]["macro_f1"] for s in seeds]
        out.append(f"- Siamese feat GFSL, {len(seeds)} seeds: accuracy {100 * statistics.mean(a):.2f} ± {100 * statistics.stdev(a):.2f} %, "
                   f"macro-F1 {statistics.mean(f):.3f} ± {statistics.stdev(f):.3f}")
    out.append("- Published 22-class results (different / unknown splits, all 22 classes trained): DMLF-Net 89.89 % acc "
               "(EAAI 2024); Hoque et al. 2026 preprint 88.31 % acc, macro-F1 0.853.")

    # ---------------- per-class 22 (siamese seed0 feat vs CE22)
    if s0.get("eval") and ce.get("eval"):
        g = s0["eval"]["feat"]["gfsl"]
        c = ce["eval"]["logits"]["closed"]
        names = g["class_names"]
        cn = c["class_names"]
        rows = []
        for i, nme in enumerate(names):
            j = cn.index(nme)
            rows.append([nme + (" *" if nme in NOVEL else ""), g["per_class"]["support"][i],
                         f3(g["per_class"]["f1"][i]), f3(c["per_class"]["f1"][j])])
        out.append("\n## T7. Per-class F1, 22 classes (* = rare class, never trained by the Siamese model)")
        out.append(table(["Class", "Test patches", "Siamese seed 0 feat (F1)", "CE on all 22 (F1)"], rows))

    # ---------------- K-shot
    if s0.get("eval"):
        rows = []
        for k in ("1", "3", "5", "10"):
            row = [k]
            for n, key in (("E1_siamese_supcon_densenet201", "feat"), ("E1_paper_densenet201", "feat"),
                           ("E1_siamese_supcon_densenet201", "head")):
                ks = get(R[n], "eval", key, "kshot", k)
                row += [f"{pct(ks['novel_acc']['mean'], 1)} ± {pct(ks['novel_acc']['ci95'], 1)}" if ks else "n/a",
                        f3(ks["hm_acc"]["mean"]) if ks else "n/a"]
            rows.append(row)
        out.append("\n## T8. E3 - rare classes with K examples each (100 repeats); 14 classes keep all training patches")
        out.append(table(["K", "Siamese feat: rare acc (%)", "HM", "Paper-model feat: rare acc (%)", "HM",
                          "Siamese head: rare acc (%)", "HM"], rows))

    # ---------------- open set + episodic
    rows = []
    for n, key, label in (("E1_siamese_supcon_densenet201", "feat", "Siamese feat"),
                          ("E1_siamese_supcon_densenet201", "head", "Siamese head"),
                          ("E1_paper_densenet201", "feat", "Paper-model feat")):
        e = get(R[n], "eval", key)
        if e and "openset" in e:
            rows.append([label, f3(e["openset"]["patch"]["auroc"]), pct(e["openset"]["patch"]["fpr_at_95tpr"], 1),
                         f3(e["openset"]["image"]["auroc"]),
                         pct(get(e, "episodic", "1", "acc", "mean"), 1), pct(get(e, "episodic", "5", "acc", "mean"), 1)])
    out.append("\n## T9. E4/E5 - 8-way image-disjoint episodes on the rare classes, and unknown-virus detection")
    out.append(table(["Embedding", "Open-set AUROC (patch)", "FPR@95%TPR (%)", "AUROC (micrograph)",
                      "8-way 1-shot acc (%)", "8-way 5-shot acc (%)"], rows))

    # ---------------- inference-time variants
    rows = []
    for n in ("E1_siamese_supcon_densenet201", "E1_siamese_supcon_densenet201_seed1", "E1_siamese_supcon_densenet201_seed2",
              "E1_paper_densenet201", "E6_supcon_auxce05_densenet201", "E6_triplet_densenet201"):
        v = get(R[n], "analysis", "variants")
        if not v:
            continue
        for var in ("feat_l2", "feat_cl2n"):
            if var not in v:
                continue
            x = v[var]
            raw, cal = x.get("gfsl_test_raw"), x.get("gfsl_test_calibrated")
            rows.append([n, var, pct(x["closed_test"]["accuracy"]), pct(raw["accuracy"]) if raw else "",
                         f3(raw["macro_f1"]) if raw else "", f"{x['gamma_selected_on_val']:+.3f}" if raw else "",
                         pct(cal["accuracy"]) if cal else "", f3(cal["macro_f1"]) if cal else ""])
    out.append("\n## T10. Inference-time variants (CL2N centring; calibrated stacking with gamma chosen on VALIDATION)")
    out.append(table(["Run", "Variant", "14-class acc (%)", "22-class acc raw (%)", "22 macro-F1 raw", "gamma (val)",
                      "22-class acc calibrated (%)", "22 macro-F1 calibrated"], rows))

    # ---------------- ablations
    rows = []
    for n, label in (("E1_siamese_supcon_densenet201", "SupCon (main, seed 0)"),
                     ("E6_supcon_auxce05_densenet201", "SupCon + 0.5 cosine-CE (Baseline++ term)"),
                     ("E6_triplet_densenet201", "Batch-hard triplet")):
        r = R[n]
        if not r.get("eval"):
            rows.append([label, "not run", "", "", "", ""])
            continue
        c, g = r["eval"]["feat"]["closed"], r["eval"]["feat"]["gfsl"]
        rows.append([label, pct(c["accuracy"]), f3(c["macro_f1"]), pct(g["accuracy"]), f3(g["macro_f1"]),
                     pct(g["gfsl_patch"]["novel_mean_class_acc"], 1)])
    out.append("\n## T11. E6 - loss ablations (backbone feat, TTA)")
    out.append(table(["Training loss", "14-class acc (%)", "14 macro-F1", "22-class acc (%)", "22 macro-F1",
                      "Rare mean-class acc (%)"], rows))

    # ---------------- training cost
    rows = [[n, r["train"]["epochs_run"], r["train"]["best_epoch"], r["train"]["select_metric"],
             f"{r['train']['total_train_time_s'] / 60:.0f}"] for n, r in R.items() if r.get("train")]
    out.append("\n## T12. Training runs (RTX 3060)")
    out.append(table(["Run", "Epochs run", "Best epoch", "Selection metric (validation)", "Train time (min)"], rows))
    print("\n".join(out))


if __name__ == "__main__":
    main()
