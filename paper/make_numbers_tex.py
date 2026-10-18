"""Write paper/numbers.tex: every number and table body used by main.tex, read from the raw result files.

    python make_numbers_tex.py --mode amp     # round-1 outputs (mixed-precision inference), available now
    python make_numbers_tex.py --mode fp32    # round-2 outputs (fp32 deterministic inference) - final numbers

Nothing in main.tex is typed by hand: re-running this script updates every number in the paper.
Missing inputs (e.g. paired tests not run yet) produce a visible red PENDING marker instead of a number.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "remote_artifacts"
RUNS = ART / "runs"
BASE = ["Adenovirus", "Astrovirus", "CCHF", "Cowpox", "Ebola", "Influenza", "Lassa", "Marburg", "Nipah virus",
        "Norovirus", "Orf", "Papilloma", "Rift Valley", "Rotavirus"]
NOVEL = ["Dengue", "Guanarito", "LCM", "Machupo", "Pseudocowpox", "Sapovirus", "TBE", "WestNile"]
SIAM = ["E1_siamese_supcon_densenet201", "E1_siamese_supcon_densenet201_seed1", "E1_siamese_supcon_densenet201_seed2"]
B1 = ["E1_paper_densenet201", "E1_paper_densenet201_seed1", "E1_paper_densenet201_seed2"]
AUX, TRI, CE = "E6_supcon_auxce05_densenet201", "E6_triplet_densenet201", "E2_ce22_densenet201"
METHODS = [("ncm", "Nearest class mean"), ("logreg", "Logistic regression"),
           ("maha", "Shrunk Mahalanobis"), ("dc", "Distribution calibration")]
WEAK = ["Dengue", "Sapovirus", "Guanarito", "LCM", "Machupo"]
PAIRS = ["Dengue|Sapovirus", "Dengue|TBE", "Dengue|Astrovirus", "Sapovirus|Norovirus", "Sapovirus|Astrovirus",
         "Guanarito|Lassa", "Guanarito|Machupo", "LCM|Machupo"]
PENDING = r"\textcolor{red}{PENDING}"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def msd(v, scale=1.0, d=3):
    """'m \\pm s' over seeds; a single value is printed without SD."""
    v = [scale * x for x in v]
    if not v:
        return PENDING
    if len(v) == 1:
        return f"{v[0]:.{d}f}"
    return f"{st.mean(v):.{d}f} $\\pm$ {st.stdev(v):.{d}f}"


def mean(v, scale=1.0, d=3):
    return f"{scale * st.mean(v):.{d}f}" if v else PENDING


def macro_specificity(conf):
    c = np.asarray(conf, dtype=np.float64)
    n = c.sum()
    spec = []
    for i in range(len(c)):
        tp = c[i, i]
        fn = c[i].sum() - tp
        fp = c[:, i].sum() - tp
        tn = n - tp - fn - fp
        spec.append(tn / (tn + fp) if tn + fp > 0 else np.nan)
    return float(np.nanmean(spec))


class Src:
    def __init__(self, mode):
        self.mode = mode
        self.eval_dir = "eval_full" if mode == "amp" else "eval_fp32"
        self.an_dir = "analysis" if mode == "amp" else "analysis_fp32"
        self.cmp_prefix = "cmp_" if mode == "amp" else "cmp_fp32_"

    def eval(self, run):
        p = RUNS / run / self.eval_dir / "results.json"
        return load(p) if p.exists() else None

    def analysis(self, run):
        p = RUNS / run / self.an_dir / "analysis.json"
        return load(p) if p.exists() else None

    def gfsl(self, run):
        if self.mode == "amp":
            p = RUNS / "gfsl_improve" / "gfsl_improve.json"
            if not p.exists():
                return None
            for k, v in load(p)["checkpoints"].items():
                if Path(k.replace("\\", "/")).parent.name == run:
                    return v
            return None
        p = RUNS / "gfsl_fp32" / run / "gfsl_improve.json"
        if not p.exists():
            return None
        return next(iter(load(p)["checkpoints"].values()))

    def cmp(self, name):
        p = RUNS / f"{self.cmp_prefix}{name}.json"
        return load(p) if p.exists() else None

    def paired(self):
        for n in ("paired_gfsl_final", "paired_gfsl_phase1"):
            p = RUNS / f"{n}.json"
            if self.mode == "fp32" and p.exists():
                return n, load(p)
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["amp", "fp32"], required=True)
    ap.add_argument("--out", default=str(HERE / "numbers.tex"))
    ap.add_argument("--dry-run-amp-eval", action="store_true",
                    help="testing only: fp32 classifier study + paired tests, but round-1 evaluate/analyze/compare files")
    a = ap.parse_args()
    S = Src(a.mode)
    if a.dry_run_amp_eval:
        S.eval_dir, S.an_dir, S.cmp_prefix = "eval_full", "analysis", "cmp_"
    M = {}  # macro name -> value
    T = {}  # table-body macro name -> rows

    def put(name, value):
        assert name.isalpha(), name
        M[name] = value
        if "$\\pm$" in value:  # mean-only variant for compact sentences (e.g. the abstract)
            M[name + "Mean"] = value.split(" $\\pm$")[0]

    siam = [r for r in SIAM if S.eval(r)]
    b1 = [r for r in B1 if S.eval(r)]
    put("NSiamSeeds", str(len(siam)))
    put("NBOneSeeds", str(len(b1)))
    put("InferenceMode", "mixed-precision (fp16 autocast)" if a.mode == "amp" else "full-precision (fp32), deterministic")

    # ------------------------------------------------------------------ training configuration table (from config.json)
    def cfgs(runs):
        cs = [load(RUNS / r / "config.json") for r in runs]
        strip = lambda c: {k: v for k, v in c.items() if k not in ("seed", "run_dir", "environment", "resume")}
        assert all(strip(c) == strip(cs[0]) for c in cs), f"configs differ between seeds of {runs[0]}"
        return cs[0]

    sc, bc = cfgs(SIAM), cfgs(B1)
    assert sc["backbone"] == bc["backbone"] == "densenet201" and sc["pretrained"] and bc["pretrained"]
    assert bc["ce_head"] == "paper" and sc["image_balanced"]
    sci = lambda x: f"${x:.0e}".replace("e-0", r"\times10^{-") + "}$"
    T["TabConfigRows"] = [
        r"Backbone & DenseNet201, ImageNet-pretrained & DenseNet201, ImageNet-pretrained",
        f"Head & projection MLP 1,920$\\to${sc['hidden_dim']}$\\to${sc['emb_dim']}, L2-normalised & 2$\\times$FC 1,024, dropout {bc['dropout']}, softmax",
        f"Loss & supervised contrastive, $\\tau={sc['temperature']}$ & cross-entropy",
        f"Training data & official training split & pre-augmented training split",
        f"Batch & {sc['P']} classes $\\times$ {sc['K']} patches (different micrographs) & {bc['batch_size']} patches",
        f"Optimiser & AdamW & Adam",
        f"Learning rate (backbone / head) & {sci(sc['lr_backbone'])} / {sci(sc['lr_head'])} & {sci(bc['lr_backbone'])} / {sci(bc['lr_head'])}",
        f"Weight decay & {sci(sc['weight_decay'])} & {bc['weight_decay']:g}",
        f"Schedule & {sc['warmup_epochs']} warm-up epochs, cosine decay & constant",
        f"Epochs & at most {sc['epochs']} ({sc['steps_per_epoch']} steps each), early stop after {sc['patience']} & {bc['epochs']} (fixed)",
        f"Gradient clipping & {sc['grad_clip']:g} & none",
        f"Online augmentation & D4 rotations/reflections, shift $\\pm${sc['max_shift']} px & none",
        f"Checkpoint selection & validation macro-F1 (nearest class mean) & validation accuracy",
    ]
    assert sc["select_metric"] == "val_feat_macro_f1" and bc["select_metric"] == "val_acc"
    assert sc["optimizer"] == "adamw" and bc["optimizer"] == "adam" and sc["schedule"] == "cosine" and bc["schedule"] == "constant"
    assert sc["train_source"] == "train" and bc["train_source"] == "augmented_train" and not bc["augment"] and bc["grad_clip"] == 0
    assert bc["patience"] == 0 and sc["loss"] == "supcon"

    # ------------------------------------------------------------------ dataset table (official counts)
    C = load(HERE.parent / "tem_fewshot" / "temfs" / "expected_counts.json")["counts"]
    rows = []
    tot = np.zeros(6, dtype=int)
    for c in BASE + NOVEL:
        e = C[c]
        v = [e[s][k] for s in ("train", "validation", "test") for k in ("patches", "images")]
        tot += np.array(v)
        name = c.replace("WestNile", "West Nile")
        rows.append(f"{name} & {'base' if c in BASE else 'rare'} & {v[0]:,} ({v[1]}) & {v[2]:,} ({v[3]}) & {v[4]:,} ({v[5]})")
    rows += [r"\midrule", f"Total & & {tot[0]:,} ({tot[1]}) & {tot[2]:,} ({tot[3]}) & {tot[4]:,} ({tot[5]})"]
    T["TabDatasetRows"] = rows

    # ------------------------------------------------------------------ 14-class closed set
    def closed(run, key):
        e = S.eval(run)
        if e is None:
            return None
        c = e[key]["closed"]
        return {"acc": c["accuracy"], "f1": c["macro_f1"], "p": c["macro_precision"], "r": c["macro_recall"],
                "spec": macro_specificity(c["confusion"]), "img": c["image_level"]["accuracy"]}

    def closed_row(label, runs, key):
        vals = [closed(r, key) for r in runs]
        vals = [v for v in vals if v]
        if not vals:
            return f"{label} & " + " & ".join([PENDING] * 6)
        g = lambda k, sc, d: msd([v[k] for v in vals], sc, d)
        return (f"{label} & {g('acc', 100, 2)} & {g('f1', 1, 3)} & {g('p', 1, 3)} & {g('r', 1, 3)} & "
                f"{g('spec', 1, 4)} & {g('img', 100, 2)}")

    T["TabClosedRows"] = [
        r"DenseNet201, published \cite{MS21}$^{a}$ & 93.10 & 0.921 & 0.926 & 0.921 & -- & --",
        closed_row(f"Baseline: benchmark model re-trained here ({len(b1)} seed{'s' if len(b1) > 1 else ''})", b1, "logits"),
        closed_row(f"Siamese SupCon, backbone feature ({len(siam)} seeds)", siam, "feat"),
        r"\midrule",
        closed_row("Ablation: batch-hard triplet (1 seed)", [TRI], "feat"),
        closed_row("Ablation: SupCon + cosine CE (1 seed)", [AUX], "feat"),
        closed_row("Siamese 128-d head (not pre-specified)", siam, "head"),
    ]
    sv = [closed(r, "feat") for r in siam]
    bv = [closed(r, "logits") for r in b1]
    put("SiamClosedAcc", msd([v["acc"] for v in sv], 100, 2))
    put("SiamClosedFOne", msd([v["f1"] for v in sv], 1, 3))
    put("BOneClosedAcc", msd([v["acc"] for v in bv], 100, 2))
    put("BOneClosedFOne", msd([v["f1"] for v in bv], 1, 3))
    put("SiamClosedSens", msd([v["r"] for v in sv], 1, 3))
    put("SiamClosedPrec", msd([v["p"] for v in sv], 1, 3))
    put("SiamClosedSpec", msd([v["spec"] for v in sv], 1, 4))
    hv = [closed(r, "head") for r in siam]
    put("SiamHeadClosedAcc", msd([v["acc"] for v in hv], 100, 2))

    for tag, name in (("Feat", "siamese_feat_vs_paper"), ("Head", "siamese_head_vs_paper"),
                      ("Gfsl", "gfsl_siamese_vs_paperfeat")):
        c = S.cmp(name)
        if c is None:
            for k in ("Diff", "CI", "FOneDiff", "FOneCI", "McN", "McP"):
                put(f"Cmp{tag}{k}", PENDING)
            continue
        put(f"Cmp{tag}Diff", f"{100 * c['acc_diff']:+.2f}")
        put(f"Cmp{tag}CI", f"[{100 * c['acc_diff_ci95'][0]:+.2f}, {100 * c['acc_diff_ci95'][1]:+.2f}]")
        put(f"Cmp{tag}FOneDiff", f"{c['macro_f1_diff']:+.3f}")
        put(f"Cmp{tag}FOneCI", f"[{c['macro_f1_diff_ci95'][0]:+.3f}, {c['macro_f1_diff_ci95'][1]:+.3f}]")
        put(f"Cmp{tag}McN", f"{c['mcnemar_a_right_b_wrong']} vs {c['mcnemar_a_wrong_b_right']}")
        put(f"Cmp{tag}McP", f"{c['mcnemar_p']:.2g}")

    # all Siamese x B1 seed pairs (paired cluster bootstrap on identical test patches; fp32 only)
    ps = RUNS / "paired_seeds_fp32_local.json"
    PS = load(ps) if (a.mode == "fp32" and ps.exists()) else None
    for tag, prefix in (("Closed", "14class_siamese_feat_vs_paper_logits"), ("Ncm", "22class_ncm_siamese_feat_vs_paper_feat")):
        if PS is None:
            for k in ("Range", "NPos", "NSig", "N", "FOneRange", "FOneNSig"):
                put(f"Seed{tag}{k}", PENDING)
            continue
        xs = [PS[f"{prefix}|{s}|{b}"] for s in siam for b in b1]
        put(f"Seed{tag}Range", f"{100 * min(x['acc_diff'] for x in xs):+.2f} to {100 * max(x['acc_diff'] for x in xs):+.2f}")
        put(f"Seed{tag}NPos", str(sum(1 for x in xs if x["acc_diff"] > 0)))
        put(f"Seed{tag}NSig", str(sum(1 for x in xs if x["acc_diff_ci95"][0] > 0 or x["acc_diff_ci95"][1] < 0)))
        put(f"Seed{tag}N", str(len(xs)))
        put(f"Seed{tag}FOneRange", f"{min(x['macro_f1_diff'] for x in xs):+.3f} to {max(x['macro_f1_diff'] for x in xs):+.3f}")
        put(f"Seed{tag}FOneNSig", str(sum(1 for x in xs if x["macro_f1_diff_ci95"][0] > 0 or x["macro_f1_diff_ci95"][1] < 0)))

    # ------------------------------------------------------------------ 22-class GFSL, nearest class mean
    def gf(run, key):
        e = S.eval(run)
        if e is None:
            return None
        g = e[key]["gfsl"]
        return {"acc": g["accuracy"], "f1": g["macro_f1"], "base": g["gfsl_patch"]["base_acc"],
                "rare": g["gfsl_patch"]["novel_mean_class_acc"], "img": g["image_level"]["accuracy"]}

    def gf_row(label, runs, key):
        vals = [v for v in (gf(r, key) for r in runs) if v]
        if not vals:
            return f"{label} & " + " & ".join([PENDING] * 4)
        g = lambda k, sc, d: msd([v[k] for v in vals], sc, d)
        return f"{label} & {g('acc', 100, 2)} & {g('f1', 1, 3)} & {g('base', 100, 1)} & {g('rare', 100, 1)}"

    ce = S.eval(CE)
    cec = ce["logits"]["closed"] if ce else None
    T["TabGfslRows"] = [
        gf_row(f"Siamese backbone feature ({len(siam)} seeds)", siam, "feat"),
        gf_row(f"Baseline backbone feature ({len(b1)} seed{'s' if len(b1) > 1 else ''})", b1, "feat"),
        gf_row("Siamese 128-d head", siam, "head"),
        gf_row("Ablation: batch-hard triplet", [TRI], "feat"),
        gf_row("Ablation: SupCon + cosine CE", [AUX], "feat"),
        r"\midrule",
        (r"\textit{Fully supervised reference, trained on all 22 classes} & " +
         (f"\\textit{{{100 * cec['accuracy']:.2f}}} & \\textit{{{cec['macro_f1']:.3f}}} & "
          f"\\textit{{{100 * cec['gfsl_patch']['base_acc']:.1f}}} & "
          f"\\textit{{{100 * cec['gfsl_patch']['novel_mean_class_acc']:.1f}}}" if cec else " & ".join([PENDING] * 4))),
    ]
    sg = [gf(r, "feat") for r in siam]
    bg = [gf(r, "feat") for r in b1]
    hg = [gf(r, "head") for r in siam]
    put("SiamGfslAcc", msd([v["acc"] for v in sg], 100, 2))
    put("SiamGfslFOne", msd([v["f1"] for v in sg], 1, 3))
    put("SiamGfslRare", msd([v["rare"] for v in sg], 100, 1))
    put("BOneGfslFOne", msd([v["f1"] for v in bg], 1, 3))
    put("BOneGfslAcc", msd([v["acc"] for v in bg], 100, 2))
    put("BOneGfslRare", msd([v["rare"] for v in bg], 100, 1))
    put("SiamGfslBase", msd([v["base"] for v in sg], 100, 1))
    put("BOneGfslBase", msd([v["base"] for v in bg], 100, 1))
    put("HeadRareMin", f"{100 * min(v['rare'] for v in hg):.0f}")
    put("HeadRareMax", f"{100 * max(v['rare'] for v in hg):.0f}")
    put("CeAcc", f"{100 * cec['accuracy']:.2f}" if cec else PENDING)
    put("CeFOne", f"{cec['macro_f1']:.3f}" if cec else PENDING)
    # Dengue -> Sapovirus confusions
    dsap = []
    for r in siam:
        g = S.eval(r)["feat"]["gfsl"]
        n = g["class_names"]
        dsap.append(g["confusion"][n.index("Dengue")][n.index("Sapovirus")])
    put("DengueToSapMin", str(min(dsap)))
    put("DengueToSapMax", str(max(dsap)))
    g0 = S.eval(siam[0])["feat"]["gfsl"]
    put("DengueTest", str(g0["per_class"]["support"][g0["class_names"].index("Dengue")]))

    # ------------------------------------------------------------------ K-shot curve (evaluate.py, NCM)
    rows = []
    for k in ("1", "3", "5", "10"):
        s = [S.eval(r)["feat"]["kshot"][k]["novel_mean_class_acc"]["mean"] for r in siam]
        b = [S.eval(r)["feat"]["kshot"][k]["novel_mean_class_acc"]["mean"] for r in b1]
        rows.append(f"{k} & {msd(s, 100, 1)} & {msd(b, 100, 1)}")
        if k in ("1", "5"):
            nm = "One" if k == "1" else "Five"
            put(f"KshotSiam{nm}", msd(s, 100, 1))
            put(f"KshotBOne{nm}", msd(b, 100, 1))
    T["TabKshotRows"] = rows
    put("KshotDraws", str(S.eval(siam[0])["feat"]["kshot"]["1"]["novel_mean_class_acc"]["n"]))

    # ------------------------------------------------------------------ episodes / open set (supplement)
    rows = []
    for label, runs in ((f"Siamese ({len(siam)} seeds)", siam), (f"Baseline ({len(b1)} seed{'s' if len(b1) > 1 else ''})", b1),
                        ("SupCon + cosine CE", [AUX]), ("Batch-hard triplet", [TRI])):
        e = [S.eval(r)["feat"] for r in runs]
        rows.append(f"{label} & {msd([x['episodic']['1']['acc']['mean'] for x in e], 100, 1)} & "
                    f"{msd([x['episodic']['5']['acc']['mean'] for x in e], 100, 1)} & "
                    f"{msd([x['openset']['patch']['auroc'] for x in e], 1, 3)} & "
                    f"{msd([x['openset']['patch']['fpr_at_95tpr'] for x in e], 100, 1)}")
    T["TabEpisodesRows"] = rows

    # ------------------------------------------------------------------ classifier study on frozen features
    G = {r: S.gfsl(r) for r in siam + b1 + [AUX]}
    gs = [r for r in siam if G[r]]
    gb = [r for r in b1 if G[r]]
    put("NGfslSiam", str(len(gs)))
    put("NGfslBOne", str(len(gb)))
    rows = []
    for label, runs in (("Siamese", gs), ("Baseline", gb)):
        for m, mlab in METHODS:
            if not runs:
                rows.append(f"{label} & {mlab} & " + " & ".join([PENDING] * 6))
                continue
            x = [G[r]["methods"][m] for r in runs]
            win = sum(1 for r in runs if max(G[r]["methods"], key=lambda q: G[r]["methods"][q]["val_macro_f1"]) == m)
            mark = f"$^{{{win}/{len(runs)}}}$" if win else ""
            rows.append(f"{label if m == 'ncm' else ''} & {mlab}{mark} & "
                        f"{msd([v['val_macro_f1'] for v in x], 1, 3)} & {msd([v['test']['accuracy'] for v in x], 100, 2)} & "
                        f"{msd([v['test']['macro_f1'] for v in x], 1, 3)} & "
                        f"{msd([v['test']['novel_mean_class_acc'] for v in x], 100, 1)} & "
                        f"{msd([v['kshot']['1']['novel_mean_class_acc']['mean'] for v in x], 100, 1)} & "
                        f"{msd([v['kshot']['5']['novel_mean_class_acc']['mean'] for v in x], 100, 1)}")
        rows.append(r"\midrule")
    T["TabClassifierRows"] = rows[:-1]
    allck = gs + gb + ([AUX] if G[AUX] else [])
    winner = {r: max(G[r]["methods"], key=lambda q: G[r]["methods"][q]["val_macro_f1"]) for r in allck}
    nlr = sum(1 for r in allck if winner[r] == "logreg")
    put("LrWinsN", str(nlr))
    put("NCheckpoints", str(len(allck)))
    names = {"ncm": "nearest class mean", "maha": "shrunk Mahalanobis", "dc": "distribution calibration"}
    pretty = lambda r: ("Siamese seed " + (r[-1] if r.endswith(("seed1", "seed2")) else "0")) if r in gs else \
        (("baseline seed " + (r[-1] if r.endswith(("seed1", "seed2")) else "0")) if r in gb else "SupCon + cosine CE")
    others = [f"{pretty(r)} ({names[winner[r]]} classifier)" for r in allck if winner[r] != "logreg"]
    put("LrWinExceptions", ("; " .join(others)) if others else "none")
    for m in ("logreg", "maha", "ncm", "dc"):
        for grp, runs in (("Siam", gs), ("BOne", gb)):
            put(f"Win{grp}{ {'logreg': 'Lr', 'maha': 'Maha', 'ncm': 'Ncm', 'dc': 'Dc'}[m] }",
                str(sum(1 for r in runs if winner[r] == m)))
    for m, nm in (("ncm", "Ncm"), ("logreg", "Lr"), ("maha", "Maha"), ("dc", "Dc")):
        put(f"Siam{nm}FOne", msd([G[r]["methods"][m]["test"]["macro_f1"] for r in gs], 1, 3))
        put(f"Siam{nm}Acc", msd([G[r]["methods"][m]["test"]["accuracy"] for r in gs], 100, 2))
        put(f"Siam{nm}Base", msd([G[r]["methods"][m]["test"]["base_acc"] for r in gs], 100, 1))
        put(f"Siam{nm}KOne", msd([G[r]["methods"][m]["kshot"]["1"]["novel_mean_class_acc"]["mean"] for r in gs], 100, 1))
        put(f"BOne{nm}FOne", msd([G[r]["methods"][m]["test"]["macro_f1"] for r in gb], 1, 3) if gb else PENDING)
        put(f"BOne{nm}KOne", msd([G[r]["methods"][m]["kshot"]["1"]["novel_mean_class_acc"]["mean"] for r in gb], 100, 1)
            if gb else PENDING)
    for c in WEAK:
        for m, nm in (("ncm", "Ncm"), ("logreg", "Lr")):
            put(f"Weak{c}{nm}", mean([G[r]["methods"][m]["test"]["weak_f1"][c] for r in gs], 1, 2))
    rows = []
    for label, runs in (("Siamese", gs), ("Baseline", gb)):
        for m, mlab in METHODS:
            if not runs:
                continue
            rows.append(f"{label}, {mlab} & " + " & ".join(
                mean([G[r]["methods"][m]["test"]["weak_f1"][c] for r in runs], 1, 2) for c in WEAK))
    T["TabWeakRows"] = rows
    # rare-class precision / recall with NCM vs LR (Siamese, mean over seeds of the macro over the 8 rare classes)
    cls22 = g0["class_names"]
    ridx = [cls22.index(c) for c in NOVEL]
    for m, nm in (("ncm", "Ncm"), ("logreg", "Lr")):
        for key, kn in (("per_class_precision", "Prec"), ("per_class_recall", "Rec")):
            vals = [sum(G[r]["methods"][m]["test"][key][i] for i in ridx) / len(ridx) for r in gs]
            put(f"Rare{kn}{nm}", mean(vals, 1, 2))
        for c, cn in (("Sapovirus", "Sapo"), ("Guanarito", "Guan")):
            put(f"Prec{cn}{nm}", mean([G[r]["methods"][m]["test"]["per_class_precision"][cls22.index(c)] for r in gs], 1, 2))
    # share of Dengue test patches predicted as Sapovirus (Siamese, pooled over seeds) and Dengue recall
    if a.mode == "fp32":
        for m, nm in (("ncm", "Ncm"), ("logreg", "Lr")):
            tot = hit = cor = 0
            for r in gs:
                z = np.load(RUNS / "gfsl_fp32" / r / f"predictions_{r}.npz", allow_pickle=False)
                cl = [str(x) for x in z["classes"]]
                yd = z["y"] == cl.index("Dengue")
                tot += int(yd.sum())
                hit += int((z[f"{m}_pred"][yd] == cl.index("Sapovirus")).sum())
                cor += int((z[f"{m}_pred"][yd] == cl.index("Dengue")).sum())
            put(f"DengueToSapPct{nm}", f"{100 * hit / tot:.0f}")
            put(f"DengueRecallPct{nm}", f"{100 * cor / tot:.0f}")
    sup = g0["per_class"]["support"]
    put("WeakSupport", ", ".join(f"{c} {sup[g0['class_names'].index(c)]}" for c in WEAK))
    # Mahalanobis selection stability: did the selected setting differ across seeds / near-ties on validation?
    gaps = []
    for r in gs + gb:
        curve = sorted((f for _, f in G[r]["methods"]["maha"]["val_curve"]), reverse=True)
        gaps.append(curve[0] - curve[1])
    put("MahaMinValGap", f"{min(gaps):.4f}")

    # separability (validation)
    rows = []
    for p in PAIRS:
        ds = [G[r]["diagnostic_validation"][p] for r in gs]
        db = [G[r]["diagnostic_validation"][p] for r in gb]
        n = ds[0]
        rows.append(f"{p.replace('|', ' vs ')} & {mean([d['auc_logreg'] for d in ds], 1, 3)} & "
                    f"{mean([d['acc_logreg'] for d in ds], 1, 2)} & {mean([d['acc_ncm'] for d in ds], 1, 2)} & "
                    f"{mean([d['auc_logreg'] for d in db], 1, 3) if db else PENDING} & {n['n_val_a']}/{n['n_val_b']}")
    T["TabSepRows"] = rows
    allauc = [G[r]["diagnostic_validation"][p]["auc_logreg"] for r in gs for p in PAIRS]
    put("SepAucMin", f"{min(allauc):.2f}")
    sa = [G[r]["diagnostic_validation"]["Sapovirus|Astrovirus"] for r in gs]
    put("SapAstNcmMin", f"{min(d['acc_ncm'] for d in sa):.2f}")
    put("SapAstNcmMax", f"{max(d['acc_ncm'] for d in sa):.2f}")
    put("SapAstLrMin", f"{min(d['acc_logreg'] for d in sa):.2f}")
    put("SapAstLrMax", f"{max(d['acc_logreg'] for d in sa):.2f}")

    # ------------------------------------------------------------------ paired tests of the classifier study
    pname, P = S.paired()
    put("PairedSource", pname or "none")

    def ci_f1(r):
        return f"{r['macro_f1_diff']:+.3f} [{r['macro_f1_diff_ci95'][0]:+.3f}, {r['macro_f1_diff_ci95'][1]:+.3f}]"

    if P:
        W = P["within_checkpoint"]
        lr = [W[f"{r}|logreg_vs_ncm"] for r in gs if f"{r}|logreg_vs_ncm" in W]
        put("PairLrNcmRange", f"{min(x['macro_f1_diff'] for x in lr):+.3f} to {max(x['macro_f1_diff'] for x in lr):+.3f}")
        put("PairLrNcmAllPos", "excludes zero for every Siamese seed" if all(x["macro_f1_diff_ci95"][0] > 0 for x in lr)
            else r"\textcolor{red}{does NOT exclude zero for every seed -- revise text}")
        put("PairLrNcmCIs", "; ".join(ci_f1(x) for x in lr))
        lra = [v for k, v in W.items() if k.endswith("|logreg_vs_ncm")]
        put("PairLrNcmAllN", str(len(lra)))
        put("PairLrNcmAllSig", str(sum(1 for x in lra if x["macro_f1_diff_ci95"][0] > 0)))
        put("PairLrNcmAllRange", f"{min(x['macro_f1_diff'] for x in lra):+.3f} to {max(x['macro_f1_diff'] for x in lra):+.3f}")
        X = P["siamese_vs_baseline"]
        sb = [X[f"{s}|{b}|logreg"] for s in gs for b in gb if f"{s}|{b}|logreg" in X]
        put("PairSiamBOneLrRange", f"{min(x['macro_f1_diff'] for x in sb):+.3f} to {max(x['macro_f1_diff'] for x in sb):+.3f}")
        put("PairSiamBOneLrNSig", str(sum(1 for x in sb if x["macro_f1_diff_ci95"][0] > 0 or x["macro_f1_diff_ci95"][1] < 0)))
        put("PairSiamBOneLrN", str(len(sb)))
        sig = [x for x in sb if x["macro_f1_diff_ci95"][0] > 0 or x["macro_f1_diff_ci95"][1] < 0]
        npos, nneg = sum(1 for x in sig if x["macro_f1_diff"] > 0), sum(1 for x in sig if x["macro_f1_diff"] < 0)
        put("PairSiamBOneLrSigDir", "none" if not sig else
            ", ".join(p for p in ((f"{nneg} in favour of the baseline" if nneg else ""), (f"{npos} in favour of the Siamese encoder" if npos else "")) if p))
        K = P["kshot"]
        k1 = [K[f"K1|ncm|{s}|{b}"] for s in gs for b in gb if f"K1|ncm|{s}|{b}" in K]
        put("PairKOneRange", f"{100 * min(x['diff']['mean'] for x in k1):+.1f} to {100 * max(x['diff']['mean'] for x in k1):+.1f}")
        put("PairKOneNSig", str(sum(1 for x in k1 if x["diff"]["mean"] - x["diff"]["ci95"] > 0)))
        put("PairKOneN", str(len(k1)))
        put("PairKOneDraws", str(k1[0]["n_draws"]))
        dc = [K[f"K1|dc_vs_ncm|{r}"] for r in gs + gb if f"K1|dc_vs_ncm|{r}" in K]
        put("PairDcRange", f"{100 * min(x['diff']['mean'] for x in dc):+.1f} to {100 * max(x['diff']['mean'] for x in dc):+.1f}")
        put("PairDcNSig", str(sum(1 for x in dc if x["diff"]["mean"] - x["diff"]["ci95"] > 0)))
        put("PairDcN", str(len(dc)))
    else:
        for k in ("PairLrNcmRange", "PairLrNcmAllPos", "PairLrNcmCIs", "PairLrNcmAllN", "PairLrNcmAllSig", "PairLrNcmAllRange", "PairSiamBOneLrRange", "PairSiamBOneLrNSig",
                  "PairSiamBOneLrN", "PairSiamBOneLrSigDir", "PairKOneRange", "PairKOneNSig", "PairKOneN", "PairKOneDraws", "PairDcRange",
                  "PairDcNSig", "PairDcN"):
            put(k, PENDING)

    # ------------------------------------------------------------------ pilots (static result files, same in both modes)
    D = load(ART / "report_data.json")
    f1 = D["f1"]
    put("PilotFrozenNcm", f"{100 * f1['ncm_test']['accuracy']:.1f}")
    put("PilotFrozenProbe", f"{100 * f1['linear_probe_test']['accuracy']:.1f}")
    rows = []
    for tag in ("f2", "f3"):
        v = D[tag]["by_k"]["5"]
        d = v["paired_diff_trained_feat_vs_frozen"]
        rows.append(f"{tag.upper()}: 4-way 5-shot on held-out classes & {100 * v['frozen_imagenet']['acc']['mean']:.1f} & "
                    f"{100 * v['trained_feat']['acc']['mean']:.1f} ({100 * d['mean']:+.1f} $\\pm$ {100 * d['ci95']:.1f}) & "
                    f"{100 * v['trained_head']['acc']['mean']:.1f}")
    t, fz = D["f3_gfsl_trained"], D["f3_gfsl_frozen"]
    rows.append(f"F3: GFSL harmonic mean (10 base + 4 held-out) & {fz['feat']['gfsl']['gfsl_patch']['hm_acc']:.3f} & "
                f"{t['feat']['gfsl']['gfsl_patch']['hm_acc']:.3f} & {t['head']['gfsl']['gfsl_patch']['hm_acc']:.3f}")
    T["TabPilotRows"] = rows
    put("PilotHeadUnseen", f"{100 * t['head']['gfsl']['gfsl_patch']['novel_acc']:.1f}")

    # ------------------------------------------------------------------ write
    out = ["% AUTO-GENERATED by make_numbers_tex.py --mode " + a.mode + " -- do not edit by hand"]
    for k, v in M.items():
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    for k, rows in T.items():
        assert k.isalpha(), k
        body = "\n".join(r if r == r"\midrule" else r + r" \\" for r in rows)
        out.append(f"\\newcommand{{\\{k}}}{{%\n{body}\n}}")
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {len(M)} numbers, {len(T)} tables, mode={a.mode}")


if __name__ == "__main__":
    main()
