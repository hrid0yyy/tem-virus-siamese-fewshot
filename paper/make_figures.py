"""Generate the data figures of main.tex from the raw result files (no hand-typed numbers).

    python make_figures.py            -> figures/kshot.pdf, figures/confusion_rare.pdf, figures/training.pdf

Sources (final round-2 outputs, fp32 deterministic inference):
  runs/<run>/eval_fp32/results.json            K-shot curve (100 support draws per K)
  runs/gfsl_fp32/<run>/predictions_<run>.npz    per-patch test predictions (NCM and LR)
  runs/<run>/history.csv, train_summary.json    training curves and selected epoch
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "remote_artifacts" / "runs"
OUT = HERE / "figures"
S = ["E1_siamese_supcon_densenet201", "E1_siamese_supcon_densenet201_seed1", "E1_siamese_supcon_densenet201_seed2"]
B = ["E1_paper_densenet201", "E1_paper_densenet201_seed1", "E1_paper_densenet201_seed2"]
RARE = ["Dengue", "Guanarito", "LCM", "Machupo", "Pseudocowpox", "Sapovirus", "TBE", "WestNile"]
C_SIAM, C_B1 = "#0072B2", "#D55E00"  # Okabe-Ito blue / vermillion
# Full-width figures are drawn at the printed size: \textwidth of the manuscript (elsarticle preprint, 12pt) = 390 pt,
# so they are included at about 100 % and no text is smaller than 7 pt in print (checked by verify_paper.py).
TEXTWIDTH_IN = 390 / 72

# Latin Modern Roman (the font of the manuscript text) from the TeX distribution, if available
_LM = sorted(Path.home().glob("AppData/Local/TectonicProject/Tectonic/cache/bundles/data/*/lmroman10-*.otf"))
if _LM:
    from matplotlib import font_manager
    for _f in _LM:
        font_manager.fontManager.addfont(str(_f))
SERIF = ["Latin Modern Roman"] if _LM else ["DejaVu Serif"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "cm", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.unicode_minus": False,
})


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fig_kshot():
    ks = ["1", "3", "5", "10"]
    data = {}
    for name, runs in (("Siamese (SupCon) features", S), ("Baseline (cross-entropy) features", B)):
        vals = np.array([[load(RUNS / r / "eval_fp32" / "results.json")["feat"]["kshot"][k]["novel_mean_class_acc"]["mean"]
                          for k in ks] for r in runs]) * 100
        data[name] = (vals.mean(0), vals.std(0, ddof=1))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.array([int(k) for k in ks])
    for (name, (m, s)), col, mk in zip(data.items(), (C_SIAM, C_B1), ("o", "s")):
        ax.errorbar(x, m, yerr=s, color=col, marker=mk, ms=4, lw=1.4, capsize=2.5, label=name)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.minorticks_off()
    ax.set_xlabel("$K$ (training patches per rare virus)")
    ax.set_ylabel("Rare-class mean accuracy (%)")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "kshot.pdf")
    plt.close(fig)
    return {n: (np.round(m, 1).tolist(), np.round(s, 1).tolist()) for n, (m, s) in data.items()}


def fig_confusion():
    """Rows: true rare class. Columns: predicted rare class, or any of the 14 base classes (pooled).
    Counts summed over the three Siamese seeds, then row-normalised to %."""
    panels = {}
    for key, title in (("ncm_pred", "Nearest class mean"), ("logreg_pred", "Logistic regression")):
        mat = np.zeros((len(RARE), len(RARE) + 1))
        for r in S:
            z = np.load(RUNS / "gfsl_fp32" / r / f"predictions_{r}.npz", allow_pickle=False)
            cls = [str(c) for c in z["classes"]]
            rare_idx = [cls.index(c) for c in RARE]
            y, p = z["y"], z[key]
            for i, ti in enumerate(rare_idx):
                rows = p[y == ti]
                for j, pj in enumerate(rare_idx):
                    mat[i, j] += int((rows == pj).sum())
                mat[i, -1] += int((~np.isin(rows, rare_idx)).sum())
        panels[title] = mat
    # standard virus abbreviations on both axes; the key inside the figure gives the full names
    abbr = ["DENV", "GTOV", "LCMV", "MACV", "PCPV", "SaV", "TBEV", "WNV"]
    key = [("DENV", "Dengue virus"), ("GTOV", "Guanarito virus"), ("LCMV", "Lymphocytic\nchoriomeningitis virus"),
           ("MACV", "Machupo virus"), ("PCPV", "Pseudocowpox virus"), ("SaV", "Sapovirus"),
           ("TBEV", "Tick-borne\nencephalitis virus"), ("WNV", "West Nile virus"), ("Base", "any of the 14\nbase classes")]
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.45), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.07, 0.95])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for k, (ax, (title, mat)) in enumerate(zip(axes, panels.items())):
        pct = 100 * mat / mat.sum(1, keepdims=True)
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(pct.shape[0]):
            for j in range(pct.shape[1]):
                if pct[i, j] >= 5:
                    ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if pct[i, j] > 55 else "black")
        ax.set_xticks(range(len(abbr) + 1))
        ax.set_xticklabels(abbr + ["Base"], rotation=90, fontsize=7)
        ax.set_yticks(range(len(abbr)))
        ax.set_yticklabels(abbr if k == 0 else [], fontsize=7)  # rows are the same in both panels
        ax.tick_params(length=0, pad=2)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Predicted class", fontsize=8)
        if k == 0:
            ax.set_ylabel("True class", fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label("% of test patches", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    kx = fig.add_subplot(gs[0, 3])
    kx.axis("off")
    y = 1.0
    for a, full in key:
        kx.annotate(a, (0, y), xycoords="axes fraction", fontsize=7, fontweight="bold", va="top")
        kx.annotate(full, (0, y), xycoords="axes fraction", xytext=(29, 0), textcoords="offset points",
                    fontsize=7, va="top", linespacing=1.1)
        y -= 0.083 * (1 + full.count("\n") * 0.85) + 0.012
    fig.savefig(OUT / "confusion_rare.pdf")
    plt.close(fig)
    # values used for verification
    return {t: (100 * m / m.sum(1, keepdims=True)).round(3).tolist() for t, m in panels.items()}, \
        {t: m.sum(1).astype(int).tolist() for t, m in panels.items()}


def read_hist(run):
    with open(RUNS / run / "history.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_training():
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.4), sharex="col", constrained_layout=True)
    info = {}
    for col, (runs, metric, mlabel, colr, name) in enumerate((
            (S, "val_feat_macro_f1", "Validation macro-F1\n(nearest class mean)", C_SIAM, "Siamese (SupCon)"),
            (B, "val_acc", "Validation accuracy", C_B1, "Baseline (benchmark classifier)"))):
        for i, r in enumerate(runs):
            h = read_hist(r)
            ep = np.array([int(e["epoch"]) for e in h])
            loss = np.array([float(e["train_loss"]) for e in h])
            val = np.array([float(e[metric]) for e in h])
            best = load(RUNS / r / "train_summary.json")["best_epoch"]
            ls = ("-", "--", ":")[i]
            axes[0, col].plot(ep, loss, color=colr, ls=ls, lw=1.1, label=f"seed {i}")
            axes[1, col].plot(ep, val, color=colr, ls=ls, lw=1.1, label=f"seed {i}")
            bi = int(np.where(ep == best)[0][0])
            axes[1, col].plot(best, val[bi], marker="*", color="black", ms=7, zorder=5)
            info[r] = {"epochs_run": int(ep.max()), "best_epoch": best, "argmax_epoch": int(ep[np.argmax(val)]),
                       "best_val": round(float(val[bi]), 4)}
        axes[0, col].set_title(name)
        axes[0, col].set_ylabel("Training loss")
        axes[1, col].set_ylabel(mlabel)
        axes[1, col].set_xlabel("Epoch")
        axes[0, col].legend(frameon=False)
        axes[1, col].grid(axis="y", lw=0.4, alpha=0.5)
    star = axes[1, 0].plot([], [], marker="*", color="black", ls="none", ms=7, label="selected checkpoint")[0]
    axes[1, 0].legend(handles=[star], frameon=False, loc="lower right")
    fig.savefig(OUT / "training.pdf")
    plt.close(fig)
    return info


def fig_dataset():
    """One official-train-split example patch per class; base classes on two rows, rare classes on one."""
    from PIL import Image
    counts = load(HERE.parent / "tem_fewshot" / "temfs" / "expected_counts.json")["counts"]
    base = ["Adenovirus", "Astrovirus", "CCHF", "Cowpox", "Ebola", "Influenza", "Lassa", "Marburg", "Nipah virus",
            "Norovirus", "Orf", "Papilloma", "Rift Valley", "Rotavirus"]
    # explicit layout in inches, so that the spacing between all rows is identical
    W, P = TEXTWIDTH_IN - 0.06, 0.56           # figure width (minus the saved padding), patch size
    GAP_X = (W - 8 * P) / 7                    # horizontal gap between patches
    HEAD, NAME, CNT, GAP_Y = 0.20, 0.15, 0.15, 0.10
    tops = [HEAD + NAME]                       # top of the patches of each row, measured from the top
    tops.append(tops[0] + P + CNT + GAP_Y + NAME)
    head2 = tops[1] + P + CNT + GAP_Y          # top of the rare-class heading
    tops.append(head2 + HEAD + NAME)
    H = tops[2] + P + CNT
    fig = plt.figure(figsize=(W, H))
    slots = [(0, j) for j in range(7)] + [(1, j) for j in range(7)] + [(2, j) for j in range(8)]
    used = {}
    for (r, cidx), cls in zip(slots, base + RARE):
        ax = fig.add_axes([cidx * (P + GAP_X) / W, (H - tops[r] - P) / H, P / W, P / H])
        im = np.asarray(Image.open(OUT / "dataset_patches" / f"{cls.replace(' ', '')}.png").convert("L"))
        assert im.shape == (256, 256)
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        rare = cls in RARE
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(C_B1 if rare else "0.55")
            sp.set_linewidth(1.6 if rare else 0.6)
        tr = counts[cls]["train"]
        name = cls.replace("WestNile", "West Nile")
        ax.set_title(name, fontsize=7.5, pad=2, color=C_B1 if rare else "black")
        ax.set_xlabel(f"{tr['patches']:,} / {tr['images']}", fontsize=7, labelpad=1.5)
        used[cls] = (tr["patches"], tr["images"])
        if (r, cidx) == (0, 0):  # 100 nm scale bar (patches are 256 x 256 px at 1 nm/px)
            ax.plot([146, 246], [238, 238], color="white", lw=2.2, solid_capstyle="butt")
            ax.text(248, 224, "100 nm", color="white", fontsize=7, ha="right", va="bottom")
    fig.text(0, 1 - HEAD / 2 / H, "Base classes (training patches / micrographs)", fontsize=8, weight="bold",
             ha="left", va="center")
    fig.text(0, 1 - (head2 + HEAD / 2) / H, "Rare classes, never used to train the encoder (training patches / micrographs)",
             fontsize=8, weight="bold", ha="left", va="center", color=C_B1)
    fig.savefig(OUT / "dataset.pdf")
    plt.close(fig)
    return used


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ds = fig_dataset()
    k = fig_kshot()
    c, n = fig_confusion()
    t = fig_training()
    json.dump({"kshot_mean_sd": k, "confusion_rare_pct": c, "confusion_rare_row_counts": n, "training": t,
               "dataset_train_patches_micrographs": ds},
              open(OUT / "figure_values.json", "w", encoding="utf-8"), indent=1)
    print("wrote figures/kshot.pdf, figures/confusion_rare.pdf, figures/training.pdf, figures/figure_values.json")
