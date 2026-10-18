"""Consistency checks for main.tex: every hand-written claim must agree with the generated numbers.

    python verify_paper.py      (run after make_numbers_tex.py --mode fp32 and a tectonic build)
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "remote_artifacts" / "runs"
tex = (HERE / "main.tex").read_text(encoding="utf-8")
num = (HERE / "numbers.tex").read_text(encoding="utf-8")
M = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", num))
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok &= bool(cond)


def ev(r):
    return json.load(open(RUNS / r / "eval_fp32" / "results.json"))


S = ["E1_siamese_supcon_densenet201", "E1_siamese_supcon_densenet201_seed1", "E1_siamese_supcon_densenet201_seed2"]
B = ["E1_paper_densenet201", "E1_paper_densenet201_seed1", "E1_paper_densenet201_seed2"]

# 1. every macro used in main.tex is defined; no PENDING / red warnings in numbers
used = set(re.findall(r"\\([A-Z][A-Za-z]+)", tex)) & set(M) | {m for m in re.findall(r"\\([A-Z][A-Za-z]+)\\?", tex) if m.startswith(("Siam", "BOne", "Pair", "Seed", "Cmp", "Kshot", "Weak", "Tab"))}
undefined = sorted(m for m in used if m not in M and not m.startswith("Tab"))
check(not undefined, f"all number macros used in main.tex are defined {undefined if undefined else ''}")
check("PENDING" not in num and "revise text" not in num, "numbers.tex has no PENDING / revise markers")
check("fp32" in M.get("InferenceMode", ""), "numbers.tex was generated from the final fp32 results")

# 2. hand-written claims
pub = 93.10
accs = {r: 100 * ev(r)["logits" if r in B else "feat"]["closed"]["accuracy"] for r in S + B}
check(all(a > pub for a in accs.values()), f"'every seed of both models exceeded 93.1%': min {min(accs.values()):.2f}")
check(M["SeedClosedNSig"] == "0", "abstract 'no seed pair differed significantly in accuracy' (14-class)")
check(M["PairLrNcmAllSig"] == M["PairLrNcmAllN"], "abstract 'LR significant for every checkpoint'")
check(M["PairKOneNSig"] == M["PairKOneN"], "abstract/discussion 'significant in all seed pairs' (1-shot)")
check(M["WinSiamLr"] == "3", "'LR selected for all three Siamese seeds'")
ks = {k: [ev(r)["feat"]["kshot"][k]["novel_mean_class_acc"]["mean"] for r in S] for k in ("1", "3", "5", "10")}
kb = {k: [ev(r)["feat"]["kshot"][k]["novel_mean_class_acc"]["mean"] for r in B] for k in ("1", "3", "5", "10")}
mean = lambda v: sum(v) / len(v)
check(mean(ks["1"]) > mean(kb["1"]), "'Siamese better at one shot' (K=1 mean)")
check(all(mean(kb[k]) >= mean(ks[k]) for k in ("3", "5", "10")), "'from three examples onwards B1 equal or better'")
ep = lambda r, k: ev(r)["feat"]["episodic"][k]["acc"]["mean"]
check(mean([ep(r, "5") for r in B]) >= mean([ep(r, "5") for r in S]) and mean([ep(r, "1") for r in B]) >= mean([ep(r, "1") for r in S]),
      "'advantage did not persist in rare-class-only episodes'")
f = lambda s: float(s.split()[0])
check(f(M["BOneDcKOne"]) >= f(M["SiamDcKOne"]) - 3, "'with DC, Siamese and B1 similar at one shot'")
check(abs(f(M["SiamGfslFOne"]) - f(M["BOneGfslFOne"])) < 0.01, "'same 22-class NCM macro-F1'")
check(f(M["BOneGfslRare"]) > f(M["SiamGfslRare"]), "'B1 recognised rare classes slightly better with NCM'")
check(f(M["SiamGfslAcc"]) > f(M["BOneGfslAcc"]) and f(M["SiamGfslBase"]) > f(M["BOneGfslBase"]), "'Siamese more accurate overall, via base classes'")
check(f(M["SiamLrFOne"]) < f(M["CeFOne"]), "'LR recovered much of the gap to B2' (still below B2)")
check(f(M["SiamHeadClosedAcc"]) > f(M["SiamClosedAcc"]), "'head scored slightly higher on base classes'")
check(int(M["HeadRareMax"]) < 50, "'head collapsed on rare classes'")
check(f(M["SiamMahaFOne"]) > f(M["SiamLrFOne"]) - 0.02, "'Mahalanobis reached a similar macro-F1'")

# 2b. figures agree with the numbers used in the text
FV = json.load(open(HERE / "figures" / "figure_values.json", encoding="utf-8"))
ks_fig = FV["kshot_mean_sd"]["Siamese (SupCon) features"][0]
kb_fig = FV["kshot_mean_sd"]["B1 (cross-entropy) features"][0]
check(f"{ks_fig[0]:.1f}" == M["KshotSiamOne"].split()[0] and f"{kb_fig[0]:.1f}" == M["KshotBOneOne"].split()[0],
      "Fig. kshot K=1 points equal the K=1 numbers in the text")
check(ks_fig[0] > kb_fig[0] and all(kb_fig[i] >= ks_fig[i] for i in (1, 2, 3)), "Fig. kshot shows the crossover described in the text")
cn = FV["confusion_rare_pct"]
check(f"{cn['Nearest class mean'][0][5]:.0f}" == M["DengueToSapPctNcm"] and f"{cn['Logistic regression'][0][5]:.0f}" == M["DengueToSapPctLr"],
      "Fig. confusion Dengue->Sapovirus cells equal the text")
check(f"{cn['Nearest class mean'][0][0]:.0f}" == M["DengueRecallPctNcm"] and f"{cn['Logistic regression'][0][0]:.0f}" == M["DengueRecallPctLr"],
      "Fig. confusion Dengue recall cells equal the text")
sup = {"Dengue": 44, "Guanarito": 7, "LCM": 14, "Machupo": 23, "Pseudocowpox": 16, "Sapovirus": 8, "TBE": 11, "WestNile": 34}
check(FV["confusion_rare_row_counts"]["Nearest class mean"] == [3 * v for v in sup.values()], "Fig. confusion rows = test patches x 3 seeds")
check(f(M["RarePrecLr"]) > f(M["RarePrecNcm"]) and (f(M["RarePrecLr"]) - f(M["RarePrecNcm"])) > (f(M["RareRecLr"]) - f(M["RareRecNcm"])),
      "'gain came mainly from precision'")
tr = FV["training"]
check(all(v["best_epoch"] == v["argmax_epoch"] for v in tr.values()), "Fig. training: starred epoch = best validation epoch")
check(all(tr[r]["epochs_run"] == tr[r]["best_epoch"] + 12 for r in S), "'Siamese stopped after 12 epochs without improvement'")
check(all(tr[r]["epochs_run"] == 50 for r in B), "'B1 trained for the fixed 50 epochs'")

# 3. CMPB format
abstract = tex[tex.index(r"\begin{abstract}"):tex.index(r"\end{abstract}")]
for h in ("Background and Objective:", "Methods:", "Results:", "Conclusions:"):
    check(h in abstract, f"structured abstract has '{h}'")
check(r"\cite" not in abstract, "no citations in abstract")
kw = tex[tex.index(r"\begin{keyword}"):tex.index(r"\end{keyword}")]
check(3 <= kw.count(r"\sep") + 1 <= 6, f"keywords 3-6 ({kw.count(chr(92) + 'sep') + 1})")
hl = [l[2:] for l in (HERE / "highlights.txt").read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
check(3 <= len(hl) <= 5 and all(len(h) <= 85 for h in hl), f"highlights: {len(hl)} bullets, max {max(map(len, hl))} chars")
for sec in ("Introduction", "Methods", "Results", "Discussion", "Conclusions"):
    check(re.search(r"\\section\{" + sec, tex), f"section '{sec}' present")
check("Declaration of generative AI" in tex, "generative-AI declaration present")
check("Declaration of competing interests" in tex, "competing-interests declaration present")
bib = (HERE / "references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{(\w+),", bib))
cited = set(k.strip() for c in re.findall(r"\\cite\{([^}]*)\}", tex + num) for k in c.split(","))
check(cited <= keys, f"all {len(cited)} cited keys exist in references.bib {sorted(cited - keys) if cited - keys else ''}")
check(len(cited) <= 50, f"references <= 50 ({len(cited)})")
check((HERE / "figures" / "architecture.pdf").exists(), "Fig. 1 PDF exists")
ds = FV.get("dataset_train_patches_micrographs", {})
EC = json.load(open(HERE.parent / "tem_fewshot" / "temfs" / "expected_counts.json"))["counts"]
check(len(ds) == 22 and all(tuple(v) == (EC[c]["train"]["patches"], EC[c]["train"]["images"]) for c, v in ds.items()),
      "dataset figure: 22 classes, train counts equal the official counts (Table 1)")
src = json.load(open(HERE / "figures" / "dataset_patches" / "SOURCES.json"))
check(len(src) == 22 and all(v["split"] == "train" for v in src.values()), "dataset figure: every example comes from the train split, provenance recorded")
print("\nOVERALL:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
