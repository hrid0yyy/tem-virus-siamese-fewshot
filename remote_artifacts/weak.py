import json, os
W = ["Dengue", "Sapovirus", "Guanarito", "LCM", "Machupo", "Astrovirus", "Adenovirus", "Lassa"]
base = r"E:\tem_fewshot\runs"
rows = []
def add(label, names, f1, rec):
    rows.append((label, {c: (f1[names.index(c)], rec[names.index(c)]) for c in W}))
for run in ["E1_siamese_supcon_densenet201", "E1_siamese_supcon_densenet201_seed1", "E1_siamese_supcon_densenet201_seed2",
            "E6_supcon_auxce05_densenet201", "E6_triplet_densenet201"]:
    ev = os.path.join(base, run, "eval_full", "results.json")
    if os.path.exists(ev):
        g = json.load(open(ev))["feat"]["gfsl"]
        add(run.replace("E1_siamese_supcon_densenet201", "siamese").replace("_densenet201", "") + " feat", g["class_names"], g["per_class"]["f1"], g["per_class"]["recall"])
    an = os.path.join(base, run, "analysis", "analysis.json")
    if os.path.exists(an):
        a = json.load(open(an)); names = a["base"] + a["novel"]
        for var in ("feat_cl2n",):
            v = a["variants"][var]["gfsl_test_calibrated"]
            add(run.replace("E1_siamese_supcon_densenet201", "siamese").replace("_densenet201", "") + " cl2n+calib", names, v["per_class_f1"], v["per_class_recall"])
ce = json.load(open(os.path.join(base, "E2_ce22_densenet201", "eval_full", "results.json")))["logits"]["closed"]
add("CE trained on 22 (reference)", ce["class_names"], ce["per_class"]["f1"], ce["per_class"]["recall"])
print("F1 (recall)".ljust(34) + "".join(c[:10].rjust(12) for c in W))
for label, d in rows:
    print(label[:33].ljust(34) + "".join(f"{d[c][0]:.2f}({d[c][1]:.2f})".rjust(12) for c in W))
