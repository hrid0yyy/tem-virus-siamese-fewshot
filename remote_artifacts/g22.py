import json
r = json.load(open(r"E:\tem_fewshot\runs\E1_siamese_supcon_densenet201\eval_full\results.json"))
for key in ("feat", "head"):
    g = r[key]["gfsl"]; ci = g["bootstrap_ci95"]; il = g["image_level"]
    print(f"[{key}] patches={g['n']} acc={g['accuracy']:.4f} CI=[{ci['accuracy']['low']:.4f},{ci['accuracy']['high']:.4f}] "
          f"macroF1={g['macro_f1']:.4f} CI=[{ci['macro_f1']['low']:.4f},{ci['macro_f1']['high']:.4f}] "
          f"macroP={g['macro_precision']:.4f} macroR={g['macro_recall']:.4f} balanced_acc={g['balanced_accuracy']:.4f}")
    print(f"      image-level: n={il['n']} acc={il['accuracy']:.4f} macroF1={il['macro_f1']:.4f}")
    gp, gi = g["gfsl_patch"], g["gfsl_image"]
    print(f"      mean-class acc base={gp['base_mean_class_acc']:.4f} novel={gp['novel_mean_class_acc']:.4f} HM={gp['hm_mean_class_acc']:.4f} | image-level base={gi['base_acc']:.4f} novel={gi['novel_acc']:.4f}")
names = r["feat"]["gfsl"]["class_names"]; cm = r["feat"]["gfsl"]["confusion"]
i = names.index("Dengue"); row = cm[i]
print("Dengue (feat) predicted as:", {names[j]: v for j, v in enumerate(row) if v})
