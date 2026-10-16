import json
r = json.load(open(r"E:\tem_fewshot\runs\E2_ce22_densenet201\eval_full\results.json"))
c = r["logits"]["closed"]; n = c["class_names"]; pc = c["per_class"]
print("ci acc", c["bootstrap_ci95"]["accuracy"], "ci f1", c["bootstrap_ci95"]["macro_f1"])
print("gfsl_patch", {k: round(v, 4) for k, v in c["gfsl_patch"].items()})
print("image", c["image_level"]["accuracy"], c["image_level"]["macro_f1"])
for i in sorted(range(len(n)), key=lambda i: pc["f1"][i])[:10]:
    print(f"{n[i]:13s} P={pc['precision'][i]:.2f} R={pc['recall'][i]:.2f} F1={pc['f1'][i]:.2f} n={pc['support'][i]}")
