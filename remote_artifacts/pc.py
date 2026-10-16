import json
r = json.load(open(r"E:\tem_fewshot\runs\E1_siamese_supcon_densenet201\eval_full\results.json"))
g = r["feat"]["gfsl"]; n = g["class_names"]; pc = g["per_class"]; cm = g["confusion"]
rows = sorted(range(len(n)), key=lambda i: pc["f1"][i])
print(f"{'class':13s} {'n':>4s} {'prec':>5s} {'rec':>5s} {'F1':>5s}  main confusions (true -> predicted)")
for i in rows:
    conf = sorted(((cm[i][j], n[j]) for j in range(len(n)) if j != i and cm[i][j]), reverse=True)[:3]
    print(f"{n[i]:13s} {pc['support'][i]:4d} {pc['precision'][i]:5.2f} {pc['recall'][i]:5.2f} {pc['f1'][i]:5.2f}  " + ", ".join(f"{c}:{k}" for k, c in conf))
