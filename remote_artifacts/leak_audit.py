"""Independent pre-run leakage audit on the real dataset (pixel-level)."""
import hashlib, sys
from collections import Counter, defaultdict
sys.path.insert(0, r"E:\tem_fewshot")
import numpy as np
from temfs.constants import BASE_CLASSES, NOVEL_CLASSES, SPLITS, AUGMENTED_SPLIT
from temfs.data import build_index, find_patch_root, load_patch, d4_transform, select_records
from temfs.pipeline import make_config

root = find_patch_root(r"E:\tem_fewshot\data")
recs, rep = build_index(root, SPLITS, include_augmented=True)
h = lambda a: hashlib.sha1(a.tobytes()).hexdigest()
print("indexed:", Counter(r.split for r in recs))

# 1) every D4 variant of every validation/test patch
evalset = {}
for r in recs:
    if r.split in ("validation", "test"):
        a = load_patch(r.path)
        for k in range(8):
            evalset.setdefault(h(d4_transform(a, k)), set()).add(f"{r.split}/{r.cls}")
# 2) all training-side patches: train + augmented_train
hits = Counter()
for r in recs:
    if r.split in ("train", AUGMENTED_SPLIT):
        key = h(load_patch(r.path))
        if key in evalset:
            hits[(r.split, r.cls, tuple(sorted(evalset[key])))] += 1
print("train/augmented patches identical to ANY rotation/flip of a val/test patch:", sum(hits.values()))
for k, v in list(hits.items())[:10]:
    print("   ", k, v)

# 3) validation vs test (pixel-level incl. D4)
vt = Counter()
valhash = {}
for r in recs:
    if r.split == "validation":
        a = load_patch(r.path)
        for k in range(8):
            valhash[h(d4_transform(a, k))] = r.cls
for r in recs:
    if r.split == "test" and h(load_patch(r.path)) in valhash:
        vt[r.cls] += 1
print("test patches identical to a rotation/flip of a validation patch:", sum(vt.values()), dict(vt))

# 4) augmented_train must be derived only from train micrographs (name -> train image id exists)
train_ids = {(r.cls, r.image_id) for r in recs if r.split == "train"}
bad = [r.path for r in recs if r.split == AUGMENTED_SPLIT and (r.cls, r.image_id) not in train_ids]
print("augmented patches whose micrograph id does not exist in train:", len(bad))

# 5) configs that will be launched
for mt, ov in (("metric", {"train_classes": "base14", "select_metric": "val_feat_macro_f1"}),
               ("classifier", {"train_classes": "base14"})):
    cfg = make_config(mt, ov)
    tr = select_records(recs, cfg["train_source"], cfg["train_classes"])
    va = select_records(recs, "validation", cfg["train_classes"])
    print(f"[{mt}] train_source={cfg['train_source']} n_train={len(tr)} n_val={len(va)} "
          f"classes={len(cfg['train_classes'])} novel_in_train={sorted(set(r.cls for r in tr) & set(NOVEL_CLASSES))} "
          f"splits_in_train={sorted(set(r.split for r in tr))} test_in_train={any(r.split=='test' for r in tr)}")
    print(f"     loss/head={cfg.get('loss', cfg.get('ce_head'))} epochs={cfg['epochs']} lr_bb={cfg['lr_backbone']} "
          f"lr_head={cfg['lr_head']} opt={cfg['optimizer']} batch={cfg.get('batch_size', (cfg.get('P'), cfg.get('K')))} "
          f"aug={cfg['augment']} select={cfg['select_metric']} patience={cfg['patience']}")
