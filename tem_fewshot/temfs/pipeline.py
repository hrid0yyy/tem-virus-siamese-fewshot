"""High-level orchestration shared by the command-line scripts."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .constants import AUGMENTED_SPLIT, BASE_CLASSES, SPLITS, resolve_classes
from .data import PatchStore, build_index, find_patch_root, select_records, verify_counts
from .embed import extract
from .engine import fit_classifier, fit_metric
from .fewshot import (EmbeddingBank, closed_set_logits, closed_set_ncm, episodic_eval,
                      generalized_fewshot, kshot_curve, open_set)
from .models import EmbeddingNet, build_model_from_config
from .utils import Logger, environment_info, get_device, load_json, save_json, seed_everything

# --------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------
METRIC_DEFAULTS = dict(
    model_type="metric", backbone="densenet201", pretrained=True, memory_efficient=True,
    loss="supcon", temperature=0.1, margin=0.2, soft_margin=False, neg_per_pos=1,
    emb_dim=128, hidden_dim=512, aux_ce_weight=0.0, aux_scale=16.0,
    P=None, K=4, steps_per_epoch=100, image_balanced=True,
    n_way=8, k_shot=3, n_query=4,
    epochs=40, lr_backbone=5e-5, lr_head=5e-4, weight_decay=1e-4, optimizer="adamw",
    schedule="cosine", warmup_epochs=2, min_lr_ratio=0.01, grad_clip=5.0,
    augment=True, max_shift=8, amp=True,
    select_metric="val_head_macro_f1", patience=12, val_support_max=300, val_every=1,
    train_source="train",
)

# Replication of the authors' Keras DenseNet201 transfer model (train.py in Mendeley kxsvzhcfgs v2):
# Adam lr=1e-5, batch 12, 50 epochs, augmented_train, no online augmentation,
# checkpoint with the best validation accuracy.
CLASSIFIER_DEFAULTS = dict(
    model_type="classifier", backbone="densenet201", pretrained=True, memory_efficient=False,
    ce_head="paper", dropout=0.3, aux_scale=16.0,
    train_source=AUGMENTED_SPLIT, sampler="shuffle", class_weights="none", label_smoothing=0.0,
    batch_size=12, epochs=50, lr_backbone=1e-5, lr_head=1e-5, weight_decay=0.0, optimizer="adam",
    schedule="constant", warmup_epochs=0, min_lr_ratio=0.01, grad_clip=0.0,
    augment=False, max_shift=0, amp=True,
    select_metric="val_acc", patience=0, val_every=1,
)

# Keys that may differ when resuming (they do not change the experiment).
RUNTIME_KEYS = {"device", "num_workers", "preload", "skip_verify", "eval_batch_size", "resume",
                "debug_stop_after_epoch", "data_root", "run_dir", "environment"}

COMMON_DEFAULTS = dict(eval_batch_size=64, num_workers=0, preload=True, seed=0, device="auto",
                       deterministic=False, max_steps=0, skip_verify=False)


def make_config(model_type, overrides: dict):
    base = dict(METRIC_DEFAULTS if model_type == "metric" else CLASSIFIER_DEFAULTS)
    base.update(COMMON_DEFAULTS)
    for k, v in overrides.items():
        if v is not None:
            base[k] = v
    classes = resolve_classes(base.get("train_classes", "base14"))
    excl = base.get("exclude_classes")
    if excl:
        ex = resolve_classes(excl)
        classes = [c for c in classes if c not in ex]
        base["exclude_classes"] = ex
    else:
        base["exclude_classes"] = []
    if len(classes) < 2:
        raise ValueError("need at least 2 training classes")
    base["train_classes"] = classes
    if model_type == "metric":
        if base["P"] is None:
            base["P"] = min(len(classes), 14)
        base["P"] = min(base["P"], len(classes))
        if base["loss"] == "proto":
            base["n_way"] = min(base["n_way"], len(classes))
    if base["train_source"] == AUGMENTED_SPLIT and not set(classes) <= set(BASE_CLASSES):
        raise ValueError("augmented_train only exists for the 14 base classes; use --train-source train")
    if base["select_metric"].startswith("val_") is False:
        raise ValueError("select_metric must start with 'val_'")
    return base


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def prepare_data(data_root, log, include_augmented=False, preload=True, skip_verify=False):
    patch_root = find_patch_root(data_root)
    log(f"Patch root: {patch_root}")
    records, report = build_index(patch_root, SPLITS, include_augmented=include_augmented)
    if report["missing_splits"]:
        raise FileNotFoundError(f"missing split folders: {report['missing_splits']}")
    if report["unknown_classes"]:
        log(f"WARNING: unknown class folders ignored: {report['unknown_classes']}")
    if report["unexpected_files"]:
        log(f"WARNING: {len(report['unexpected_files'])} unexpected files ignored, e.g. "
            f"{report['unexpected_files'][:3]}")
    problems = verify_counts([r for r in records if r.split in SPLITS])
    if problems:
        msg = "Dataset does not match the official counts:\n  " + "\n  ".join(problems[:30])
        if skip_verify:
            log("WARNING: " + msg)
        else:
            raise RuntimeError(msg + "\nRun f0_verify_data.py for details, or pass --skip-verify.")
    else:
        log(f"Dataset verified: {sum(r.split in SPLITS for r in records)} patches match the official counts.")
    t0 = time.time()
    store = PatchStore(records, preload=preload, verbose=preload)
    if preload:
        log(f"Pre-loaded {len(records)} patches into RAM in {time.time() - t0:.0f}s "
            f"({store.array.nbytes / 1e9:.2f} GB)")
    return records, store


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def run_training(cfg, data_root, out_dir, run_name=None):
    seed_everything(cfg["seed"], cfg["deterministic"])
    device = get_device(cfg["device"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = cfg["loss"] if cfg["model_type"] == "metric" else f"ce-{cfg['ce_head']}"
    run_dir = Path(out_dir) / (run_name or f"{stamp}_{cfg['model_type']}_{tag}_{cfg['backbone']}")
    if run_dir.exists() and (run_dir / "config.json").exists():
        if not cfg.get("resume"):
            raise FileExistsError(f"{run_dir} exists. Use --resume to continue it, or choose another --run-name.")
        log = Logger(run_dir / "log.txt")
        if (run_dir / "train_summary.json").exists():
            log(f"Run already finished: {run_dir} - skipping training.")
            return run_dir, run_dir / "best.pt"
        saved = load_json(run_dir / "config.json")
        diffs = sorted(k for k in cfg if k not in RUNTIME_KEYS and k in saved and saved[k] != cfg[k])
        if diffs:
            log(f"WARNING: --resume ignores changed hyper-parameters {diffs}; using the saved config.")
        runtime = {k: cfg[k] for k in RUNTIME_KEYS if k in cfg}
        # runtime-only keys never carry over from the saved config (e.g. a debug interruption flag)
        saved = {k: v for k, v in saved.items() if k not in RUNTIME_KEYS or k in ("data_root", "run_dir")}
        cfg = {**saved, **runtime, "resume": True, "environment": environment_info()}
        log(f"===== resuming run {run_dir} =====")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        log = Logger(run_dir / "log.txt")
        cfg = dict(cfg, data_root=str(data_root), run_dir=str(run_dir), environment=environment_info())
        save_json(cfg, run_dir / "config.json")
    log(f"Run dir: {run_dir}")
    log(f"Environment: {cfg['environment']}")
    log(f"Training classes ({len(cfg['train_classes'])}): {cfg['train_classes']}")
    if cfg["exclude_classes"]:
        log(f"Held out from training: {cfg['exclude_classes']}")

    need_aug = cfg["train_source"] == AUGMENTED_SPLIT
    records, store = prepare_data(data_root, log, include_augmented=need_aug, preload=cfg["preload"],
                                  skip_verify=cfg["skip_verify"])
    train_recs = select_records(records, cfg["train_source"], cfg["train_classes"])
    val_recs = select_records(records, "validation", cfg["train_classes"])
    log(f"Train patches: {len(train_recs)} ({cfg['train_source']}), validation patches: {len(val_recs)}")
    train_groups = {r.group for r in train_recs}
    val_groups = {r.group for r in val_recs}
    assert not (train_groups & val_groups), "train/validation micrograph overlap"
    fit = fit_metric if cfg["model_type"] == "metric" else fit_classifier
    best = fit(cfg, train_recs, val_recs, store, device, run_dir, log)
    log(f"Finished. Best checkpoint: {best}")
    return run_dir, best


# --------------------------------------------------------------------------------------
# Models for evaluation
# --------------------------------------------------------------------------------------
def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model_from_config(cfg, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, cfg, ckpt


def frozen_imagenet_model(backbone, device):
    """Untrained ImageNet backbone; only its 'feat' embeddings are meaningful."""
    model = EmbeddingNet(backbone, pretrained=True, memory_efficient=False)
    return model.to(device).eval()


def build_bank(model, records, store, device, tta=True, batch_size=64, num_workers=0, keys=None):
    emb = extract(model, records, store, device, batch_size=batch_size, tta=tta, amp=True,
                  num_workers=num_workers)
    if keys is not None:
        emb = {k: v for k, v in emb.items() if k in keys}
    return EmbeddingBank(records, emb)


# --------------------------------------------------------------------------------------
# Evaluation orchestration
# --------------------------------------------------------------------------------------
def evaluate_bank(bank, base, novel, emb_keys, protocols, opts, log, model_type="metric"):
    """Run the requested protocols. base = classes the encoder was trained on (known classes)."""
    results = {"base_classes": base, "novel_classes": novel, "protocols": list(protocols)}
    novel = [c for c in novel if c not in base]
    for key in emb_keys:
        r = results.setdefault(key, {})
        if "closed" in protocols:
            log(f"[{key}] closed-set nearest-prototype on {len(base)} training classes (test split)")
            r["closed"] = closed_set_ncm(bank, key, base, n_boot=opts["n_boot"], seed=opts["seed"])
            log(f"    acc={r['closed']['accuracy']:.4f} macroF1={r['closed']['macro_f1']:.4f} "
                f"| image-level acc={r['closed']['image_level']['accuracy']:.4f}")
        if "gfsl" in protocols and novel:
            log(f"[{key}] generalized few-shot: {len(base)} base + {len(novel)} novel classes")
            g = generalized_fewshot(bank, key, base, novel, support_splits=tuple(opts["novel_support_splits"]),
                                    n_boot=opts["n_boot"], seed=opts["seed"])
            r["gfsl"] = g
            gp = g["gfsl_patch"]
            log(f"    all-class acc={g['accuracy']:.4f} macroF1={g['macro_f1']:.4f} | base acc={gp['base_acc']:.4f} "
                f"novel acc={gp['novel_acc']:.4f} HM={gp['hm_acc']:.4f} | novel-only acc={g['novel_only_acc']:.4f}")
        if "kshot" in protocols and novel:
            log(f"[{key}] K-shot curve on novel classes, K={opts['kshot_list']}, repeats={opts['kshot_repeats']}")
            kc = kshot_curve(bank, key, base, novel, ks=opts["kshot_list"], repeats=opts["kshot_repeats"],
                             seed=opts["seed"])
            r["kshot"] = kc
            for k, v in kc.items():
                log(f"    K={k}: base={v['base_acc']['mean']:.4f} novel={v['novel_acc']['mean']:.4f}"
                    f"+-{v['novel_acc']['ci95']:.4f} HM={v['hm_acc']['mean']:.4f} "
                    f"novel-only={v['novel_only_acc']['mean']:.4f}")
        if "episodic" in protocols:
            ep_classes = opts.get("episodic_classes") or novel
            leak = [c for c in ep_classes if c in base]
            if leak:
                raise ValueError(f"episodic protocol pools all splits; classes seen in training are not "
                                 f"allowed: {leak}")
            if len(ep_classes) >= 2:
                r["episodic"] = {}
                n_way = opts.get("episodic_way") or len(ep_classes)
                for k in opts["episodic_k"]:
                    e = episodic_eval(bank, key, ep_classes, n_way=n_way, k_shot=k,
                                      n_query=opts["n_query"], episodes=opts["episodes"], seed=opts["seed"])
                    r["episodic"][str(k)] = e
                    log(f"[{key}] episodic {n_way}-way {k}-shot on {ep_classes}: "
                        f"acc={e['acc']['mean']:.4f}+-{e['acc']['ci95']:.4f}")
        if "openset" in protocols and novel:
            o = open_set(bank, key, base, novel)
            r["openset"] = o
            log(f"[{key}] open-set (known={len(base)}, unknown={len(novel)}): patch AUROC={o['patch']['auroc']:.4f} "
                f"FPR@95TPR={o['patch']['fpr_at_95tpr']:.4f} | image AUROC={o['image']['auroc']:.4f}")
    if model_type == "classifier" and "logits" in bank.emb and "closed" in protocols:
        log("[logits] closed-set classifier evaluation (test split)")
        c = closed_set_logits(bank, base, n_boot=opts["n_boot"], seed=opts["seed"])
        results["logits"] = {"closed": c}
        msg = f"    acc={c['accuracy']:.4f} macroF1={c['macro_f1']:.4f}"
        if "gfsl_patch" in c:
            gp = c["gfsl_patch"]
            msg += f" | base acc={gp['base_acc']:.4f} novel acc={gp['novel_acc']:.4f} HM={gp['hm_acc']:.4f}"
        log(msg)
    return results
