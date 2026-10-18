"""Training loops for the Siamese/metric encoder and for cross-entropy classifier baselines."""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import PatchDataset
from .embed import extract
from .fewshot import EmbeddingBank, closed_set_ncm
from .losses import (batch_hard_triplet_loss, contrastive_pair_loss, pair_bce_loss,
                     prototypical_loss, supcon_loss)
from .metrics import classification_scores
from .models import build_model_from_config, count_parameters
from .samplers import EpisodeBatchSampler, PKBatchSampler, draw_image_balanced
from .utils import append_csv, save_json


# --------------------------------------------------------------------------------------
# Optimiser / schedule
# --------------------------------------------------------------------------------------
def build_optimizer(model, cfg):
    head_ids = {id(p) for p in model.head_parameters()}
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "backbone"
        decay = "no_decay" if (p.ndim <= 1 or name.endswith(".bias")) else "decay"
        groups.setdefault((part, decay), []).append(p)
    param_groups = []
    for (part, decay), params in groups.items():
        param_groups.append({
            "params": params,
            "lr": cfg["lr_head"] if part == "head" else cfg["lr_backbone"],
            "weight_decay": cfg["weight_decay"] if decay == "decay" else 0.0,
            "name": f"{part}/{decay}",
        })
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(param_groups)
    if cfg["optimizer"] == "adam":
        # Keras Adam (paper) has no weight decay; weight_decay here is plain L2 if set.
        return torch.optim.Adam(param_groups)
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown optimizer {cfg['optimizer']}")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = max(1, cfg["epochs"] * steps_per_epoch)
    warm = int(cfg.get("warmup_epochs", 0) * steps_per_epoch)
    min_ratio = cfg.get("min_lr_ratio", 0.01)

    def factor(step):
        if warm > 0 and step < warm:
            return (step + 1) / warm
        if cfg["schedule"] == "constant":
            return 1.0
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def format_row(row):
    parts = []
    for k, v in row.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.2e}" if (k.startswith("lr") or (v != 0 and abs(v) < 1e-3)) else f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def _amp(device, cfg):
    return bool(cfg.get("amp", True)) and device.type == "cuda"


def current_lrs(opt):
    lrs = {}
    for g in opt.param_groups:
        part = g.get("name", "backbone/").split("/")[0]
        lrs.setdefault(f"lr_{part}", g["lr"])
    return lrs


def _finish(run_dir, best, best_epoch, sel_key, history, log):
    best_path = run_dir / "best.pt"
    if not best_path.exists():
        log(f"WARNING: {sel_key} never improved (or was never computed); using last.pt as best.pt")
        shutil.copy(run_dir / "last.pt", best_path)
    save_json({"best_metric": best, "best_epoch": best_epoch, "select_metric": sel_key,
               "history": history}, run_dir / "train_summary.json")
    return best_path


def _save_checkpoint(path, model, cfg, epoch, best, history, train_state=None):
    """best.pt: model only. last.pt: + optimizer/scheduler/scaler/bookkeeping for exact resumption."""
    ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "best_metric": best, "history": history}
    if train_state is not None:
        ckpt["train_state"] = train_state
    tmp = Path(str(path) + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)  # atomic: a power cut never leaves a half-written checkpoint


def _try_resume(cfg, run_dir, model, opt, sched, scaler, log):
    """Restore the full training state from last.pt if resuming. Returns the bookkeeping tuple."""
    fresh = (0, -float("inf"), -1, [], 0)
    last = Path(run_dir) / "last.pt"
    if not cfg.get("resume") or not last.exists():
        return fresh
    ck = torch.load(last, map_location="cpu", weights_only=False)
    ts = ck.get("train_state")
    if ts is None:
        log("WARNING: last.pt has no training state (old format); starting from epoch 0.")
        return fresh
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ts["optimizer"])
    sched.load_state_dict(ts["scheduler"])
    scaler.load_state_dict(ts["scaler"])
    log(f"RESUMED from {last}: {ck['epoch']} epochs done, best {cfg['select_metric']}={ts['best']:.4f} "
        f"(epoch {ts['best_epoch']}), bad_epochs={ts['bad_epochs']}")
    start = cfg["epochs"] if ts.get("stopped_early") else ck["epoch"]  # early-stopped runs stay stopped
    return start, ts["best"], ts["best_epoch"], list(ck["history"]), ts["bad_epochs"]


def _end_of_epoch(row, epoch, model, cfg, opt, sched, scaler, state, run_dir, log):
    """Update best/early-stopping bookkeeping, then checkpoint. Returns (state, stop)."""
    best, best_epoch, bad_epochs, history = state
    sel_key = cfg["select_metric"]
    history.append(row)
    append_csv(run_dir / "history.csv", row)
    log(format_row(row))
    stop = False
    if sel_key in row:
        if row[sel_key] > best:
            best, best_epoch, bad_epochs = row[sel_key], epoch + 1, 0
            _save_checkpoint(run_dir / "best.pt", model, cfg, epoch + 1, best, history)
            log(f"  new best {sel_key}={best:.4f} (epoch {epoch + 1}) -> best.pt")
        else:
            bad_epochs += cfg["val_every"]
        if cfg.get("patience", 0) > 0 and bad_epochs >= cfg["patience"]:
            log(f"Early stopping: no improvement of {sel_key} for {bad_epochs} epochs.")
            stop = True
    ts = {"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
          "best": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "stopped_early": stop}
    _save_checkpoint(run_dir / "last.pt", model, cfg, epoch + 1, best, history, ts)
    return (best, best_epoch, bad_epochs, history), stop


def cap_records_per_class(records, max_per_class, seed=0):
    """Image-balanced, deterministic subset with at most max_per_class patches per class."""
    if not max_per_class or max_per_class <= 0:
        return list(records)
    rng = np.random.default_rng(seed)
    by_cls = {}
    for i, r in enumerate(records):
        by_cls.setdefault(r.cls, {}).setdefault(r.group, []).append(i)
    keep = []
    for cls in sorted(by_cls):
        g2i = by_cls[cls]
        order = sorted(g2i)
        order = [order[i] for i in rng.permutation(len(order))]
        keep += draw_image_balanced(rng, g2i, order, max_per_class, allow_repeat=False)
    return [records[i] for i in sorted(keep)]


# --------------------------------------------------------------------------------------
# Metric (Siamese) training
# --------------------------------------------------------------------------------------
def compute_metric_loss(model, z, y, cfg):
    """Returns (loss, stats_dict). z: embeddings (B, d), y: labels in training label space."""
    loss_name = cfg["loss"]
    stats = {}
    if loss_name == "supcon":
        loss = supcon_loss(z, y, cfg["temperature"])
    elif loss_name == "triplet":
        loss = batch_hard_triplet_loss(z, y, cfg["margin"], soft=cfg.get("soft_margin", False))
    elif loss_name == "proto":
        ns = cfg["n_way"] * cfg["k_shot"]
        loss, acc = prototypical_loss(z[:ns], y[:ns], z[ns:], y[ns:], cfg["temperature"])
        stats["episode_acc"] = float(acc.detach())
    elif loss_name == "pair":
        loss, acc = pair_bce_loss(model.pair_head, z, y, cfg.get("neg_per_pos", 1))
        stats["pair_acc"] = float(acc.detach())
    elif loss_name == "contrastive":
        loss = contrastive_pair_loss(z, y, cfg["margin"], cfg.get("neg_per_pos", 1))
    else:
        raise ValueError(f"unknown loss {loss_name}")
    if model.aux is not None and cfg.get("aux_ce_weight", 0) > 0:
        aux = F.cross_entropy(model.aux(z.float()), y)
        stats["aux_ce"] = float(aux.detach())
        loss = loss + cfg["aux_ce_weight"] * aux
    return loss, stats


def validate_metric(model, cfg, support_records, val_records, store, device, log):
    """Nearest-prototype accuracy / macro-F1 on the validation split (prototypes from train split)."""
    classes = cfg["train_classes"]
    recs = list(support_records) + list(val_records)
    emb = extract(model, recs, store, device, batch_size=cfg["eval_batch_size"], tta=False,
                  amp=_amp(device, cfg), num_workers=cfg["num_workers"])
    bank = EmbeddingBank(recs, emb)
    out = {}
    for key in ("head", "feat"):
        r = closed_set_ncm(bank, key, classes, support_splits=("train",), query_split="validation",
                           n_boot=0)
        out[f"val_{key}_acc"] = r["accuracy"]
        out[f"val_{key}_macro_f1"] = r["macro_f1"]
    return out


def fit_metric(cfg, train_records, val_records, store, device, run_dir, log):
    run_dir = Path(run_dir)
    classes = cfg["train_classes"]
    label_map = {c: i for i, c in enumerate(classes)}
    ds = PatchDataset(train_records, label_map, store, augment=cfg["augment"],
                      max_shift=cfg["max_shift"], seed=cfg["seed"])
    if cfg["loss"] == "proto":
        sampler = EpisodeBatchSampler(ds.labels, ds.groups, cfg["n_way"], cfg["k_shot"],
                                      cfg["n_query"], cfg["steps_per_epoch"], seed=cfg["seed"])
        batch_desc = f"{cfg['n_way']}-way {cfg['k_shot']}-shot {cfg['n_query']}-query episodes"
    else:
        sampler = PKBatchSampler(ds.labels, ds.groups, cfg["P"], cfg["K"], cfg["steps_per_epoch"],
                                 seed=cfg["seed"], image_balanced=cfg["image_balanced"])
        batch_desc = f"P={cfg['P']} classes x K={cfg['K']} samples"
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=cfg["num_workers"],
                    pin_memory=(device.type == "cuda"), persistent_workers=cfg["num_workers"] > 0)

    model = build_model_from_config(cfg).to(device)
    log(f"Model: EmbeddingNet({cfg['backbone']}) params={count_parameters(model):,} | loss={cfg['loss']} "
        f"| batches: {batch_desc} | steps/epoch={len(sampler)}")
    opt = build_optimizer(model, cfg)
    steps_per_epoch = min(len(sampler), cfg["max_steps"]) if cfg.get("max_steps") else len(sampler)
    sched = build_scheduler(opt, cfg, steps_per_epoch)
    use_amp = _amp(device, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    support_records = cap_records_per_class(train_records, cfg["val_support_max"], seed=cfg["seed"])
    sel_key = cfg["select_metric"]
    start_epoch, best, best_epoch, history, bad_epochs = _try_resume(cfg, run_dir, model, opt, sched, scaler, log)
    for epoch in range(start_epoch, cfg["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        t0, losses, extra = time.time(), [], {}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step, (x, y, _) in enumerate(dl):
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                z, _ = model(x)
            loss, stats = compute_metric_loss(model, z, y, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.get("grad_clip", 0) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
            for k, v in stats.items():
                extra.setdefault(k, []).append(v)
        train_time = time.time() - t0
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else float("nan"),
               **current_lrs(opt), "train_time_s": round(train_time, 1)}
        row.update({f"train_{k}": float(np.mean(v)) for k, v in extra.items()})
        if device.type == "cuda":
            row["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        if (epoch + 1) % cfg["val_every"] == 0 or epoch + 1 == cfg["epochs"]:
            row.update(validate_metric(model, cfg, support_records, val_records, store, device, log))
        state, stop = _end_of_epoch(row, epoch, model, cfg, opt, sched, scaler,
                                    (best, best_epoch, bad_epochs, history), run_dir, log)
        best, best_epoch, bad_epochs, history = state
        if stop:
            break
        if cfg.get("debug_stop_after_epoch") and epoch + 1 >= cfg["debug_stop_after_epoch"]:
            log(f"DEBUG: simulated interruption after epoch {epoch + 1}")
            return None
    return _finish(run_dir, best, best_epoch, sel_key, history, log)


# --------------------------------------------------------------------------------------
# Classifier (cross-entropy) training - paper replication and 22-class baselines
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_logits(model, records, label_map, store, device, cfg):
    emb = extract(model, records, store, device, batch_size=cfg["eval_batch_size"], tta=False,
                  amp=_amp(device, cfg), num_workers=cfg["num_workers"])
    y = np.array([label_map[r.cls] for r in records])
    res = classification_scores(y, emb["logits"].argmax(1), len(label_map))
    return res


def fit_classifier(cfg, train_records, val_records, store, device, run_dir, log):
    run_dir = Path(run_dir)
    classes = cfg["train_classes"]
    label_map = {c: i for i, c in enumerate(classes)}
    ds = PatchDataset(train_records, label_map, store, augment=cfg["augment"],
                      max_shift=cfg["max_shift"], seed=cfg["seed"])
    counts = np.bincount(ds.labels, minlength=len(classes)).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"classes without training samples: {[c for c, n in zip(classes, counts) if n == 0]}")
    if cfg["sampler"] == "balanced":
        w = (1.0 / counts)[ds.labels]
        gen = torch.Generator().manual_seed(cfg["seed"])
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                        replacement=True, generator=gen)
        dl = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["num_workers"],
                        pin_memory=(device.type == "cuda"), drop_last=True,
                        persistent_workers=cfg["num_workers"] > 0)
    else:
        gen = torch.Generator().manual_seed(cfg["seed"])
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, generator=gen,
                        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
                        drop_last=False, persistent_workers=cfg["num_workers"] > 0)
    class_w = None
    if cfg["class_weights"] == "balanced":
        class_w = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)

    model = build_model_from_config(cfg).to(device)
    log(f"Model: {type(model).__name__}({cfg['backbone']}, head={cfg['ce_head']}) "
        f"trainable params={count_parameters(model):,} | train samples={len(ds)} | steps/epoch={len(dl)}")
    opt = build_optimizer(model, cfg)
    steps_per_epoch = min(len(dl), cfg["max_steps"]) if cfg.get("max_steps") else len(dl)
    sched = build_scheduler(opt, cfg, steps_per_epoch)
    use_amp = _amp(device, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sel_key = cfg["select_metric"]
    start_epoch, best, best_epoch, history, bad_epochs = _try_resume(cfg, run_dir, model, opt, sched, scaler, log)
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t0, losses, correct, seen = time.time(), [], 0, 0
        for step, (x, y, _) in enumerate(dl):
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, weight=class_w,
                                   label_smoothing=cfg.get("label_smoothing", 0.0))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.get("grad_clip", 0) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
            correct += int((logits.argmax(1) == y).sum())
            seen += int(y.numel())
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else float("nan"),
               "train_acc": correct / max(1, seen), **current_lrs(opt),
               "train_time_s": round(time.time() - t0, 1)}
        if (epoch + 1) % cfg["val_every"] == 0 or epoch + 1 == cfg["epochs"]:
            v = evaluate_logits(model, val_records, label_map, store, device, cfg)
            row.update({"val_acc": v["accuracy"], "val_macro_f1": v["macro_f1"]})
        state, stop = _end_of_epoch(row, epoch, model, cfg, opt, sched, scaler,
                                    (best, best_epoch, bad_epochs, history), run_dir, log)
        best, best_epoch, bad_epochs, history = state
        if stop:
            break
        if cfg.get("debug_stop_after_epoch") and epoch + 1 >= cfg["debug_stop_after_epoch"]:
            log(f"DEBUG: simulated interruption after epoch {epoch + 1}")
            return None
    return _finish(run_dir, best, best_epoch, sel_key, history, log)
