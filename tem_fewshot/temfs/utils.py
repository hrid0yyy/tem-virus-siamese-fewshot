"""Small helpers: seeding, device, logging, JSON."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False. "
                           "Install a CUDA build of PyTorch (see README).")
    return dev


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=_NumpyEncoder)
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Logger:
    """Print to stdout and append to a log file."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def append_csv(path, row: dict):
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def write_confusion_csv(path, cm, class_names):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([""] + list(class_names))
        for name, row in zip(class_names, cm):
            w.writerow([name] + list(map(int, row)))


def environment_info():
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:  # pragma: no cover
        info["torchvision"] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        info["cuda_runtime"] = torch.version.cuda
    return info
