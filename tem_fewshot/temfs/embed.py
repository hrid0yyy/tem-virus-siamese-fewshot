"""Embedding / logit extraction with optional D4 test-time augmentation (8 rotations+flips)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constants import ALL_CLASSES
from .data import PatchDataset
from .models import EmbeddingNet, GAPClassifier, PaperClassifier, to_three_channels


def d4_torch(x, k):
    """Element k (0..7) of D4 on a (B,C,H,W) tensor (same convention as data.d4_transform)."""
    out = torch.rot90(x, k % 4, dims=(2, 3))
    if k >= 4:
        out = torch.flip(out, dims=(3,))
    return out


def _forward_parts(model, x):
    """Returns dict with 'feat' (GAP features) and 'head' (embedding) or 'logits'."""
    if isinstance(model, EmbeddingNet):
        f = model.features(x)
        return {"feat": f, "head": model.embed_from_features(f)}
    if isinstance(model, PaperClassifier):
        fmap = model.backbone(to_three_channels(x))
        return {"feat": torch.flatten(model.pool(fmap), 1), "logits": model.classifier(fmap)}
    if isinstance(model, GAPClassifier):
        f = model.features(x)
        return {"feat": f, "logits": model.classifier(f)}
    raise TypeError(f"unsupported model type {type(model)}")


@torch.no_grad()
def extract(model, records, store, device, batch_size=64, tta=False, amp=True, num_workers=0):
    """Returns dict of numpy arrays aligned with `records`:
       'feat'  : L2-normalised GAP backbone features
       'head'  : L2-normalised embedding (metric models)
       'logits': class logits (classifier models; averaged over TTA views)
    """
    model.eval()
    ds = PatchDataset(records, {c: i for i, c in enumerate(ALL_CLASSES)}, store=store, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=(device.type == "cuda"))
    views = range(8) if tta else range(1)
    use_amp = amp and device.type == "cuda"
    out = {}
    for x, _, _ in dl:
        x = x.to(device, non_blocking=True)
        acc = {}
        for k in views:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                parts = _forward_parts(model, d4_torch(x, k))
            for key, v in parts.items():
                v = v.float()
                if key == "head":
                    v = F.normalize(v, dim=1)
                acc[key] = acc.get(key, 0) + v
        for key, v in acc.items():
            v = v / len(views)
            if key in ("feat", "head"):
                v = F.normalize(v, dim=1)
            out.setdefault(key, []).append(v.cpu().numpy())
    return {k: np.concatenate(v, 0).astype(np.float32) for k, v in out.items()}
