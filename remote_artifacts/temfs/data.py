"""Data indexing, verification, loading and augmentation for the official TEM virus patch set.

Patch set layout (verified against the Mendeley listing of x4dwwfwtw3 v3):

    context_virus_1nm_256x256/
        train/<Class>/<img>_<vertex>.tif            (14 base classes)
        train/_EXCLUDED/<Class>/<img>_<vertex>.tif  (8 excluded classes)
        train/<Class>/crop_outlines/*.tif           (visualisations - NOT patches)
        validation/... test/...                     (same structure)
        augmented_train/<Class>/<img>_<vertex>[_<aug>].tif   (14 classes, 736 each)

<img> is the 1-based index of the source micrograph inside that split/class folder
(authors' prepareContextVirusImages.m), so (split, class, img) identifies one micrograph.
All patches from one micrograph must stay together (image-level grouping) to avoid leakage.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import (ALL_CLASSES, AUGMENTED_SPLIT, EXCLUDED_DIR, PATCH_ROOT_NAME, PATCH_SIZE,
                        SPLITS)

PATCH_RE = re.compile(r"^(\d+)_(\d+)\.tif$", re.IGNORECASE)
AUG_PATCH_RE = re.compile(r"^(\d+)_(\d+)(?:_(\d+))?\.tif$", re.IGNORECASE)
EXPECTED_COUNTS_FILE = Path(__file__).with_name("expected_counts.json")


# --------------------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchRecord:
    path: str
    split: str        # train | validation | test | augmented_train
    cls: str          # class folder name, e.g. "Nipah virus"
    image_id: int     # source micrograph index within (split, class)
    patch_id: int     # annotation vertex index within the micrograph
    aug_id: int = 0   # augmented_train suffix (0 = no suffix)

    @property
    def group(self) -> str:
        """Unique micrograph key. augmented_train patches come from the train micrographs."""
        split = "train" if self.split == AUGMENTED_SPLIT else self.split
        return f"{split}/{self.cls}/{self.image_id}"


def find_patch_root(data_root) -> Path:
    """Locate the 'context_virus_1nm_256x256' folder under data_root (searches 5 levels deep)."""
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if data_root.name == PATCH_ROOT_NAME and (data_root / "train").is_dir():
        return data_root
    found = []
    frontier = [data_root]
    for _ in range(5):
        nxt = []
        for d in frontier:
            try:
                children = [c for c in d.iterdir() if c.is_dir()]
            except PermissionError:
                continue
            for c in children:
                if c.name == PATCH_ROOT_NAME and (c / "train").is_dir():
                    found.append(c)
                else:
                    nxt.append(c)
        if found:
            break
        frontier = nxt
    if not found:
        raise FileNotFoundError(f"Could not find a '{PATCH_ROOT_NAME}' folder (with a 'train' "
                                f"sub-folder) under {data_root}. Did you download/extract the dataset?")
    if len(found) > 1:
        raise RuntimeError(f"Found several '{PATCH_ROOT_NAME}' folders, pass one explicitly: {found}")
    return found[0]


def _class_dirs(split_dir: Path):
    """Yield (class_name, class_dir) for normal and _EXCLUDED class folders of one split."""
    for entry in sorted(split_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == EXCLUDED_DIR:
            for sub in sorted(entry.iterdir()):
                if sub.is_dir():
                    yield sub.name, sub
        else:
            yield entry.name, entry


def build_index(patch_root, splits=SPLITS, include_augmented=False):
    """Scan the patch folders. Returns (records, report).

    report = {"unexpected_files": [...], "unknown_classes": [...], "missing_splits": [...]}
    Sub-folders inside class folders (crop_outlines) are skipped on purpose.
    """
    patch_root = Path(patch_root)
    splits = list(splits) + ([AUGMENTED_SPLIT] if include_augmented else [])
    records, unexpected, unknown, missing = [], [], set(), []
    for split in splits:
        split_dir = patch_root / split
        if not split_dir.is_dir():
            missing.append(split)
            continue
        regex = AUG_PATCH_RE if split == AUGMENTED_SPLIT else PATCH_RE
        for cls, cdir in _class_dirs(split_dir):
            if cls not in ALL_CLASSES:
                unknown.add(cls)
                continue
            for f in sorted(cdir.iterdir()):
                if f.is_dir():
                    continue  # crop_outlines etc.
                m = regex.match(f.name)
                if not m:
                    unexpected.append(str(f))
                    continue
                aug = int(m.group(3)) if (split == AUGMENTED_SPLIT and m.group(3)) else 0
                records.append(PatchRecord(str(f), split, cls, int(m.group(1)), int(m.group(2)), aug))
    report = {"unexpected_files": unexpected, "unknown_classes": sorted(unknown), "missing_splits": missing}
    return records, report


def load_expected_counts():
    with open(EXPECTED_COUNTS_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)["counts"]


def count_index(records):
    """{class: {split: {"patches": n, "images": n, "image_ids": sorted list}}}"""
    patches = defaultdict(int)
    images = defaultdict(set)
    for r in records:
        patches[(r.cls, r.split)] += 1
        images[(r.cls, r.split)].add(r.image_id)
    out = defaultdict(dict)
    for (cls, split), n in patches.items():
        ids = sorted(images[(cls, split)])
        out[cls][split] = {"patches": n, "images": len(ids), "image_ids": ids}
    return out


def verify_counts(records, expected=None, splits=SPLITS):
    """Compare indexed counts with the expected official counts. Returns list of problem strings."""
    expected = expected or load_expected_counts()
    got = count_index(records)
    problems = []
    for cls in ALL_CLASSES:
        for split in splits:
            exp = expected[cls][split]
            g = got.get(cls, {}).get(split)
            if g is None:
                problems.append(f"{split}/{cls}: missing (expected {exp['patches']} patches)")
                continue
            if g["patches"] != exp["patches"]:
                problems.append(f"{split}/{cls}: {g['patches']} patches, expected {exp['patches']}")
            if g["images"] != exp["images"]:
                problems.append(f"{split}/{cls}: {g['images']} micrographs, expected {exp['images']}")
            if g["image_ids"] != list(range(1, g["images"] + 1)):
                problems.append(f"{split}/{cls}: micrograph ids are not contiguous 1..{g['images']}")
    return problems


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_cross_split_duplicates(records):
    """Byte-identical patch files that appear in more than one split (possible leakage)."""
    by_hash = defaultdict(list)
    for r in records:
        if r.split == AUGMENTED_SPLIT:
            continue
        by_hash[file_md5(r.path)].append(r)
    dups = []
    for h, rs in by_hash.items():
        if len({r.split for r in rs}) > 1:
            dups.append([f"{r.split}/{r.cls}/{Path(r.path).name}" for r in rs])
    return dups


def select_records(records, splits, classes):
    """Filter records by split(s) and class list (order of records preserved)."""
    splits = {splits} if isinstance(splits, str) else set(splits)
    classes = set(classes)
    return [r for r in records if r.split in splits and r.cls in classes]


# --------------------------------------------------------------------------------------
# Image loading / pre-processing
# --------------------------------------------------------------------------------------
def load_patch(path) -> np.ndarray:
    """Load one patch as uint8 (256, 256). Official patches are 8-bit grayscale TIFF."""
    with Image.open(path) as im:
        if im.mode != "L":
            im = im.convert("L")
        arr = np.asarray(im, dtype=np.uint8).copy()
    if arr.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"{path}: expected {PATCH_SIZE}x{PATCH_SIZE}, got {arr.shape}")
    return arr


def zscore(img: np.ndarray) -> np.ndarray:
    """Per-patch standardisation exactly as the authors' normalizeImage (img_as_float, -mean, /std)."""
    x = img.astype(np.float32) / 255.0
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-6:
        return np.zeros_like(x)
    return (x - mean) / std


def d4_transform(img: np.ndarray, k: int) -> np.ndarray:
    """Element k (0..7) of the dihedral group D4: rotate by k%4 * 90 deg, then mirror if k >= 4.
    Exact (no interpolation), so the 1 nm/pixel scale and particle size are preserved."""
    out = np.rot90(img, k % 4)
    if k >= 4:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def random_translate(img: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """Shift by up to +-max_shift pixels using symmetric padding (as the authors' padarray 'symmetric')."""
    if max_shift <= 0:
        return img
    h, w = img.shape
    dy, dx = rng.integers(-max_shift, max_shift + 1, size=2)
    padded = np.pad(img, max_shift, mode="symmetric")
    y0, x0 = max_shift + int(dy), max_shift + int(dx)
    return np.ascontiguousarray(padded[y0:y0 + h, x0:x0 + w])


class PatchStore:
    """Optionally keeps every patch in RAM as uint8 (10.6k patches ~ 0.7 GB)."""

    def __init__(self, records, preload=True, verbose=True):
        self.paths = [r.path for r in records]
        self.index = {p: i for i, p in enumerate(self.paths)}
        self.array = None
        if preload:
            self.array = np.empty((len(self.paths), PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
            for i, p in enumerate(self.paths):
                self.array[i] = load_patch(p)
                if verbose and (i + 1) % 2000 == 0:
                    print(f"  loaded {i + 1}/{len(self.paths)} patches", flush=True)

    def get(self, path) -> np.ndarray:
        if self.array is not None:
            return self.array[self.index[path]]
        return load_patch(path)


class PatchDataset(Dataset):
    """Returns (tensor[1,256,256] float32 z-scored, label, dataset_index)."""

    def __init__(self, records, label_map, store: PatchStore | None = None, augment=False,
                 max_shift=0, seed=0):
        self.records = list(records)
        self.label_map = dict(label_map)
        self.labels = np.array([self.label_map[r.cls] for r in self.records], dtype=np.int64)
        self.groups = [r.group for r in self.records]
        self.store = store
        self.augment = augment
        self.max_shift = max_shift
        self.seed = seed
        self._rng = None

    def __len__(self):
        return len(self.records)

    def _get_rng(self):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng([self.seed, wid, int(torch.initial_seed() % (2 ** 31))])
        return self._rng

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = self.store.get(rec.path) if self.store is not None else load_patch(rec.path)
        if self.augment:
            rng = self._get_rng()
            img = d4_transform(img, int(rng.integers(0, 8)))
            img = random_translate(img, self.max_shift, rng)
        x = torch.from_numpy(zscore(img)).unsqueeze(0)
        return x, int(self.labels[idx]), idx
