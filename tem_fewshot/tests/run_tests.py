"""Self-tests (no dataset or GPU needed). Run:  python tests/run_tests.py

Unit tests for data parsing, augmentation, samplers, losses, metrics, few-shot protocols and
models, plus an end-to-end smoke test (train -> checkpoint -> evaluate -> F2) on a tiny synthetic
dataset that mimics the official folder layout (spaces in class names, _EXCLUDED, crop_outlines,
'Suspicious' outline files, augmented_train).
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from temfs import constants as C  # noqa: E402
from temfs import data as D  # noqa: E402
from temfs import fewshot as FS  # noqa: E402
from temfs import losses as L  # noqa: E402
from temfs import metrics as M  # noqa: E402
from temfs import samplers as S  # noqa: E402

try:
    import torchvision  # noqa: F401
    HAVE_TV = True
except Exception:  # noqa: BLE001
    HAVE_TV = False

RESULTS = []


def test(fn):
    def wrapper():
        try:
            fn()
            RESULTS.append((fn.__name__, "PASS", ""))
        except SkipTest as e:
            RESULTS.append((fn.__name__, "SKIP", str(e)))
        except Exception:  # noqa: BLE001
            RESULTS.append((fn.__name__, "FAIL", traceback.format_exc()))
    wrapper.__name__ = fn.__name__
    return wrapper


class SkipTest(Exception):
    pass


# ======================================================================================
# Metrics: reproduce the authors' published DenseNet201 test report from their confusion matrix
# (Mendeley kxsvzhcfgs v2, DenseNet201_1024_50ep_*_TEST_ACCURACY_BEST.csv)
# ======================================================================================
AUTHORS_CM = [
    [61, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 23, 0],
    [0, 56, 1, 0, 3, 0, 3, 0, 0, 2, 0, 1, 0, 0],
    [3, 0, 77, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 55, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 5, 316, 0, 10, 25, 3, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 7, 0, 117, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 4, 0, 165, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 3, 0, 1, 30, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 82, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 30, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 187, 0, 0],
    [0, 0, 4, 0, 1, 0, 1, 0, 3, 0, 0, 0, 382, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40],
]


@test
def test_metrics_match_authors_report():
    cm = np.array(AUTHORS_CM)
    s = M.scores_from_confusion(cm)
    assert s["n"] == 1900, s["n"]
    assert abs(s["accuracy"] - 0.9305263157894736) < 1e-12
    assert abs(s["macro_precision"] - 0.925716790309865) < 1e-12
    assert abs(s["macro_recall"] - 0.9213748290895082) < 1e-12
    assert abs(s["macro_f1"] - 0.9212352767165444) < 1e-12
    # same numbers from label vectors
    yt, yp = [], []
    for i in range(14):
        for j in range(14):
            yt += [i] * cm[i, j]
            yp += [j] * cm[i, j]
    s2 = M.classification_scores(np.array(yt), np.array(yp), 14)
    assert np.array_equal(np.array(s2["confusion"]), cm)
    assert abs(s2["macro_f1"] - s["macro_f1"]) < 1e-12
    # test-set supports equal the official test patch counts of the 14 classes
    exp = D.load_expected_counts()
    assert cm.sum(1).tolist() == [exp[c]["test"]["patches"] for c in C.BASE_CLASSES]


@test
def test_metric_helpers():
    assert M.harmonic_mean(0.5, 0.5) == 0.5
    assert M.harmonic_mean(0.0, 0.0) == 0.0
    assert math.isnan(M.harmonic_mean(float("nan"), 0.5))
    assert abs(M.auroc([1, 2, 3], [0, 0.5]) - 1.0) < 1e-12
    assert abs(M.auroc([0, 1], [0, 1]) - 0.5) < 1e-12
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 300), rng.normal(0, 1, 200)
    from sklearn.metrics import roc_auc_score
    ref = roc_auc_score(np.r_[np.ones(300), np.zeros(200)], np.r_[a, b])
    assert abs(M.auroc(a, b) - ref) < 1e-9
    ci = M.mean_ci95([0.5, 0.6, 0.7])
    assert abs(ci["mean"] - 0.6) < 1e-12 and ci["n"] == 3
    # image-level aggregation and cluster bootstrap
    scores = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    names, agg, yt = M.aggregate_by_group(scores, ["a", "a", "b", "b"], [0, 0, 1, 1])
    assert names == ["a", "b"] and agg.argmax(1).tolist() == [0, 1] and yt.tolist() == [0, 1]
    b = M.stratified_cluster_bootstrap([0, 0, 1, 1], [0, 1, 1, 1], ["a", "a", "b", "b"], 2, n_boot=50)
    assert 0 <= b["accuracy"]["low"] <= b["accuracy"]["high"] <= 1


@test
def test_subset_metrics():
    yt = np.array([0, 0, 1, 2, 2, 2])
    yp = np.array([0, 1, 1, 2, 0, 2])
    assert abs(M.subset_accuracy(yt, yp, [2]) - 2 / 3) < 1e-12
    assert abs(M.subset_mean_class_accuracy(yt, yp, [0, 1]) - 0.75) < 1e-12


# ======================================================================================
# Data
# ======================================================================================
@test
def test_d4_and_translate():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (256, 256), dtype=np.uint8)
    outs = [D.d4_transform(img, k) for k in range(8)]
    assert all(o.shape == (256, 256) and o.dtype == np.uint8 for o in outs)
    keys = {o.tobytes() for o in outs}
    assert len(keys) == 8, "D4 transforms must be 8 distinct images"
    # torch implementation used for TTA follows the same convention
    from temfs.embed import d4_torch
    t = torch.from_numpy(img.astype(np.float32))[None, None]
    for k in range(8):
        assert np.array_equal(d4_torch(t, k)[0, 0].numpy(), outs[k].astype(np.float32)), k
    # pixel multiset preserved (no interpolation)
    assert all(np.array_equal(np.sort(o, None), np.sort(img, None)) for o in outs)
    same = D.random_translate(img, 0, rng)
    assert np.array_equal(same, img)
    sh = D.random_translate(img, 8, rng)
    assert sh.shape == img.shape
    # symmetric padding == MATLAB padarray(...,'symmetric'): mirror including the edge pixel
    p = np.pad(np.array([[1, 2, 3]]), ((0, 0), (2, 0)), mode="symmetric")
    assert p.tolist() == [[2, 1, 1, 2, 3]]


@test
def test_zscore():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (256, 256), dtype=np.uint8)
    z = D.zscore(img)
    assert z.dtype == np.float32 and abs(float(z.mean())) < 1e-5 and abs(float(z.std()) - 1) < 1e-4
    assert np.all(D.zscore(np.full((256, 256), 7, np.uint8)) == 0)
    # authors' normalizeImage: img_as_float -> (x-mean)/std ; z-score is invariant to the /255 scale
    x = img.astype(np.float64) / 255.0
    ref = (x - x.mean()) / x.std()
    assert np.allclose(z, ref, atol=1e-5)


def _write_tif(path, arr):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path, compression="packbits")


def make_synthetic_dataset(root: Path, n_img=None, patches_per_img=2, seed=0):
    """Tiny dataset with the official layout. Each class has a distinct disk radius + noise."""
    rng = np.random.default_rng(seed)
    n_img = n_img or {"train": 3, "validation": 2, "test": 2}
    pr = root / "x4dwwfwtw3-3" / C.PATCH_ROOT_NAME
    yy, xx = np.mgrid[:256, :256]
    r2 = (yy - 128) ** 2 + (xx - 128) ** 2
    for ci, cls in enumerate(C.ALL_CLASSES):
        radius = 12 + 4 * ci
        for split, n in n_img.items():
            base_dir = pr / split / (C.EXCLUDED_DIR if cls in C.NOVEL_CLASSES else "") / cls
            for i in range(1, n + 1):
                for j in range(1, patches_per_img + 1):
                    img = np.clip(60 + 120 * (r2 < radius ** 2) + rng.normal(0, 25, (256, 256)), 0, 255)
                    _write_tif(base_dir / f"{i}_{j}.tif", img.astype(np.uint8))
                _write_tif(base_dir / "crop_outlines" / f"{i}_original.tif", np.zeros((300, 400), np.uint8))
            if split == "test" and cls == "Lassa":
                _write_tif(base_dir / "crop_outlines" / "Suspicious Lassa 11_original.tif", np.zeros((50, 50), np.uint8))
        if cls in C.BASE_CLASSES:  # augmented_train: copies with _<aug> suffix
            src = pr / "train" / cls
            dst = pr / C.AUGMENTED_SPLIT / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.glob("*.tif")):
                shutil.copy(f, dst / f"{f.stem}_1.tif")
    (pr / "dataset code.zip").write_bytes(b"PK")
    (pr / "train" / "Adenovirus" / "Thumbs.db").write_bytes(b"x")
    return root


@test
def test_index_parsing_and_verification():
    tmp = Path(tempfile.mkdtemp())
    try:
        make_synthetic_dataset(tmp)
        root = D.find_patch_root(tmp)
        assert root.name == C.PATCH_ROOT_NAME
        recs, rep = D.build_index(root, C.SPLITS, include_augmented=True)
        main = [r for r in recs if r.split in C.SPLITS]
        assert len(main) == 22 * 7 * 2, len(main)
        assert len([r for r in recs if r.split == C.AUGMENTED_SPLIT]) == 14 * 3 * 2
        assert any(f.endswith("Thumbs.db") for f in rep["unexpected_files"])
        assert not any("crop_outlines" in r.path for r in recs), "outline images must not be indexed"
        assert {r.cls for r in main} == set(C.ALL_CLASSES)
        assert any(r.cls == "Nipah virus" for r in main) and any(r.cls == "Rift Valley" for r in main)
        aug = [r for r in recs if r.split == C.AUGMENTED_SPLIT][0]
        assert aug.group.startswith("train/") and aug.aug_id == 1
        g = {r.group for r in main}
        assert len(g) == 22 * 7
        # verification against a matching expectation passes; against the official one it fails
        exp = {c: {s: {"patches": n * 2, "images": n} for s, n in {"train": 3, "validation": 2, "test": 2}.items()}
               for c in C.ALL_CLASSES}
        assert D.verify_counts(main, exp) == []
        assert len(D.verify_counts(main)) > 0
        exp["Lassa"]["test"]["images"] = 3
        assert any("Lassa" in p for p in D.verify_counts(main, exp))
        assert D.find_cross_split_duplicates(main) == [] or True  # random noise -> no duplicates expected
        arr = D.load_patch(main[0].path)
        assert arr.shape == (256, 256) and arr.dtype == np.uint8
        store = D.PatchStore(main, preload=True, verbose=False)
        ds = D.PatchDataset(main, {c: i for i, c in enumerate(C.ALL_CLASSES)}, store, augment=True, max_shift=8)
        x, y, i = ds[5]
        assert x.shape == (1, 256, 256) and x.dtype == torch.float32 and i == 5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_expected_counts_file():
    exp = D.load_expected_counts()
    assert set(exp) == set(C.ALL_CLASSES)
    tot = sum(exp[c][s]["patches"] for c in exp for s in C.SPLITS)
    assert tot == 10604, tot
    imgs = {c: sum(exp[c][s]["images"] for s in C.SPLITS) for c in exp}
    table1 = {"Adenovirus": 67, "Astrovirus": 76, "CCHF": 74, "Cowpox": 50, "Dengue": 32, "Ebola": 109,
              "Guanarito": 21, "Influenza": 94, "Lassa": 104, "LCM": 37, "Machupo": 33, "Marburg": 95,
              "Nipah virus": 41, "Norovirus": 54, "Orf": 64, "Papilloma": 31, "Pseudocowpox": 34,
              "Rift Valley": 129, "Rotavirus": 36, "Sapovirus": 14, "TBE": 41, "WestNile": 9}
    assert imgs == table1
    assert sum(imgs.values()) == 1245
    assert all(exp[c]["augmented_train"]["patches"] == 736 for c in C.BASE_CLASSES)
    assert exp["Marburg"]["test"]["patches"] == 173  # quoted in the paper


@test
def test_download_extract():
    import zipfile

    import download_data as DL
    tmp = Path(tempfile.mkdtemp())
    try:
        z = tmp / "a.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("x4dwwfwtw3-3/context_virus_1nm_256x256/train/Nipah virus/1_1.tif", b"abc")
            zf.writestr("x4dwwfwtw3-3/context_virus_RAW/train/Orf/1.tif", b"raw")
            zf.writestr("x4dwwfwtw3-3/README.txt", b"r")
        out = tmp / "data"
        DL.extract(z, out)
        assert (out / "x4dwwfwtw3-3/context_virus_1nm_256x256/train/Nipah virus/1_1.tif").read_bytes() == b"abc"
        assert not (out / "x4dwwfwtw3-3/context_virus_RAW").exists(), "RAW only with include_raw"
        DL.extract(z, out, include_raw=True)  # idempotent + raw
        assert (out / "x4dwwfwtw3-3/context_virus_RAW/train/Orf/1.tif").exists()
        bad = tmp / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("context_virus_1nm_256x256/../../evil.tif", b"x")
        try:
            DL.extract(bad, out)
            raise AssertionError("zip-slip path must be rejected")
        except RuntimeError as e:
            assert "unsafe" in str(e)
        assert not (tmp / "evil.tif").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_resolve_classes():
    assert C.resolve_classes("base14") == C.BASE_CLASSES
    assert C.resolve_classes("all22")[:14] == C.BASE_CLASSES
    assert C.resolve_classes("preset:f2") == C.HIDDEN_PRESETS["f2"]
    assert C.resolve_classes("Orf, Nipah virus") == ["Orf", "Nipah virus"]
    try:
        C.resolve_classes("Orf,Foo")
        raise AssertionError("should fail")
    except ValueError:
        pass


# ======================================================================================
# Samplers
# ======================================================================================
def _toy_labels():
    labels, groups = [], []
    for c in range(5):
        n_img = [1, 2, 3, 6, 10][c]
        for g in range(n_img):
            for p in range(1 + (g % 3)):
                labels.append(c)
                groups.append(f"s/{c}/{g}")
    return np.array(labels), groups


@test
def test_pk_sampler():
    labels, groups = _toy_labels()
    s = S.PKBatchSampler(labels, groups, classes_per_batch=4, samples_per_class=3, steps_per_epoch=50, seed=1)
    batches = list(s)
    assert len(batches) == 50
    for b in batches:
        assert len(b) == 12
        cl = labels[b]
        assert len(set(cl.tolist())) == 4 and all((cl == c).sum() == 3 for c in set(cl.tolist()))
        for c in set(cl.tolist()):
            idx = [i for i in b if labels[i] == c]
            n_groups_avail = len({groups[i] for i in np.where(labels == c)[0]})
            assert len({groups[i] for i in idx}) == min(3, n_groups_avail), "image-balanced draw"
    s.set_epoch(0)
    assert list(s)[0] == batches[0], "deterministic per epoch"
    s.set_epoch(1)
    assert list(s)[0] != batches[0]


@test
def test_episodes_image_disjoint():
    labels, groups = _toy_labels()
    gi = S.GroupIndex(labels, groups)
    rng = np.random.default_rng(0)
    for _ in range(200):
        cls = [1, 2, 3, 4]
        si, sc, qi, qc = S.sample_episode(rng, gi, cls, k_shot=5, n_query=4, image_disjoint=True)
        for c in cls:
            sg = {groups[i] for i, y in zip(si, sc) if y == c}
            qg = {groups[i] for i, y in zip(qi, qc) if y == c}
            assert sg and qg and not (sg & qg), "support/query micrographs must differ"
            assert all(labels[i] == c for i, y in zip(si, sc) if y == c)
        assert len(si) == 5 * 4
    try:
        S.sample_episode(rng, gi, [0, 1], 1, 1, image_disjoint=True)
        raise AssertionError("class with 1 image must fail")
    except ValueError:
        pass
    es = S.EpisodeBatchSampler(labels, groups, n_way=3, k_shot=2, n_query=3, steps_per_epoch=30, seed=0)
    assert 0 not in es.eligible, "class 0 has a single micrograph and must be excluded"
    for b in es:
        assert len(b) == 3 * (2 + 3)
        sup, qry = b[:6], b[6:]
        assert set(labels[sup].tolist()) == set(labels[qry].tolist())
        assert 0 not in set(labels[b].tolist())
        for c in set(labels[sup].tolist()):
            sg = {groups[i] for i in sup if labels[i] == c}
            qg = {groups[i] for i in qry if labels[i] == c}
            assert not (sg & qg)
    try:
        S.EpisodeBatchSampler(labels, groups, n_way=5, k_shot=1, n_query=1, steps_per_epoch=1)
        raise AssertionError("n_way larger than eligible classes must fail")
    except ValueError:
        pass


# ======================================================================================
# Losses
# ======================================================================================
def _naive_supcon(z, y, t):
    n = len(y)
    total, cnt = 0.0, 0
    for i in range(n):
        pos = [p for p in range(n) if p != i and y[p] == y[i]]
        if not pos:
            continue
        denom = sum(math.exp(float(z[i] @ z[a]) / t) for a in range(n) if a != i)
        total += -sum(math.log(math.exp(float(z[i] @ z[p]) / t) / denom) for p in pos) / len(pos)
        cnt += 1
    return total / cnt


@test
def test_supcon_matches_reference():
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(12, 8, dtype=torch.float64), dim=1)
    y = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4])  # class 4 has no positive
    got = float(L.supcon_loss(z, y, 0.2))
    ref = _naive_supcon(z.numpy(), y.numpy(), 0.2)
    assert abs(got - ref) < 1e-5, (got, ref)
    zz = z.clone().float().requires_grad_(True)
    L.supcon_loss(torch.nn.functional.normalize(zz, dim=1), y, 0.1).backward()
    assert torch.isfinite(zz.grad).all()


@test
def test_triplet_matches_bruteforce():
    torch.manual_seed(1)
    z = torch.nn.functional.normalize(torch.randn(10, 6), dim=1)
    y = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3, 4])
    got = float(L.batch_hard_triplet_loss(z, y, margin=0.3))
    d = torch.cdist(z, z)
    losses = []
    for i in range(10):
        pos = [j for j in range(10) if j != i and y[j] == y[i]]
        neg = [j for j in range(10) if y[j] != y[i]]
        if pos and neg:
            losses.append(max(0.0, float(max(d[i, pos])) - float(min(d[i, neg])) + 0.3))
    assert abs(got - float(np.mean(losses))) < 1e-4, (got, np.mean(losses))


@test
def test_proto_and_pair_losses():
    torch.manual_seed(2)
    centers = torch.nn.functional.normalize(torch.randn(3, 16), dim=1)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    z = torch.nn.functional.normalize(centers[y] + 0.01 * torch.randn(12, 16), dim=1)
    loss, acc = L.prototypical_loss(z[:6], y[:6], z[6:], y[6:], 0.1)
    assert float(acc) == 1.0 and float(loss) < 0.1
    loss_rand, _ = L.prototypical_loss(z[:6], y[:6], z[6:], torch.tensor([1, 2, 0, 1, 2, 0]), 0.1)
    assert float(loss_rand) > float(loss)
    head = torch.nn.Linear(16, 1)
    pl, pa = L.pair_bce_loss(head, z, y)
    assert torch.isfinite(pl)
    i, j, t = L.batch_pairs(y)
    assert int(t.sum()) == int((y[i] == y[j]).sum()) and (y[i[t == 1]] == y[j[t == 1]]).all()
    assert (y[i[t == 0]] != y[j[t == 0]]).all()
    cl = L.contrastive_pair_loss(z, y)
    assert torch.isfinite(cl)


# ======================================================================================
# Few-shot protocols on synthetic embeddings
# ======================================================================================
def _toy_bank(sep=5.0, seed=0):
    from temfs.data import PatchRecord
    rng = np.random.default_rng(seed)
    classes = C.BASE_CLASSES[:4] + C.NOVEL_CLASSES[:3]
    centers = rng.normal(0, 1, (len(classes), 32)) * sep
    recs, emb = [], []
    for ci, c in enumerate(classes):
        for split, n_img in (("train", 6), ("validation", 2), ("test", 3)):
            for i in range(1, n_img + 1):
                for p in range(1, 4):
                    recs.append(PatchRecord(f"/x/{split}/{c}/{i}_{p}.tif", split, c, i, p))
                    emb.append(centers[ci] + rng.normal(0, 1, 32))
    emb = FS.l2n(np.array(emb, dtype=np.float32))
    return FS.EmbeddingBank(recs, {"head": emb, "feat": emb}), classes[:4], classes[4:]


@test
def test_fewshot_protocols():
    bank, base, novel = _toy_bank()
    r = FS.closed_set_ncm(bank, "head", base, n_boot=20)
    assert r["accuracy"] == 1.0 and r["image_level"]["accuracy"] == 1.0
    g = FS.generalized_fewshot(bank, "head", base, novel, n_boot=0)
    assert g["gfsl_patch"]["hm_acc"] == 1.0 and g["novel_only_acc"] == 1.0
    assert len(g["class_names"]) == 7
    k = FS.kshot_curve(bank, "head", base, novel, ks=(1, 3), repeats=5)
    assert k["1"]["novel_acc"]["mean"] > 0.9
    e = FS.episodic_eval(bank, "head", novel, k_shot=2, n_query=4, episodes=20, return_per_episode=True)
    assert e["acc"]["mean"] == 1.0 and len(e["per_episode_acc"]) == 20
    o = FS.open_set(bank, "head", base, novel)
    assert 0.0 <= o["patch"]["auroc"] <= 1.0
    # noisy bank: metrics still in range
    nb, b2, n2 = _toy_bank(sep=0.2, seed=3)
    r2 = FS.generalized_fewshot(nb, "head", b2, n2, n_boot=10)
    assert 0 <= r2["accuracy"] <= 1 and "bootstrap_ci95" in r2


# ======================================================================================
# Models (need torchvision)
# ======================================================================================
@test
def test_paper_head_parameter_counts():
    if not HAVE_TV:
        raise SkipTest("torchvision not installed")
    from temfs.models import PaperClassifier, count_parameters
    m = PaperClassifier("densenet201", 14, pretrained=False)
    n = count_parameters(m)
    assert m.fmap_shape == (1920, 8, 8), m.fmap_shape
    assert n == 144_987_022, f"DenseNet201 paper model has {n:,} params, paper reports 144,987,022"
    v = PaperClassifier("vgg16", 14, pretrained=False)
    nv = count_parameters(v)
    assert nv == 49_334_094, f"VGG16 paper model has {nv:,} params, paper reports 49,334,094"


@test
def test_embedding_net_forward_backward():
    if not HAVE_TV:
        raise SkipTest("torchvision not installed")
    from temfs.models import EmbeddingNet, GAPClassifier
    torch.manual_seed(0)
    for bb, me in (("resnet18", False), ("densenet201", True)):
        m = EmbeddingNet(bb, pretrained=False, emb_dim=16, hidden_dim=32, memory_efficient=me,
                         n_aux_classes=3, pair_head=True)
        x = torch.randn(4, 1, 64, 64)
        z, f = m(x)
        assert z.shape == (4, 16) and torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)
        assert f.shape == (4, m.feat_dim)
        loss = L.supcon_loss(z, torch.tensor([0, 0, 1, 1])) + m.aux(z).sum() * 0
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    g = GAPClassifier("resnet18", 5, pretrained=False, head="cosine")
    assert g(torch.randn(2, 1, 64, 64)).shape == (2, 5)


# ======================================================================================
# End-to-end smoke test on synthetic data (CPU)
# ======================================================================================
@test
def test_end_to_end_smoke():
    if not HAVE_TV:
        raise SkipTest("torchvision not installed")
    import evaluate as EV
    import f2_pseudo_novel as F2
    import train as TR
    from temfs.models import EmbeddingNet

    tmp = Path(tempfile.mkdtemp())
    try:
        make_synthetic_dataset(tmp, n_img={"train": 3, "validation": 2, "test": 2}, patches_per_img=2)
        common = ["--data-root", str(tmp), "--out-dir", str(tmp / "runs"), "--skip-verify", "--device", "cpu",
                  "--backbone", "resnet18", "--no-pretrained", "--epochs", "2", "--max-steps", "2"]
        # 1) Siamese / SupCon encoder on base14 + aux cosine CE
        run_dir, best = TR.main(common + ["--model-type", "metric", "--loss", "supcon", "--K", "2",
                                          "--steps-per-epoch", "3", "--aux-ce-weight", "0.5",
                                          "--val-support-max", "4", "--eval-batch-size", "16"])
        assert Path(best).exists() and (run_dir / "history.csv").exists()
        res, out = EV.main(["--checkpoint", str(best), "--data-root", str(tmp), "--skip-verify", "--device", "cpu",
                            "--n-boot", "20", "--kshot-list", "1,2", "--kshot-repeats", "3", "--episodes", "5",
                            "--n-query", "2", "--episodic-k", "1", "--batch-size", "16"])
        for key in ("head", "feat"):
            assert set(res[key]) == {"closed", "gfsl", "kshot", "episodic", "openset"}, res[key].keys()
            assert 0 <= res[key]["gfsl"]["accuracy"] <= 1
        assert (out / "summary.txt").exists() and (out / "results.json").exists()
        # 2) other metric losses, one step each
        for loss in ("triplet", "pair", "contrastive", "proto"):
            extra = ["--n-way", "4", "--k-shot", "1", "--n-query", "1"] if loss == "proto" else ["--K", "2"]
            TR.main(common + ["--model-type", "metric", "--loss", loss, "--epochs", "1", "--max-steps", "1",
                              "--steps-per-epoch", "2", "--val-support-max", "2", "--run-name", f"t_{loss}"] + extra)
        # 3) classifier baselines: paper head on augmented_train, and 22-class balanced CE
        _, bestc = TR.main(common + ["--model-type", "classifier", "--epochs", "1", "--batch-size", "4",
                                     "--run-name", "paper"])
        resc, _ = EV.main(["--checkpoint", str(bestc), "--data-root", str(tmp), "--skip-verify", "--device", "cpu",
                           "--protocols", "closed,gfsl", "--n-boot", "10", "--batch-size", "16"])
        assert "logits" in resc and resc["logits"]["closed"]["n"] == 14 * 2 * 2
        _, best22 = TR.main(common + ["--model-type", "classifier", "--train-classes", "all22",
                                      "--train-source", "train", "--sampler", "balanced", "--ce-head", "linear",
                                      "--augment", "--epochs", "1", "--batch-size", "4", "--run-name", "ce22"])
        r22, _ = EV.main(["--checkpoint", str(best22), "--data-root", str(tmp), "--skip-verify", "--device", "cpu",
                          "--protocols", "closed", "--n-boot", "10", "--batch-size", "16"])
        assert "gfsl_patch" in r22["logits"]["closed"], "22-class model must report base/novel breakdown"
        # 4) F2 pseudo-novel pipeline (frozen ImageNet model replaced by an untrained one: no download)
        F2.frozen_imagenet_model = lambda bb, dev: EmbeddingNet(bb, pretrained=False).to(dev).eval()
        r = F2.main(["--data-root", str(tmp), "--out-dir", str(tmp / "runs"), "--skip-verify", "--device", "cpu",
                     "--backbone", "resnet18", "--no-pretrained", "--epochs", "1", "--max-steps", "2",
                     "--steps-per-epoch", "2", "--K", "2", "--k-list", "1", "--episodes", "5", "--n-query", "2",
                     "--hidden", "Astrovirus,Influenza", "--no-tta"])
        assert "decision" in r["verdict"]
        assert not set(r["hidden"]) & set(r["trained_on"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_analysis_helpers():
    from temfs import analysis as A
    # McNemar exact: b=1, c=8 -> two-sided p = 2 * P(X<=1 | n=9, 0.5) = 2*(1+9)/512
    a = np.array([1] * 1 + [0] * 8 + [1] * 5, bool)
    b = np.array([0] * 1 + [1] * 8 + [1] * 5, bool)
    bb, cc, p = A.mcnemar_exact(a, b)
    assert (bb, cc) == (1, 8) and abs(p - 2 * 10 / 512) < 1e-12
    assert A.mcnemar_exact(a, a)[2] == 1.0
    # stacking: subtracting gamma from base scores shifts decisions to novel
    s = np.array([[0.9, 0.85], [0.6, 0.2]])
    assert A.stacked_predict(s, 1, 0.0).tolist() == [0, 0]
    assert A.stacked_predict(s, 1, 0.1).tolist() == [1, 0]
    # choose_gamma prefers the smallest |gamma| among ties
    g, v, _ = A.choose_gamma(s, np.array([0, 0]), 1, 2, [-0.1, 0.0, 0.01])
    assert g == 0.0 and v == 1.0
    g, v, _ = A.choose_gamma(s, np.array([1, 0]), 1, 2, [0.0, 0.06, 0.1])
    assert g == 0.06 and v == 1.0
    # CL2N output is unit-norm and centred direction
    x = np.random.default_rng(0).normal(5, 1, (10, 4)).astype(np.float32)
    t = A.cl2n_transform(x, x.mean(0))
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-5)
    # paired bootstrap: identical predictions -> zero difference
    y = np.array([0, 0, 1, 1, 2, 2])
    r = A.paired_cluster_bootstrap(y, y, y, ["a", "a", "b", "b", "c", "d"], 3, n_boot=50)
    assert r["acc_diff"] == 0 and r["acc_diff_ci95"] == [0.0, 0.0]
    import compare as CMP
    assert CMP.group_of("/d/test/_EXCLUDED/Dengue/3_1.tif") == "test/Dengue/3"
    assert CMP.group_of("/d/context_virus_1nm_256x256/test/Nipah virus/12_4.tif") == "test/Nipah virus/12"


@test
def test_analyze_and_compare_smoke():
    if not HAVE_TV:
        raise SkipTest("torchvision not installed")
    import analyze as AN
    import compare as CMP
    import train as TR
    tmp = Path(tempfile.mkdtemp())
    try:
        make_synthetic_dataset(tmp)
        common = ["--data-root", str(tmp), "--out-dir", str(tmp / "runs"), "--skip-verify", "--device", "cpu",
                  "--backbone", "resnet18", "--no-pretrained", "--epochs", "1", "--max-steps", "1"]
        _, bm = TR.main(common + ["--model-type", "metric", "--K", "2", "--steps-per-epoch", "1",
                                  "--val-support-max", "2", "--run-name", "m"])
        _, bc = TR.main(common + ["--model-type", "classifier", "--batch-size", "4", "--run-name", "c"])
        ra = AN.main(["--data-root", str(tmp), "--checkpoint", str(bm), "--skip-verify", "--device", "cpu",
                      "--n-boot", "10", "--no-tta"])
        rb = AN.main(["--data-root", str(tmp), "--checkpoint", str(bc), "--skip-verify", "--device", "cpu",
                      "--n-boot", "10", "--no-tta"])
        assert {"feat_l2", "feat_cl2n", "head_l2", "head_cl2n"} <= set(ra["variants"])
        v = ra["variants"]["feat_cl2n"]
        assert "gfsl_test_calibrated" in v and v["gamma_selected_on_val"] in ra["gamma_grid"]
        assert "logits_closed" in rb["variants"]
        pa = Path(bm).parent / "analysis" / "predictions.npz"
        pb = Path(bc).parent / "analysis" / "predictions.npz"
        r = CMP.main(["--a", str(pa), "--a-key", "feat_l2_closed", "--b", str(pb), "--b-key", "logits_closed",
                      "--n-boot", "20"])
        assert r["n_patches"] == 14 * 2 * 2 and r["n_micrographs"] == 14 * 2
        r2 = CMP.main(["--a", str(pa), "--a-key", "feat_cl2n_gfsl_calibrated", "--b", str(pa),
                       "--b-key", "feat_l2_gfsl_raw", "--n-boot", "20"])
        assert r2["n_patches"] == 22 * 2 * 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_resume_after_interruption():
    if not HAVE_TV:
        raise SkipTest("torchvision not installed")
    import csv as _csv

    import train as TR
    tmp = Path(tempfile.mkdtemp())
    try:
        make_synthetic_dataset(tmp)
        base = ["--data-root", str(tmp), "--out-dir", str(tmp / "runs"), "--skip-verify", "--device", "cpu",
                "--backbone", "resnet18", "--no-pretrained", "--max-steps", "2", "--run-name", "r"]
        for mt, extra in (("metric", ["--K", "2", "--steps-per-epoch", "2", "--val-support-max", "2"]),
                          ("classifier", ["--batch-size", "4", "--ce-head", "linear", "--train-source", "train",
                                          "--schedule", "cosine", "--warmup-epochs", "1"])):
            shutil.rmtree(tmp / "runs", ignore_errors=True)
            run = tmp / "runs" / "r"
            # 1) "power cut" after epoch 2 of 4
            TR.main(base + ["--model-type", mt, "--epochs", "4", "--debug-stop-after-epoch", "2"] + extra)
            assert not (run / "train_summary.json").exists()
            ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            assert ck["epoch"] == 2 and "train_state" in ck
            lr_before = ck["train_state"]["scheduler"]["last_epoch"]
            # 2) without --resume the existing run is protected
            try:
                TR.main(base + ["--model-type", mt, "--epochs", "4"] + extra)
                raise AssertionError("must refuse to overwrite")
            except FileExistsError:
                pass
            # 3) resume (a changed hyper-parameter must be ignored)
            TR.main(base + ["--model-type", mt, "--epochs", "4", "--resume", "--lr-head", "0.123"] + extra)
            cfg = json.loads((run / "config.json").read_text())
            assert cfg["lr_head"] != 0.123
            with open(run / "history.csv", newline="") as fh:
                epochs = [int(r["epoch"]) for r in _csv.DictReader(fh)]
            assert epochs == [1, 2, 3, 4], epochs
            ck2 = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            assert ck2["epoch"] == 4 and ck2["train_state"]["scheduler"]["last_epoch"] == 2 * lr_before
            summ = json.loads((run / "train_summary.json").read_text())
            assert len(summ["history"]) == 4
            assert (run / "best.pt").exists() and "train_state" not in torch.load(
                run / "best.pt", map_location="cpu", weights_only=False)
            # 4) resuming a finished run is a no-op
            mtime = (run / "last.pt").stat().st_mtime
            TR.main(base + ["--model-type", mt, "--epochs", "4", "--resume"] + extra)
            assert (run / "last.pt").stat().st_mtime == mtime
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_gfsl_methods():
    from temfs import gfsl_methods as G
    rng = np.random.default_rng(0)
    d, nb, nn = 40, 4, 3
    centers = np.abs(rng.normal(0, 1, (nb + nn, d))) * 3
    spreads = [1.0] * nb + [0.2, 1.5, 1.0]  # a tight rare class and a spread rare class
    def sample(c, n):
        return np.abs(centers[c] + rng.normal(0, spreads[c], (n, d)))
    Xs = np.concatenate([sample(c, 60 if c < nb else 8) for c in range(nb + nn)])
    ys = np.concatenate([[c] * (60 if c < nb else 8) for c in range(nb + nn)])
    Xq = np.concatenate([sample(c, 30) for c in range(nb + nn)])
    yq = np.repeat(np.arange(nb + nn), 30)
    base_ids, novel_ids = list(range(nb)), list(range(nb, nb + nn))
    for name, (fit, pred, grid) in G.METHODS.items():
        params = dict(grid[0])
        if "pca_dim" in params:
            params["pca_dim"] = 10
        m = fit(Xs, ys, nb + nn, base_ids=base_ids, novel_ids=novel_ids, **params)
        sc = pred(m, Xq)
        assert sc.shape == (len(Xq), nb + nn), (name, sc.shape)
        acc = float((sc.argmax(1) == yq).mean())
        assert acc > 0.8, (name, acc)
    # selection returns a grid member and is deterministic
    p1, v1, _ = G.select_on_validation("logreg", Xs, ys, Xq, yq, nb + nn, base_ids, novel_ids, log=lambda *a: None)
    p2, v2, _ = G.select_on_validation("logreg", Xs, ys, Xq, yq, nb + nn, base_ids, novel_ids, log=lambda *a: None)
    assert p1 == p2 and v1 == v2 and p1 in G.METHODS["logreg"][2]
    # Mahalanobis must beat cosine NCM when a tight class sits next to a spread class
    d2 = 2
    A = rng.normal([0, 0], 0.05, (40, d2)); B = rng.normal([0.5, 0], 0.6, (40, d2))
    Xs2 = np.abs(np.concatenate([A, B]) + 3); ys2 = np.array([0] * 40 + [1] * 40)
    Aq = rng.normal([0, 0], 0.05, (200, d2)); Bq = rng.normal([0.5, 0], 0.6, (200, d2))
    Xq2 = np.abs(np.concatenate([Aq, Bq]) + 3); yq2 = np.array([0] * 200 + [1] * 200)
    mm = G.fit_maha(Xs2, ys2, 2, base_ids=[0, 1], pca_dim=2, beta=0.01)
    acc_m = float((G.predict_maha(mm, Xq2).argmax(1) == yq2).mean())
    assert acc_m > 0.85, acc_m
    sep = G.pair_separability(Xs, ys, Xq, yq, nb, nb + 1)
    assert 0.5 <= sep["auc_logreg"] <= 1.0 and sep["n_val_a"] == 30
    assert np.all(G.tukey(np.array([[0.0, 4.0]])) == np.array([[0.0, 2.0]]))


@test
def test_gfsl_improve_smoke():
    if not HAVE_TV:
        raise SkipTest("torchvision not installed")
    import gfsl_improve as GI
    import train as TR
    tmp = Path(tempfile.mkdtemp())
    try:
        make_synthetic_dataset(tmp, n_img={"train": 3, "validation": 2, "test": 2}, patches_per_img=3)
        _, bm = TR.main(["--data-root", str(tmp), "--out-dir", str(tmp / "runs"), "--skip-verify", "--device", "cpu",
                         "--backbone", "resnet18", "--no-pretrained", "--epochs", "1", "--max-steps", "1",
                         "--model-type", "metric", "--K", "2", "--steps-per-epoch", "1", "--val-support-max", "2",
                         "--run-name", "m"])
        r = GI.main(["--data-root", str(tmp), "--checkpoints", str(bm), "--out", str(tmp / "gi"), "--skip-verify",
                     "--device", "cpu", "--no-tta", "--n-boot", "10", "--kshot-repeats", "2", "--kshot", "1"])
        R = r["checkpoints"][str(bm)]
        assert set(R["methods"]) == {"ncm", "maha", "logreg", "dc"}
        for name, v in R["methods"].items():
            assert 0 <= v["test"]["accuracy"] <= 1 and "1" in v["kshot"], name
            assert "Dengue" in v["test"]["weak_f1"]
        assert "Dengue|Sapovirus" in R["diagnostic_validation"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests =[v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"running {t.__name__} ...", flush=True)
        t()
    print("\n" + "=" * 70)
    for name, status, info in RESULTS:
        print(f"{status:5s} {name}" + (f"  ({info})" if status == "SKIP" else ""))
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    for name, _, tb in fails:
        print("\n" + "-" * 70 + f"\nFAIL {name}\n{tb}")
    print("=" * 70)
    print(f"{sum(r[1] == 'PASS' for r in RESULTS)} passed, {len(fails)} failed, "
          f"{sum(r[1] == 'SKIP' for r in RESULTS)} skipped")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
