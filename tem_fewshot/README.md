# Siamese / Few-Shot Virus Classification on the TEM Virus Dataset

Code for the feasibility tests (F0–F3) and experiments (E1–E6) described in
`../TEM_siamese_fewshot_feasibility.md`. The dataset is Matuszewski & Sintorn (2021), Mendeley Data
[10.17632/x4dwwfwtw3.3](https://data.mendeley.com/datasets/x4dwwfwtw3/3).

## What was verified against the real dataset / paper (before any training)

| Check | How |
|---|---|
| Folder layout, class folder names (incl. `Nipah virus`, `Rift Valley`, `_EXCLUDED/`) | Mendeley public API file listing of v3 |
| Per-class patch and micrograph counts for train/validation/test (10,604 patches, 1,245 micrographs) | API listing; micrograph counts equal Table 1 of the paper; shipped in `temfs/expected_counts.json` |
| Patch file format (8-bit grayscale TIFF, 256×256, packbits) | sample file |
| Patch naming `<micrograph>_<vertex>.tif` (micrograph = 1-based index inside split/class) | authors' `prepareContextVirusImages.m` |
| Pre-processing (min–max stretch, Lanczos-3 to 1 nm/px, symmetric padding, 256×256 crops, per-patch z-score) | authors' MATLAB + Keras code |
| Baseline architecture = DenseNet201 + Flatten → 1024 → Dropout 0.3 → 1024 → Dropout 0.3 → 14 | authors' `train.py`; our replica has **exactly 144,987,022** params (VGG16: **49,334,094**) = Table 2 of the paper (unit-tested) |
| Metric definitions (per-patch accuracy, macro P/R/F1) | our metric code reproduces the authors' published test report from their confusion matrix (acc 0.93053, macro-F1 0.92124) — unit-tested |

## Folder map

```
tem_fewshot/
  download_data.py      download (resumable) + extract the dataset
  f0_verify_data.py     F0  data audit (counts, readability, cross-split duplicates)
  f1_frozen_probe.py    F1  frozen ImageNet features: nearest-prototype + linear probe
  f2_pseudo_novel.py    F2/F3 hide 4 base classes, train Siamese on 10, test on the hidden 4 (GO/NO-GO)
  train.py              train Siamese/metric encoder or CE classifier baselines
  evaluate.py           E1 closed-set, E2 GFSL, E3 K-shot curve, E4 episodic, E5 open-set
  temfs/                library (data, samplers, models, losses, metrics, few-shot protocols, engine)
  tests/run_tests.py    self-tests (no dataset / GPU needed)
```

## 1. Setup (remote PC: i7-13700K, 16 GB RAM, RTX 3060 12 GB, driver 610.88 / CUDA 13.1)

Use Python 3.10–3.12 (64-bit). In a terminal inside `tem_fewshot/`:

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
Install PyTorch with CUDA (pick the newest CUDA build listed on https://pytorch.org/get-started/locally/ —
the 610.x driver runs any cu12x/cu13x build up to 13.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
```bash
pip install -r requirements.txt
```
Check the GPU is visible:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Run the self-tests (≈5–10 min on CPU; all must pass):
```bash
python tests/run_tests.py
```

The first run of F1/F2/evaluate downloads the ImageNet weights (DenseNet201 ≈ 80 MB) via torchvision.

## 2. Data

```bash
python download_data.py --out data
```
Downloads the 5.0 GB archive (resumes if interrupted; re-run the same command) and extracts only the
official patch set `context_virus_1nm_256x256` (~4.7 GB). Add `--include-raw` for the RAW micrographs
(not needed by this code). If the download is blocked, download "Download All" in a browser and run
`python download_data.py --zip <file.zip> --out data`.

## 3. Feasibility tests (run in this order)

| Step | Command | Time (3060, rough) | Pass criterion |
|---|---|---|---|
| F0 | `python f0_verify_data.py --data-root data` | 1–3 min | prints `F0 RESULT: PASS` |
| F1 | `python f1_frozen_probe.py --data-root data` | 2–5 min | NCM test accuracy far above chance (7.1 %) |
| F2 | `python f2_pseudo_novel.py --data-root data` | ~30–60 min | prints `F2 DECISION ... GO` |
| F3 | `python f2_pseudo_novel.py --data-root data --hidden preset:f3` | ~30–60 min | shows how much harder look-alike classes are |

F2 hides Astrovirus, Influenza, Papilloma, Rotavirus; F3 hides Marburg, Orf, Adenovirus, CCHF
(each has a look-alike still in training). Hidden classes never enter training or model selection;
episodes are image-disjoint (support and query from different micrographs), and the trained and the
frozen ImageNet encoder are scored on **identical episodes** (paired comparison).
If F2 says NO-GO, prefer the CE + cosine-classifier route (`--model-type classifier --ce-head cosine`)
over episodic/Siamese training.

## 4. Experiments

**E1 – paper baseline replication (14 classes, official split)**
```bash
python train.py --data-root data --model-type classifier
```
(DenseNet201 + paper head, Adam 1e-5, batch 12, 50 epochs, `augmented_train`, best val accuracy.)

**Main Siamese model (SupCon, DenseNet201, 14 base classes)**
```bash
python train.py --data-root data --model-type metric --loss supcon
```

**Evaluate any checkpoint (E1 closed-set, E2 GFSL 14+8, E3 K-shot, E4 episodic, E5 open-set)**
```bash
python evaluate.py --data-root data --checkpoint runs/<run_dir>/best.pt
```
```bash
python evaluate.py --data-root data --frozen-backbone densenet201
```

**E2 baseline – 22-class cross-entropy model (same split)**
```bash
python train.py --data-root data --model-type classifier --train-classes all22 --train-source train --sampler balanced --augment --max-shift 8 --ce-head linear --optimizer adamw --lr-backbone 5e-5 --lr-head 5e-4 --schedule cosine --epochs 40 --batch-size 32 --memory-efficient true --select-metric val_macro_f1 --patience 12
```

**E6 – ablations**
```bash
python train.py --data-root data --model-type metric --loss triplet
```
```bash
python train.py --data-root data --model-type metric --loss proto
```
```bash
python train.py --data-root data --model-type metric --loss pair
```
```bash
python train.py --data-root data --model-type metric --loss supcon --aux-ce-weight 0.5
```
```bash
python train.py --data-root data --model-type metric --loss supcon --backbone resnet50
```

## 5. Defaults and design choices

* **Input**: official 256×256 patches at 1 nm/px, per-patch z-score, grayscale repeated to 3 channels.
  Scale is never changed (particle size is a key cue).
* **Training augmentation** (metric models): the 8 exact rotations/flips (D4) + up to ±8 px shift.
* **Siamese encoder**: ImageNet backbone → GAP → Linear(→512) → BN → ReLU → Linear(→128) → L2-norm.
  Both the 128-d embedding (`head`) and the GAP backbone features (`feat`) are evaluated.
* **Batches**: P classes × K samples (default 14 × 4 = 56), drawing micrographs first so that
  positives come from different micrographs and big-but-few-image classes (Papilloma) don't dominate.
* **Model selection**: nearest-prototype macro-F1 on the official validation split
  (prototypes from ≤300 train patches per class); early stopping patience 12.
* **Optimiser**: AdamW, backbone lr 5e-5, head lr 5e-4, cosine schedule, 2 warm-up epochs, AMP fp16,
  DenseNet gradient checkpointing (`memory_efficient`).
* **Evaluation**: 8-view TTA (average embedding over D4), per-patch metrics (as in the paper) and
  per-micrograph metrics, 95 % CIs by stratified micrograph bootstrap (1000×), K-shot/episodic results
  as mean ± 95 % CI over repeats.

These hyper-parameters are reasonable starting values, not tuned results; tune them on the
validation split only.

## 6. If something goes wrong

* **CUDA out of memory** (metric training): `--K 3` (batch 42) or `--P 10`; or `--backbone resnet50`.
  Contrastive losses need the whole batch at once, so gradient accumulation is not used.
* **Slow data loading**: patches are pre-loaded into RAM (~0.7 GB; +0.7 GB for `augmented_train`) and
  `--num-workers 0` is the default (best on Windows). `--no-preload` reads from disk instead.
* **Count mismatch in F0**: the extraction is incomplete — re-run `download_data.py` (it skips files
  that are already complete). `--skip-verify` exists only for debugging.
* Every run writes `config.json`, `log.txt`, `history.csv`, `best.pt`, `last.pt` in `runs/<run>/`;
  evaluation writes `results.json`, `summary.txt` and confusion-matrix CSVs.
