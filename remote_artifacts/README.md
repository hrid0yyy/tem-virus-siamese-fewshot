# Artifacts copied from the GPU PC (E:\tem_fewshot), 17 Oct 2026

This folder holds everything the experiments produced **except model weights** (`*.pt`). The final `best.pt` files are attached to the GitHub release `checkpoints-v1`.

**Round 2 (18 Oct 2026; final numbers of the manuscript).** All of these were produced with `TEMFS_FP32=1` (fp32, deterministic cuDNN; every JSON records `inference_precision: fp32-deterministic`):
- `runs/<run>/eval_fp32/` and `runs/<run>/analysis_fp32/`: full-precision re-evaluation of all 9 checkpoints.
- `runs/E1_paper_densenet201_seed1/`, `_seed2/`: two more seeds of the baseline B1.
- `runs/gfsl_fp32/<run>/`: the rare-class classifier study per checkpoint (30 K-shot draws), with per-patch test predictions in `predictions_<run>.npz`.
- `runs/paired_gfsl_phase1.json`, `runs/paired_gfsl_final.json`: paired tests of the classifier study (`paired_gfsl.py`).
- `runs/cmp_fp32_*.json`: seed-0 paired comparisons.
- `runs/paired_seeds_fp32_local.json`: all 9 Siamese x B1 seed pairs (14-class and 22-class NCM), computed locally from the `analysis_fp32` predictions. Spot-checked: it is identical to `cmp_fp32_*.json` for seed 0 and to an independent `compare.py` recomputation.
- `run_round2_status.txt`: queue status. Every step exited 0. The line `gfsl_E1_siamese_supcon_densenet201=0 21:24:52` comes from an accidental duplicate queue that was stopped; it is annotated there.
- `tests_remote5.log`: 23/23 self-tests passed before round 2.

**Round 1 (16–17 Oct, mixed-precision inference).** `eval_full/`, `analysis/`, `runs/gfsl_improve/` and `runs/cmp_*.json` are kept for transparency. They differ from round 2 by at most one test patch per checkpoint.

- `runs/<run>/`:
  - `config.json` (the exact configuration and environment);
  - `log.txt`, `history.csv` and `train_summary.json` (training);
  - `eval_full/` (`evaluate.py`: results.json, summary.txt, confusion matrices);
  - `analysis/` (`analyze.py`: inference-time variants, plus `predictions.npz` with per-patch predictions used for the paired tests).
- `runs/cmp_*.json`: paired comparisons (`compare.py`: cluster bootstrap and McNemar).
- `runs/gfsl_improve/`: the rare-class classifier study (`gfsl_improve.py`), with `gfsl_improve.json` and `log.txt`.
- **Archived runs.** The following are kept for transparency and are not used in any result:
  - `runs/E1_paper_densenet201_interrupted*`: runs killed by PC power loss before resumable training was added.
  - `runs/E2_ce22_densenet201_duplicate_discarded`: an accidental duplicate launch, stopped and discarded.
- `reports/`: F0 data verification, F1 frozen-backbone probe, F3 frozen-backbone evaluation.
- `20261014-*_f2_*`, `20261014-*_f3_*`: the F2/F3 pilot runs.
- `report_data.json`: everything collected by `collect_results.py` (input to `paper/extract_numbers.py`).
- `leak_audit.py` / `leak_audit.log`: pixel-level train/val/test leakage audit, which found 0 duplicates.
- `logs_*.log`, `run_*_status.txt`: console logs and status files of the batch queues.
- `tests_remote*.log`: self-test runs on the GPU PC.
- `*.py` / `*.bat` at the top level: snapshot of the code as run. The code is identical to `../tem_fewshot/`, checked by hash. Helper scripts (`weak.py`, `pc.py`, `g22.py`, `ce22pc.py`, `gpucheck.py`) printed ad-hoc summaries.
