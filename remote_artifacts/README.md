# Artifacts copied from the GPU PC (E:\tem_fewshot), 17 Oct 2026

This folder holds everything the experiments produced **except model weights** (`*.pt`). The final `best.pt` files are attached to the GitHub release `checkpoints-v1`.

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
