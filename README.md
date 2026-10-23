# Siamese few-shot recognition of rare viruses in TEM images

This is a research project on the public **TEM virus dataset** (Matuszewski & Sintorn 2021, Mendeley Data [10.17632/x4dwwfwtw3.3](https://doi.org/10.17632/x4dwwfwtw3.3)). It studies a Siamese (shared-weight) metric-learning encoder trained with supervised contrastive loss on the 14 benchmark classes. The 8 rare classes are added at test time as **generalised few-shot** classes, and are never used to train the encoder.

**Status (18 Oct 2026):** all experiments are finished, including round 2:
- 3 seeds of the baseline;
- full-precision deterministic re-evaluation of every checkpoint;
- paired tests for the rare-class study.

The manuscript for *Computer Methods and Programs in Biomedicine* is [`paper/main.tex`](paper/main.tex), built as [`paper/main.pdf`](paper/main.pdf) with `tectonic main.tex`. Red text in the PDF marks author-only items: names, affiliation, funding, CRediT and repository URL.

## Repository layout

| Path | Contents |
|---|---|
| `paper/main.tex` | Manuscript (Elsevier `elsarticle`, numbered references). Contains no hand-typed numbers. |
| `paper/make_numbers_tex.py` → `paper/numbers.tex` | Generates every number and table body of `main.tex` from the raw result files (`--mode fp32` = final round-2 results). |
| `paper/figures/architecture.tex` → `architecture.pdf` | Method overview (TikZ), with example patches from the dataset (CC BY 4.0, provenance in `figures/patches/SOURCES.txt`). |
| `paper/make_figures.py` → `figures/*.pdf` | Data figures generated from the result files: dataset examples, rare-class confusion, K-shot curve, training curves. `figure_values.json` holds the plotted values; `dataset_patches/SOURCES.json` records where each example patch came from. |
| `paper/highlights.txt`, `paper/references.bib` | CMPB highlights (5 bullets, each at most 85 characters) and the bibliography. |
| `paper/extract_numbers_final.py` → `paper_numbers_final.md` | Independent Markdown cross-check of the final numbers. |
| `paper/manuscript_draft.md`, `paper/extract_numbers.py` | First Markdown draft (round-1 numbers); superseded by `main.tex`. |
| `tem_fewshot/` | All code: data indexing and verification, training, evaluation protocols, statistics, the rare-class classifier study (`gfsl_improve.py`) and its paired tests (`paired_gfsl.py`). See [`tem_fewshot/README.md`](tem_fewshot/README.md). |
| `tem_fewshot/tests/run_tests.py` | Self-tests (23 tests). All pass on the GPU PC (`remote_artifacts/tests_remote5.log`). |
| `remote_artifacts/` | Everything produced on the GPU PC except model weights; see its README. |
| `TEM_virus_dataset_published_papers.md`, `TEM_virus_dataset_overview.md`, `TEM_siamese_fewshot_feasibility.md` | Literature review, dataset overview and feasibility study. |
| `TEM_fewshot_report.html`, `TEM_fewshot_report_tables.md` | Interim report of 16 Oct 2026; superseded. |

Model checkpoints (`best.pt`, 9 runs) are **not** in git. They are attached to the GitHub release `checkpoints-v1`, with SHA-256 checksums in `SHA256SUMS.txt`.

## Main results

All results are on the official test split: full-precision deterministic inference, 3 seeds per encoder, mean ± SD. They come from `paper/numbers.tex`.

| Setting | Siamese SupCon | Baseline B1 (paper model re-trained) |
|---|---|---|
| 14 classes, accuracy / macro-F1 | 94.79 ± 0.21 % / 0.943 ± 0.004 | 93.91 ± 0.54 % / 0.930 ± 0.007 (published: 93.1 % / 0.921) |
| 22 classes, nearest class mean, macro-F1 | 0.786 ± 0.008 | 0.785 ± 0.008 |
| 22 classes, logistic regression on frozen features, macro-F1 | 0.849 ± 0.009 | 0.855 ± 0.013 |
| 1-shot rare-class mean accuracy (nearest class mean) | **37.3 ± 2.3 %** | 30.8 ± 2.4 % |
| 5-shot rare-class mean accuracy (nearest class mean) | 64.6 ± 1.2 % | 67.9 ± 1.8 % |

Paired tests (micrograph-level cluster bootstrap):
- **14 classes:** Siamese is ahead in 9 of 9 seed pairs, but significant in 0 of 9 on accuracy.
- **Logistic regression vs nearest class mean:** significant on 7 of 7 checkpoints.
- **Siamese vs B1, both with logistic regression:** significant in 1 of 9 pairs, in favour of B1.
- **1-shot Siamese vs B1:** significant in 9 of 9 pairs.
- **Rare-class diagnosis:** the confused rare-class pairs are almost perfectly linearly separable on validation (e.g. Dengue vs Sapovirus AUROC = 1.000). Their errors come from the nearest-prototype rule, not the representation.

## Reproducing

1. **Set up and test.** Python 3.12, PyTorch 2.11 + CUDA 12.8 (`tem_fewshot/requirements.txt`). Run `python tests/run_tests.py`.
2. **Get the data.** `python download_data.py --out data`, then `python f0_verify_data.py --data-root data`.
3. **Run the experiments.**
   - Round 1: `tem_fewshot/run_queue.bat` and `run_gfsl.bat`.
   - Round 2 (final numbers): `tem_fewshot/run_round2.bat`, with `TEMFS_FP32=1` for evaluation.

   The exact command lines are in those files and in each run's `config.json`.
4. **Rebuild the paper.** Run `python paper/make_numbers_tex.py --mode fp32`, then `tectonic paper/main.tex`.

All runs used one NVIDIA RTX 3060 (12 GB).

## License

- Code in this repository is released under the MIT License; see [`LICENSE`](LICENSE).
- `elsarticle/` is Elsevier's LaTeX class, distributed under the LaTeX Project Public Licence, version 1.3 or later (LPPL), not MIT.
- Example TEM image patches in `paper/figures/` are derived from the TEM virus dataset (Matuszewski & Sintorn 2021, Mendeley Data [10.17632/x4dwwfwtw3.3](https://doi.org/10.17632/x4dwwfwtw3.3), CC BY 4.0).
