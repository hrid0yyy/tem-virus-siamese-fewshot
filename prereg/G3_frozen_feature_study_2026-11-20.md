# Pre-registration: G3 frozen-feature study (written 20 Nov 2026, before any G3 result)

This note fixes the analyses, selection rules and expected outcomes of the G3 study *before* it is run.
Anything not listed here is exploratory and will be labelled as such. Nothing is tuned after test results are seen.

## Disclosure of results already seen

- The official test split has been used in all earlier analyses of this project (paper version of 18 Oct 2026).
- Centred-and-L2-normalised features (CL2N) and calibrated stacking were already run in `analyze.py` on all checkpoints
  (logs in `remote_artifacts/`); for seed 0, calibrated stacking moved nearest-class-mean macro-F1 from 0.796 to 0.805
  (Siamese) and from 0.779 to 0.800 (baseline). These rungs are therefore **not blind**.
- The four-classifier results of the paper (nearest class mean, shrunk Mahalanobis, logistic regression, distribution
  calibration) are known.

## Step 0: reproduction gate (before any new analysis)

The round-2 command `gfsl_improve.py --kshot 1,5 --kshot-repeats 30` is re-run for the seven checkpoints on the new
machine (RTX 5060 Ti; torch 2.11.0+cu128 identical to the original runs; numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1;
the original numpy/scikit-learn versions were not recorded). Pass criterion, declared before running: for every checkpoint,
identical selected hyper-parameters, test macro-F1 within ±0.005 for every method, and nearest-class-mean per-patch test
predictions agreeing on ≥99.5 % of patches with `remote_artifacts/runs/gfsl_fp32/*/predictions_*.npz`.
A failure is reported to the author before anything else is run.

## Feature sets

- Trained encoders (7 checkpoints): Siamese SupCon seeds 0–2, benchmark-classifier (cross-entropy) seeds 0–2, SupCon with
  auxiliary cosine-classifier term. Backbone feature `feat` (1,920-d), D4 test-time augmentation, fp32 deterministic.
- **New:** frozen ImageNet DenseNet201 (`frozen_imagenet_model("densenet201")`), same feature, same TTA.
- Embeddings of all 10,604 patches (train/validation/test, base and rare) are cached once per feature set, with and
  without TTA, together with labels, micrograph ids and paths, for reuse in G6–G10.

## Q1. Frozen ImageNet features in the 22-class protocol (confirmatory)

- Protocol: unchanged primary protocol of the paper: fit on the official training split, choose hyper-parameters by
  22-class validation macro-F1 over the paper's grids, apply once to test. Selection pool = the paper's four classifiers;
  the reported classifier = highest validation macro-F1 (rule fixed in the paper).
- Comparison: 22-class test macro-F1 of the protocol-selected classifier, frozen ImageNet vs each trained encoder seed,
  paired micrograph-cluster bootstrap (1,000 resamples).
- **Expected:** frozen ImageNet features are clearly worse than both trained encoders (14-class pilot: nearest mean
  66.0 %, linear probe 88.1 %), with the gap smaller under logistic regression than under nearest class mean.
- Deviation if: frozen ImageNet macro-F1 ≥ any trained encoder under the selected classifier.

## Q2. Decision-rule ladder (diagnostic; no selection among rungs)

Each rung changes one factor. Rungs with hyper-parameters choose them by 22-class validation macro-F1 (same rule as Q1).
All rungs are reported for all eight feature sets; none is "the improvement".

| Rung | Classifier | Change vs previous |
|---|---|---|
| R1 | nearest class mean, cosine on L2 features | paper baseline |
| R2 | CL2N nearest class mean (subtract base-class training mean, L2) | centring (seen on seed 0) |
| R3 | standardised nearest class mean (z-score per feature on training split, then cosine) | per-feature scaling |
| R4 | LDA, shared covariance, Ledoit–Wolf shrinkage | shared covariance |
| R5 | shrunk class-specific Mahalanobis (paper grid) | class-specific covariance |
| R6a | logistic regression, raw features, unweighted | discriminative |
| R6b | logistic regression, standardised, unweighted | + standardisation |
| R6c | logistic regression, raw, class-balanced | + class weights |
| R6d | logistic regression, standardised, class-balanced (paper) | both |
| R7 | R1 + calibrated stacking (γ on validation) | base-class bias correction (seen on seed 0) |

- **Expected:** R2 ≥ R1; R4–R6 > R1–R3; R6a–R6c each below or close to R6d, i.e. standardisation and class weighting
  account for part, but not all, of the logistic-regression gain.
- Deviation if: R6a (no standardisation, no weights) is within 0.01 macro-F1 of R1 (then the "isotropic distance"
  explanation is not supported), or if R3/R7 reach R6d (then the gain is scaling/bias, not a learned boundary).

## Q3. Base-class-only selection (sensitivity analysis; no rare-class labels used for any choice)

- Pseudo-rare episodes on the base classes: 20 draws (seed 0). In each, 4 of the 14 base classes are pseudo-rare; for each,
  m training micrographs are kept, with m drawn from the rare classes' training-micrograph counts {5, 8, 13, 19, 20, 20, 22, 25}.
  The other 10 base classes keep all training patches and are the only "base" classes for any base statistic (Mahalanobis
  PCA, distribution-calibration neighbours, CL2N mean). Score = 14-class macro-F1 on the base-class validation patches.
- For each classifier, the setting with the highest mean score over the 20 draws is chosen; ties → simplest setting.
  The chosen setting is then fitted on the full training split (22 classes) and applied once to test.
- Limitation stated in advance: pseudo-rare classes were seen by the encoder, so these episodes are easier than real rare
  classes.
- **Expected:** the same classifier wins as under Q1 and test macro-F1 changes by ≤0.01.
- Deviation if: a different classifier wins, or macro-F1 of the selected classifier changes by >0.01.

## Q4. Per-K selection (sensitivity, under the base-only rule of Q3)

- K ∈ {1, 3, 5, 10}; in the pseudo-rare episodes each pseudo-rare class keeps K training patches from K different
  micrographs; hyper-parameters are chosen per K and per classifier. Evaluation on the real rare classes uses 30 support
  draws per K (seed 0, identical draws for every feature set and classifier).
- **Expected:** re-tuning per K raises logistic regression at K = 1 above the paper's 18.0 % but it stays below
  distribution calibration; the Siamese one-shot advantage under nearest class mean remains; under the best one-shot
  classifier the two encoders remain similar.
- Deviation if: under the best one-shot classifier either encoder is better by more than the seed spread, or the ranking
  of classifiers at K = 1 changes.

## Q5. Separability of the confusable pairs

- Primary (does not touch test): leave-one-micrograph-out on training + validation patches of each of the 8 pairs, binary
  logistic regression (standardised, class-balanced, C = 0.1 as in the paper), AUROC with micrograph-cluster bootstrap CI.
- Descriptive: train → test AUROC for the same pairs.
- **Expected:** most pairs keep AUROC ≥ 0.95; Dengue vs Sapovirus falls clearly below its validation value of 1.000.
- Deviation if: Dengue vs Sapovirus stays ≥ 0.99 under leave-one-micrograph-out (then the representation-limit sentence
  added in G2 is wrong), or most pairs fall below 0.9 (then "nearly separable" must be removed).

## Statistics

Per-checkpoint paired micrograph-cluster bootstrap and the existing paired seed comparisons. The pooled hierarchical
bootstrap over seeds × micrographs with Holm correction is part of G4 and will be applied to these saved per-patch
predictions without re-running.

## Outputs

`runs/g3/` on the experiment machine: cached embeddings, one JSON per feature set, per-patch test predictions for every
feature set × classifier × protocol, and logs. No manuscript text is changed until the author has seen the results against
the expectations above.

## Amendment (20 Nov 2026, before any G3 result)

For efficiency, the Step 0 gate is run on two checkpoints (Siamese seed 0 and benchmark-classifier seed 0) instead of
seven. For the other five, the G3 primary results (same four classifiers, same grids, same selection rule) are compared
with the reference outputs under the same tolerance (selected hyper-parameters identical, test macro-F1 within ±0.005);
any mismatch is reported to the author before the G3 results are interpreted.
