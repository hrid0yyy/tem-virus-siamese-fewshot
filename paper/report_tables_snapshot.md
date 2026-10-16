## T1. F1 - frozen ImageNet DenseNet201 (no training), 14 classes
| Read-out | Validation acc (%) | Test acc (%) | Test macro-F1 |
|---|---|---|---|
| Nearest prototype (cosine) | 71.9 | 66.0 | 0.628 |
| Logistic-regression probe | 90.4 | 88.1 | 0.832 |
| Chance |  | 7.1 |  |

## T2a. F2 - unseen classes Astrovirus, Influenza, Papilloma, Rotavirus (600 image-disjoint episodes, 15 queries/class)
| Task | Frozen ImageNet (%) | Trained backbone feat (%) | Trained 128-d head (%) | Paired diff feat-frozen (pts, 95% CI) |
|---|---|---|---|---|
| 4-way 1-shot | 67.8 | 88.0 | 76.4 | +20.1 ± 0.8 |
| 4-way 5-shot | 86.9 | 96.9 | 84.8 | +10.0 ± 0.4 |
| 4-way 10-shot | 89.3 | 97.7 | 85.2 | +8.3 ± 0.4 |
Decision: **GO** (pre-defined rule at K=5). Checkpoint selected on `val_head_macro_f1`.

## T2b. F3 - unseen classes Marburg, Orf, Adenovirus, CCHF (600 image-disjoint episodes, 15 queries/class)
| Task | Frozen ImageNet (%) | Trained backbone feat (%) | Trained 128-d head (%) | Paired diff feat-frozen (pts, 95% CI) |
|---|---|---|---|---|
| 4-way 1-shot | 77.4 | 91.2 | 86.8 | +13.8 ± 0.8 |
| 4-way 5-shot | 91.2 | 96.0 | 93.1 | +4.8 ± 0.3 |
| 4-way 10-shot | 92.7 | 96.4 | 93.7 | +3.7 ± 0.3 |
Decision: **GO** (pre-defined rule at K=5). Checkpoint selected on `val_head_macro_f1`.

## T3. F3 generalised test - 10 trained classes + 4 hidden look-alike classes, official test split
| Embedding | Trained-class acc (%) | Hidden-class acc (%) | HM | Hidden acc, 5 examples (%) | Open-set AUROC |
|---|---|---|---|---|---|
| Frozen ImageNet (feat) | 65.2 | 70.2 | 0.676 | 50.7 | 0.656 |
| Trained (feat) | 87.3 | 81.4 | 0.843 | 72.5 | 0.856 |
| Trained (head) | 95.2 | 26.9 | 0.419 | 26.8 | 0.827 |

## T4. E1 - 14 classes, official test split (1,900 patches); 8-view TTA unless stated
| Model | Accuracy (%) | Macro-F1 | Acc 95% CI (micrograph bootstrap) | Note |
|---|---|---|---|---|
| Paper, published (Matuszewski & Sintorn 2021) | 93.10 | 0.921 |  | no TTA |
| Paper model re-trained here (same evaluation) | 94.53 | 0.938 | [92.3, 96.3] | best epoch 20 |
| Siamese SupCon seed 0, backbone feat (pre-specified) | 94.95 | 0.943 | [92.2, 97.6] | best epoch 25 |
| Siamese SupCon seed 0, 128-d head | 95.89 | 0.952 | [93.5, 98.3] | best epoch 25 |
| Siamese SupCon seed 1, backbone feat (pre-specified) | 94.79 | 0.946 | [92.1, 97.4] | best epoch 13 |
| Siamese SupCon seed 1, 128-d head | 95.95 | 0.955 | [94.3, 97.5] | best epoch 13 |
| Siamese SupCon seed 2, backbone feat (pre-specified) | 94.58 | 0.938 | [92.0, 97.3] | best epoch 14 |
| Siamese SupCon seed 2, 128-d head | 95.26 | 0.948 | [93.1, 97.3] | best epoch 14 |
- Siamese feat, 3 seeds: accuracy 94.77 ± 0.18 %, macro-F1 0.942 ± 0.004 (mean ± SD)
- Siamese head, 3 seeds: accuracy 95.70 ± 0.38 %, macro-F1 0.952 ± 0.003 (mean ± SD)

## T5. Paired comparisons on identical test patches
| Comparison (A vs B) | Acc A (%) | Acc B (%) | Diff (pts) | 95% CI cluster bootstrap | Macro-F1 diff | McNemar p | Patches | Micrographs |
|---|---|---|---|---|---|---|---|---|
| cmp_gfsl_siamese_vs_paperfeat | 89.99 | 87.99 | +1.99 | [-0.60, +4.75] | +0.016 | 0.000699 | 2057 | 252 |
| cmp_siamese_feat_vs_paper | 95.00 | 94.53 | +0.47 | [-1.29, +2.09] | +0.006 | 0.391 | 1900 | 206 |
| cmp_siamese_head_vs_paper | 95.89 | 94.53 | +1.37 | [-0.38, +3.06] | +0.014 | 0.00544 | 1900 | 206 |

## T6. E2 - all 22 classes, official test split (2,057 patches)
| Model | Accuracy (%) | Macro-F1 | Acc 95% CI | 14-class acc (%) | Rare mean-class acc (%) |
|---|---|---|---|---|---|
| CE trained on all 22 classes (rare classes seen in training) | 93.10 | 0.897 | [89.5, 96.3] | 95.1 | 81.8 |
| Paper model features, rare classes as prototypes | 87.99 | 0.779 | [83.2, 91.7] | 90.1 | 75.4 |
| Siamese seed 0 (feat), rare classes as prototypes | 89.94 | 0.795 | [85.8, 93.6] | 92.1 | 76.9 |
| Siamese seed 0 (head), rare classes as prototypes | 89.50 | 0.684 | [85.8, 93.0] | 94.1 | 41.1 |
| Siamese seed 1 (feat), rare classes as prototypes | 89.35 | 0.782 | [85.2, 92.9] | 91.5 | 75.1 |
| Siamese seed 1 (head), rare classes as prototypes | 89.16 | 0.686 | [85.9, 92.0] | 94.2 | 33.3 |
| Siamese seed 2 (feat), rare classes as prototypes | 89.35 | 0.780 | [85.3, 93.0] | 91.7 | 71.5 |
| Siamese seed 2 (head), rare classes as prototypes | 89.35 | 0.708 | [85.6, 92.6] | 93.9 | 43.0 |
- Siamese feat GFSL, 3 seeds: accuracy 89.55 ± 0.34 %, macro-F1 0.786 ± 0.008
- Published 22-class results (different / unknown splits, all 22 classes trained): DMLF-Net 89.89 % acc (EAAI 2024); Hoque et al. 2026 preprint 88.31 % acc, macro-F1 0.853.

## T7. Per-class F1, 22 classes (* = rare class, never trained by the Siamese model)
| Class | Test patches | Siamese seed 0 feat (F1) | CE on all 22 (F1) |
|---|---|---|---|
| Adenovirus | 86 | 0.846 | 0.897 |
| Astrovirus | 66 | 0.817 | 0.780 |
| CCHF | 86 | 0.926 | 0.942 |
| Cowpox | 59 | 0.862 | 0.919 |
| Ebola | 363 | 0.931 | 0.931 |
| Influenza | 170 | 0.967 | 0.971 |
| Lassa | 128 | 0.895 | 0.889 |
| Marburg | 173 | 0.937 | 0.950 |
| Nipah virus | 35 | 0.914 | 0.853 |
| Norovirus | 84 | 0.988 | 0.982 |
| Orf | 31 | 0.903 | 0.968 |
| Papilloma | 187 | 0.997 | 1.000 |
| Rift Valley | 392 | 0.956 | 0.959 |
| Rotavirus | 40 | 1.000 | 0.988 |
| Dengue * | 44 | 0.108 | 0.264 |
| Guanarito * | 7 | 0.467 | 1.000 |
| LCM * | 14 | 0.526 | 0.692 |
| Machupo * | 23 | 0.698 | 0.810 |
| Pseudocowpox * | 16 | 0.882 | 1.000 |
| Sapovirus * | 8 | 0.203 | 1.000 |
| TBE * | 11 | 0.714 | 0.957 |
| WestNile * | 34 | 0.957 | 0.986 |

## T8. E3 - rare classes with K examples each (100 repeats); 14 classes keep all training patches
| K | Siamese feat: rare acc (%) | HM | Paper-model feat: rare acc (%) | HM | Siamese head: rare acc (%) | HM |
|---|---|---|---|---|---|---|
| 1 | 35.3 ± 1.8 | 0.506 | 27.3 ± 1.6 | 0.414 | 17.3 ± 1.4 | 0.286 |
| 3 | 54.9 ± 0.7 | 0.689 | 54.6 ± 0.7 | 0.682 | 22.0 ± 1.4 | 0.352 |
| 5 | 59.0 ± 0.6 | 0.720 | 60.1 ± 0.5 | 0.723 | 29.2 ± 0.7 | 0.444 |
| 10 | 63.3 ± 0.4 | 0.751 | 63.1 ± 0.3 | 0.744 | 33.2 ± 0.5 | 0.491 |

## T9. E4/E5 - 8-way image-disjoint episodes on the rare classes, and unknown-virus detection
| Embedding | Open-set AUROC (patch) | FPR@95%TPR (%) | AUROC (micrograph) | 8-way 1-shot acc (%) | 8-way 5-shot acc (%) |
|---|---|---|---|---|---|
| Siamese feat | 0.883 | 56.7 | 0.840 | 66.4 | 79.3 |
| Siamese head | 0.907 | 52.9 | 0.818 | 55.6 | 63.3 |
| Paper-model feat | 0.877 | 63.7 | 0.832 | 66.6 | 81.4 |

## T10. Inference-time variants (CL2N centring; calibrated stacking with gamma chosen on VALIDATION)
| Run | Variant | 14-class acc (%) | 22-class acc raw (%) | 22 macro-F1 raw | gamma (val) | 22-class acc calibrated (%) | 22 macro-F1 calibrated |
|---|---|---|---|---|---|---|---|
| E1_siamese_supcon_densenet201 | feat_l2 | 95.00 | 89.99 | 0.795 | -0.010 | 90.33 | 0.805 |
| E1_siamese_supcon_densenet201 | feat_cl2n | 95.37 | 90.67 | 0.797 | -0.015 | 91.20 | 0.807 |
| E1_siamese_supcon_densenet201_seed1 | feat_l2 | 94.79 | 89.35 | 0.782 | -0.025 | 90.13 | 0.793 |
| E1_siamese_supcon_densenet201_seed1 | feat_cl2n | 94.95 | 90.23 | 0.795 | -0.035 | 90.52 | 0.793 |
| E1_siamese_supcon_densenet201_seed2 | feat_l2 | 94.58 | 89.35 | 0.780 | -0.015 | 89.65 | 0.785 |
| E1_siamese_supcon_densenet201_seed2 | feat_cl2n | 94.84 | 90.03 | 0.785 | -0.055 | 90.37 | 0.776 |
| E1_paper_densenet201 | feat_l2 | 92.58 | 87.99 | 0.779 | -0.025 | 89.11 | 0.800 |
| E1_paper_densenet201 | feat_cl2n | 91.74 | 87.89 | 0.769 | -0.075 | 88.67 | 0.796 |
| E6_supcon_auxce05_densenet201 | feat_l2 | 94.37 | 89.35 | 0.795 | -0.025 | 90.18 | 0.813 |
| E6_supcon_auxce05_densenet201 | feat_cl2n | 94.79 | 90.18 | 0.797 | -0.020 | 90.47 | 0.801 |
| E6_triplet_densenet201 | feat_l2 | 93.58 | 88.67 | 0.779 | -0.025 | 89.89 | 0.801 |
| E6_triplet_densenet201 | feat_cl2n | 93.79 | 89.35 | 0.784 | -0.040 | 90.08 | 0.800 |

## T11. E6 - loss ablations (backbone feat, TTA)
| Training loss | 14-class acc (%) | 14 macro-F1 | 22-class acc (%) | 22 macro-F1 | Rare mean-class acc (%) |
|---|---|---|---|---|---|
| SupCon (main, seed 0) | 94.95 | 0.943 | 89.94 | 0.795 | 76.9 |
| SupCon + 0.5 cosine-CE (Baseline++ term) | 94.37 | 0.938 | 89.35 | 0.794 | 79.4 |
| Batch-hard triplet | 93.58 | 0.929 | 88.67 | 0.779 | 76.4 |

## T12. Training runs (RTX 3060)
| Run | Epochs run | Best epoch | Selection metric (validation) | Train time (min) |
|---|---|---|---|---|
| E1_siamese_supcon_densenet201 | 37 | 25 | val_feat_macro_f1 | 28 |
| E1_siamese_supcon_densenet201_seed1 | 25 | 13 | val_feat_macro_f1 | 19 |
| E1_siamese_supcon_densenet201_seed2 | 26 | 14 | val_feat_macro_f1 | 20 |
| E1_paper_densenet201 | 50 | 20 | val_acc | 134 |
| E2_ce22_densenet201 | 32 | 20 | val_macro_f1 | 34 |
| E6_supcon_auxce05_densenet201 | 33 | 21 | val_feat_macro_f1 | 25 |
| E6_triplet_densenet201 | 31 | 19 | val_feat_macro_f1 | 24 |
