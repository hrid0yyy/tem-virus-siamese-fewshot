# Siamese / Few-Shot Classification on the TEM Virus Dataset: Feasibility Study and Proposed Architecture

**Status:** research and reasoning only. **No code has been run.**
**Date:** 14 October 2026
**Dataset:** Matuszewski & Sintorn, TEM virus dataset, Mendeley `x4dwwfwtw3` v3 (22 classes, 1245 images)

Labels used in this document:
- **[Paper]**: a fact taken from a published source (listed at the end).
- **[Data]**: a number taken from Table 1 of Matuszewski & Sintorn (2021).
- **[Reasoning]**: my own analysis or proposal. Treat it as a hypothesis to test, not a result.

---

## 1. Has anyone used Siamese or few-shot methods on this dataset?

**No, as far as I could find.**

- **Journal papers on the dataset.** None of the seven journal papers found earlier uses Siamese, triplet, prototypical, or any other metric-learning or few-shot approach. Their methods are:
  - fine-tuned CNNs (Matuszewski & Sintorn 2021);
  - CNN + transformer ensembles (Nanni et al. 2023);
  - a rotation-equivariant CNN (Bengtsson Bernander et al. 2024);
  - a dual-branch CNN (Sikder et al. 2024);
  - DMLF-Net (Usman et al. 2024);
  - Virus-FusionNet (Bisht et al. 2025).

  The 2026 arXiv preprint (Hoque et al.) uses CNN and transformer classifiers with weighted cross-entropy.
- **Web searches.** I searched for "Siamese + TEM virus", "few-shot / prototypical / metric learning + TEM virus", and "Matuszewski dataset + few-shot". These found no such work.
- **Closest few-shot work in TEM virology.** Xiao et al. (2023) used few-shot learning for **SARS-CoV-2 segmentation**, on a **different** dataset. It is not classification and not this dataset.
- **Caveat.** I can't rule out an unindexed thesis, workshop paper, or very recent preprint. The claim to make in a paper is: *"to the best of our knowledge, the first metric-learning / few-shot study on this dataset."*

---

## 2. Is this dataset really "few-shot"? It depends on the classes

This is the most important point for feasibility.

### 2.1 The 14 benchmark classes are not really few-shot [Data]

| Class | Train images | Train particles |
|---|---|---|
| Nipah | 25 | 75 |
| Marburg | 57 | 78 |
| Orf | 38 | 92 |
| Adenovirus | 40 | 160 |
| … | … | … |
| Rift Valley | 77 | 1140 |

- The smallest benchmark classes still have about 75–90 training particles.
- After the official augmentation (flips and 90° rotations), every class reaches **736 training patches** [Paper: Matuszewski & Sintorn 2021].
- So, on the 14-class benchmark, the problem is **imbalance and limited diversity**, not a true few-shot problem.

### 2.2 The 8 excluded classes are genuinely few-shot [Data]

| Class | Train images | Train particles | Test images | Test particles |
|---|---|---|---|---|
| WestNile | **5** | 88 | **2** | 34 |
| Sapovirus | **8** | **19** | 3 | 8 |
| Guanarito | 13 | **31** | 4 | **7** |
| LCM | 22 | **31** | 8 | 14 |
| TBE | 25 | 39 | 8 | 11 |
| Pseudocowpox | 20 | 46 | 7 | 16 |
| Machupo | 20 | 69 | 7 | 23 |
| Dengue | 19 | 72 | 7 | 44 |

- The dataset authors excluded these 8 classes because they have too few samples for normal training [Paper].
- Of the later papers, only DMLF-Net (89.89% accuracy, protocol not accessible) and a 2026 preprint (88.31% accuracy) attempted all 22 classes [Paper].

**[Reasoning]** A Siamese / few-shot approach is **most justified and most novel** if it brings these 8 classes back in. The aim would be to **recognise them from a handful of examples without retraining the network**. Using it only on the 14-class benchmark is a weaker story, because those classes have enough data for ordinary fine-tuning.

---

## 3. What the few-shot literature says (needed for a realistic expectation)

1. **Siamese networks** (Koch et al., 2015) learn a shared-weight encoder plus a similarity function on image pairs, then classify new classes by comparing them to one or a few reference images. They are one member of the **metric-based few-shot family**.
2. **Prototypical networks** (Snell et al., 2017) extend this to N classes. Each class is represented by the **mean embedding (prototype)** of its support examples, and a query takes the label of the nearest prototype. They are generally simpler and stronger than pairwise Siamese verification for N-way classification.
3. **A good embedding often matters more than the episodic trick.** Chen et al. (2019, "A Closer Look at Few-Shot Classification") and Tian et al. (2020, "Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?") found that a well-trained backbone with a simple classifier is competitive with, or better than, complex meta-learning.
4. **Domain shift is critical.** In the cross-domain few-shot benchmark of Guo et al. (2020, ECCV), all meta-learning methods underperformed simple fine-tuning by **12.8% average accuracy**, and accuracy correlated with how similar the target domain was to natural images. Negative-stain TEM is very far from ImageNet.
5. **Batch-based metric losses.** Supervised contrastive learning (Khosla et al., 2020) and batch-hard triplet loss (Hermans et al., 2017) train a shared-weight (Siamese-style) encoder using **all positive and negative pairs inside a class-balanced batch**. This is more data-efficient than feeding one pair at a time.

**[Reasoning]** What follows from this:
- A "pure" classic Siamese network (one pair at a time, sigmoid "same/different" head) is **not** the best choice. It should be kept only as an ablation.
- The best-supported design is:
  - a **pretrained, established backbone fine-tuned on TEM data** (not frozen, because of the domain shift);
  - trained with a **batch-based Siamese metric loss**;
  - classifying by **nearest prototype** at inference.

---

## 4. Feasibility assessment

| Question | Verdict | Why |
|---|---|---|
| Technically feasible? | **Yes** | About 7.7k annotated particles, plus more patches for elongated viruses [Data]. Standard backbones and losses. Training is no heavier than the DenseNet201 fine-tuning already done on this dataset [Paper]. |
| Will it beat 93.1% on the 14-class official benchmark? | **Uncertain; don't promise it** | Classes have enough data for ordinary fine-tuning, and the literature (§3) suggests metric learning gives little or no advantage there [Reasoning]. Treat this as a "no worse than the baseline" check. |
| Useful for imbalance? | **Yes, moderately** | Nearest-prototype classification uses **one mean per class**, so a large class does not get more "votes". Class-balanced batch sampling removes the imbalance during training [Reasoning]. |
| Useful for the 8 rare classes (22-class setting)? | **Yes. This is the main contribution** | New classes can be added from a few support images with no retraining. No published work does this on this dataset (§1). |
| Useful for novelty (unknown-virus) detection? | **Yes, as a bonus** | The dataset paper lists novelty detection as an intended task [Paper]. I found no published results for it. A distance threshold to the nearest prototype gives an "unknown" decision for free [Reasoning]. |
| Statistically reliable on rare classes? | **Weak if only the official test set is used** | WestNile has 2 test images, Sapovirus 3, Guanarito 4 (7 particles) [Data]. Patches from the same image are correlated, so the effective sample size is closer to the number of images [Reasoning]. Needs the extra evaluation protocol in §6. |

### 4.1 Main risks [Reasoning]

1. **Novel classes are close relatives of base classes.** This is what makes the task hard, and also scientifically interesting.
   - The sizes in Table 1 [Data] match the virus families (general virology knowledge):
     - **Pseudocowpox (145)** is a parapoxvirus, like **Orf (145)**.
     - **Guanarito, Machupo, LCM (120–140)** are arenaviruses, like **Lassa (140)**.
     - **Sapovirus (30)** is a calicivirus, like **Norovirus (30)**.
     - **Dengue, West Nile, TBE (45–50)** are flaviviruses. They have **no relative** among the 14 base classes.
   - Expect novel-to-base confusion within these families. Report a confusion matrix grouped by family.
2. **Leakage.** All pairs, triplets, and episodes must respect the image-level split. Never put patches from the same image in both support and query. This follows the warnings in the dataset paper and in Bengtsson Bernander et al. (2024) [Paper].
3. **Scale must be kept.** Particle size is a strong cue, and the official 1 nm/pixel rescaling preserves it [Paper]. Do not resize arbitrary raw images to 224×224 as the 2026 preprint did [Paper]. Use the official 256×256 patches.
4. **Small-sample metrics.** On rare classes, report confidence intervals, not single numbers.

**Overall verdict: FEASIBLE and worth doing, if it is framed as *generalised few-shot learning*:**
- 14 base classes plus 8 novel classes;
- class-imbalance robustness;
- novelty detection as a bonus.

It is **not recommended** as a way to raise the 14-class accuracy alone.

---

## 5. Proposed architecture: Siamese encoder with prototype classifier

**[Reasoning: proposal built only from established components]**

```
                   ┌──────────── shared weights (Siamese) ────────────┐
patch A (256×256) →│ ImageNet-pretrained backbone → GAP → projection  │→ z_A (128-d, L2-norm)
patch B (256×256) →│ (same network)                                    │→ z_B
                   └───────────────────────────────────────────────────┘
Training:  class-balanced P×K batches → supervised contrastive loss (in-batch pairs)
           (+ optional cosine-classifier cross-entropy on base classes)
Inference: prototype_c = mean(z of class-c support patches, over 8 rotations/flips)
           predict = argmax_c cos(z_query, prototype_c);  "unknown" if max cos < τ
```

### 5.1 Input [Paper: official preprocessing]
- Lanczos-3 rescaling to 1 nm/px.
- 256×256 patches centred on the annotations.
- Per-patch z-score normalisation.
- Grayscale replicated to 3 channels for ImageNet backbones.

### 5.2 Backbone (shared encoder)

| Option | Why | Role |
|---|---|---|
| **DenseNet201** (ImageNet) | Best model on this exact dataset (93.1% accuracy, F1 0.921 when fine-tuned) [Paper]. 1920-d features | **Primary** |
| ResNet50 (ImageNet) | The most widely used backbone in metric-learning work. 2048-d features. It underperformed DenseNet201 here (89.8%) [Paper] | Ablation |
| VGG16 (ImageNet) | Second best in the benchmark (92.3%) [Paper] | Optional ablation |

- All layers are fine-tuned (low learning rate for the backbone, higher for the head), because of the domain shift (§3, point 4).
- Note: the 145 M parameters reported for DenseNet201 in the dataset paper include their large fully connected head [Paper]. The convolutional backbone itself is much smaller (roughly 18 M, a standard figure, not from the paper).

### 5.3 Projection / embedding head
Linear(1920→512) → BatchNorm → ReLU → Linear(512→128) → L2-normalise. This is the standard SimCLR / SupCon style head.

### 5.4 Training objective (main choice and ablations)

| Variant | Loss | Purpose |
|---|---|---|
| **Main** | Supervised contrastive (Khosla 2020) on class-balanced P×K batches (e.g. 14 classes × 4 patches) | Uses every in-batch positive and negative pair. Balanced sampling handles imbalance |
| Ablation 1 | Batch-hard triplet (Hermans 2017) | Classic Siamese / triplet alternative |
| Ablation 2 | Prototypical episodic loss (Snell 2017), N-way K-shot episodes | The classic few-shot formulation |
| Ablation 3 | Classic pairwise Siamese (Koch 2015): \|z_A − z_B\| → FC → sigmoid, binary cross-entropy | Shows why the pairwise version is weaker or stronger |
| Optional add-on | + cosine-classifier cross-entropy on base classes (Chen 2019 "Baseline++") | Often stabilises the embedding |

- **Augmentation:** the 8 dihedral transforms (flips + 90° rotations), as used by the dataset authors and Bengtsson Bernander et al. (2024) [Paper].
- Mild contrast and blur changes may be added, but random scaling should **not** be used, because it destroys the size cue [Reasoning].

### 5.5 Inference
- **Prototype per class:** the mean of the embeddings of that class's support patches. Each patch is averaged over its 8 orientations, which makes the prototype rotation-invariant (motivated by the rotation-equivariance results of Bengtsson Bernander et al. 2024).
- **Novel classes:** compute their prototypes from their **training-split images only**. The network is not retrained.
- **Prediction:** the nearest prototype by cosine similarity.
- **Open-set:** if the highest similarity is below a threshold τ (tuned on validation), answer "unknown".
- **Optional per-image decision:** average the particle predictions within one image. Report this separately from per-particle accuracy.

---

## 6. Evaluation design (so the results are publishable and comparable)

| Experiment | Setting | Compare with |
|---|---|---|
| **E1** | 14 classes, **official split**, per-particle accuracy and macro-F1 (same metric as the dataset paper) | Matuszewski & Sintorn: 93.1% / 0.921 (same protocol) |
| **E2: main** | **Generalised few-shot, 22 classes.** Train the encoder on the 14 base classes (official training set). Add the 8 novel classes via prototypes from their training images. Test on the official test set of all 22 classes. Report **base accuracy, novel accuracy, their harmonic mean, and macro-F1** | (a) DenseNet201 + weighted cross-entropy trained on all 22 classes (your own re-run, same split); (b) DMLF-Net 89.89% and the 88.31% preprint, for context only (different or unknown protocols) |
| **E3** | **K-shot curve on novel classes.** Support restricted to K = 1, 3, 5, 10 particles (drawn from distinct images where possible), with many random repetitions, mean ± 95% CI | Frozen-ImageNet-feature prototypes (no TEM training) |
| **E4** | **Image-disjoint cross-validation on novel classes.** Novel classes never touch encoder training, so all their images (train+val+test) can rotate between support and query, always split by image. This fixes the tiny-test-set problem (§4) | Same as E3 |
| **E5** | **Novelty detection.** Leave one class out entirely and check whether the distance threshold flags it as "unknown" (AUROC) | None published for this dataset |
| **E6** | **Ablations.** Loss variants (§5.4), backbone (DenseNet201 / ResNet50), with or without 8-orientation prototype averaging | — |

---

## 7. Feasibility test to run first, before any full training

**[Reasoning]** These are small, cheap steps. Each one answers a go/no-go question.

| Step | What to do | Go criterion |
|---|---|---|
| **F0: data audit** | Run the official preprocessing. Count patches per class and split. Confirm no image appears in two splits. Confirm the counts match Table 1 | Counts match Table 1; no overlap |
| **F1: frozen-feature probe** | Extract frozen ImageNet DenseNet201 embeddings (no training). Run nearest-prototype classification on the 14-class official test set, and compare with a linear probe on the same features | Nearest-prototype accuracy is far above chance (7.1%) and not far below the linear probe, so the embedding space carries class structure |
| **F2: pseudo-novel simulation** (the key test) | Hide 4 of the 14 base classes. Train the Siamese encoder on the other 10 for a short run. Evaluate 4-way K-shot on the hidden classes, image-disjoint | The trained encoder **beats the frozen ImageNet encoder** on the hidden classes. This proves the learned metric transfers to classes it has never seen, which is the core assumption of the whole approach |
| **F3: relative-pair check** | In F2, choose hidden classes that have relatives among the training classes (e.g. hide Orf and keep others), and see how far below average they fall | Tells you how hard Pseudocowpox vs Orf, or arenaviruses vs Lassa, will be |

- If **F2 fails** (the trained metric does not beat frozen features on unseen classes), pivot to "fine-tuned classifier + cosine classifier for novel classes" (Chen 2019 Baseline++) instead of episodic or Siamese training.
- F0–F2 need only a fraction of a full training run. F1 involves no training at all.

---

## 8. Bottom line

- **Novel:** no Siamese or few-shot work on this dataset was found.
- **Feasible:** yes, using established parts: an ImageNet-pretrained DenseNet201 (proven on this dataset), a supervised-contrastive Siamese encoder, and a nearest-prototype classifier.
- **Where it adds value:** bringing the **8 excluded rare classes** back in without retraining, handling **imbalance**, and enabling **novelty detection**.
- **Where it probably won't:** pushing 14-class accuracy far above 93.1%. Don't make that the headline claim.
- **Biggest risks:** fine-grained confusion between related viruses, and very small test sets for rare classes. These are mitigated by E3/E4 and confidence intervals.

---

## Sources

- Matuszewski & Sintorn (2021), CMPB 209:106318. https://doi.org/10.1016/j.cmpb.2021.106318
- Bengtsson Bernander et al. (2024), Sci. Rep. 14:14995. https://doi.org/10.1038/s41598-024-65597-x
- Sikder et al. (2024), Sci. Rep. 14:28954. https://doi.org/10.1038/s41598-024-80013-0
- Usman et al. (2024), EAAI 138:109348. https://doi.org/10.1016/j.engappai.2024.109348
- Hoque et al. (2026), arXiv:2606.14722 (preprint). https://arxiv.org/abs/2606.14722
- Xiao et al. (2023), Int. J. Wavelets Multiresolution Inf. Process., few-shot SARS-CoV-2 segmentation. https://doi.org/10.1142/s0219691323500479
- Koch, Zemel & Salakhutdinov (2015), *Siamese Neural Networks for One-shot Image Recognition*, ICML Deep Learning Workshop.
- Snell, Swersky & Zemel (2017), *Prototypical Networks for Few-shot Learning*, NeurIPS. https://arxiv.org/abs/1703.05175
- Chen et al. (2019), *A Closer Look at Few-shot Classification*, ICLR.
- Tian et al. (2020), *Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?*, ECCV.
- Guo et al. (2020), *A Broader Study of Cross-Domain Few-Shot Learning*, ECCV. https://arxiv.org/abs/1912.07200
- Khosla et al. (2020), *Supervised Contrastive Learning*, NeurIPS.
- Hermans, Beyer & Leibe (2017), *In Defense of the Triplet Loss for Person Re-Identification*, arXiv:1703.07737.
