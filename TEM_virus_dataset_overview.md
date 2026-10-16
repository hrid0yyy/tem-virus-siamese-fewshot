# TEM Virus Dataset: Composition, Official Preprocessing and Split, Pros and Cons

**Dataset:** Matuszewski, D. J. & Sintorn, I.-M. (2021). *TEM virus dataset*. Mendeley Data, V3. DOI: [10.17632/x4dwwfwtw3.3](https://data.mendeley.com/datasets/x4dwwfwtw3/3)
**Companion paper:** Matuszewski & Sintorn, *TEM virus images: Benchmark dataset and deep learning classification*, *Computer Methods and Programs in Biomedicine* 209 (2021) 106318. DOI: [10.1016/j.cmpb.2021.106318](https://doi.org/10.1016/j.cmpb.2021.106318)

**Compiled:** 14 October 2026

> **Sourcing rule.** Every statement below comes from the Mendeley dataset page, the companion paper, or the published papers that used the dataset (listed in `TEM_virus_dataset_published_papers.md`). Each point is tagged with its source:
> **[M]** Mendeley page · **[MS21]** Matuszewski & Sintorn 2021 · **[BB24]** Bengtsson Bernander et al., Sci. Rep. 2024 · **[S24]** Sikder et al., Sci. Rep. 2024 · **[N23]** Nanni et al., IEEE Access 2023 · **[U24]** Usman et al., EAAI 2024 (abstract) · **[H26]** Hoque et al., arXiv 2026 (preprint, not peer-reviewed).
> The few lines that are my own reading of these sources, not a claim made by any paper, are marked **"Observation"**.

---

## 1. Dataset at a glance

| Property | Value | Source |
|---|---|---|
| Publisher / host | Mendeley Data, Uppsala Universitet | [M] |
| Version / date | Version 3, published 11 May 2021 | [M] |
| Licence (dataset) | CC BY 4.0 | [M] |
| Contributors | Damian Matuszewski, Ida-Maria Sintorn | [M] |
| Total images | **1245** | [M], [MS21] |
| Classes | **22** virus classes | [M], [MS21] |
| Microscopes | LEO (Zeiss) with Morada (Olympus) camera; Tecnai 10 (FEI) with MegaView III (Olympus) camera | [M], [MS21] |
| Sample preparation | 10% phosphate-buffered saline; carbon-coated TEM grids; 2% phosphotungstic acid (negative stain) | [M], [MS21] |
| Raw image size | 1376 × 1032 or 2048 × 2048 px, depending on the microscope | [M], [MS21] |
| Pixel size | 0.26 – 5.57 nm (different magnifications) | [M], [MS21] |
| Image format | Grayscale, `.tif` | [S24] |
| Annotation type | Approximate **centre point** per particle (isolated spherical viruses) or a **centre-line** (clustered or elongated viruses). **No segmentation masks.** | [M], [MS21] |
| Annotation storage | Coordinate points in a separate text file for each image | [M], [MS21] |
| Class imbalance | 9 – 129 images per class; 38 – 1934 particles per class | [M], [MS21] |
| Curation | Overlapping images (those capturing the same virus particles) removed before release | [MS21] |
| Intended tasks | Virus detection, segmentation, classification, and novelty detection | [MS21] |

### Released variants [MS21]
The paper lists these variants. Its text says "all four datasets", but only three are itemised:
1. **Raw images**: the original TEM images with their annotations and metadata.
2. **Image patches**: cropped from images rescaled to 1 nm per pixel.
3. **Selected classes**: the virus classes used in the paper's classification experiments, including the **augmented training set**.

All variants are split into training, validation, and test sets in the same way [MS21].

---

## 2. Per-class composition and official split (Table 1 of [MS21])

**Bold** classes are the 14 used for classification in [MS21]. The other 8 were excluded.

| Virus | Av. particle size* | Images train | val | test | **Images total** | Particles train | val | test | **Particles total** |
|---|---|---|---|---|---|---|---|---|---|
| **Adenovirus** | 80 | 40 | 13 | 14 | **67** | 160 | 44 | 86 | **290** |
| **Astrovirus** | 25 | 46 | 15 | 15 | **76** | 266 | 86 | 66 | **418** |
| **CCHF** | 120 | 44 | 15 | 15 | **74** | 249 | 100 | 86 | **435** |
| **Cowpox** | 270 | 30 | 10 | 10 | **50** | 189 | 70 | 57 | **316** |
| Dengue | 45 | 19 | 6 | 7 | 32 | 72 | 15 | 44 | 131 |
| **Ebola** | 80 | 65 | 22 | 22 | **109** | 173 | 70 | 60 | **303** |
| Guanarito | 140 | 13 | 4 | 4 | 21 | 31 | 10 | 7 | 48 |
| **Influenza** | 110 | 56 | 19 | 19 | **94** | 365 | 158 | 125 | **648** |
| **Lassa** | 140 | 62 | 21 | 21 | **104** | 222 | 92 | 80 | **394** |
| LCM | 120 | 22 | 7 | 8 | 37 | 31 | 10 | 14 | 55 |
| Machupo | 120 | 20 | 6 | 7 | 33 | 69 | 16 | 23 | 108 |
| **Marburg** | 80 | 57 | 19 | 19 | **95** | 78 | 28 | 24 | **130** |
| **Nipah** | 95 | 25 | 8 | 8 | **41** | 75 | 22 | 17 | **114** |
| **Norovirus** | 30 | 32 | 11 | 11 | **54** | 231 | 104 | 84 | **419** |
| **Orf** | 145 | 38 | 13 | 13 | **64** | 92 | 76 | 31 | **199** |
| **Papilloma** | 55 | 19 | 6 | 6 | **31** | 693 | 227 | 187 | **1107** |
| Pseudocowpox | 145 | 20 | 7 | 7 | 34 | 46 | 21 | 16 | 83 |
| **Rift Valley** | 90 | 77 | 26 | 26 | **129** | 1140 | 408 | 386 | **1934** |
| **Rotavirus** | 80 | 22 | 7 | 7 | **36** | 169 | 48 | 40 | **257** |
| Sapovirus | 30 | 8 | 3 | 3 | 14 | 19 | 11 | 8 | 38 |
| TBE | 50 | 25 | 8 | 8 | 41 | 39 | 12 | 11 | 62 |
| WestNile | 50 | 5 | 2 | 2 | 9 | 88 | 66 | 34 | 188 |
| **Total (22)** | | **745** | **248** | **252** | **1245** | **4497** | **1694** | **1486** | **7677** |

\* The column header in the published PDF reads "av. particle size [μm]". Observation: the values (25–270) are in the typical size range of viruses in nanometres, so the unit label looks like a typo. The numbers are copied exactly as printed.

**Subtotals** (my arithmetic from the table above; the row sums were checked against the printed totals):

| Subset | Images (train / val / test / total) | Particles (train / val / test / total) |
|---|---|---|
| 14 classes used in classification | 613 / 205 / 206 / **1024** | 4102 / 1533 / 1329 / **6964** |
| 8 excluded classes (Dengue, Guanarito, LCM, Machupo, Pseudocowpox, Sapovirus, TBE, WestNile) | 132 / 43 / 46 / **221** | 395 / 161 / 157 / **713** |

**Why 8 classes were excluded [MS21]:** they are the least populous in terms of image or particle count. Too few samples would likely fail to cover each class's variation and would not be enough for good generalisation or effective training.

---

## 3. Official data-splitting procedure [MS21]

1. **Motivation.** The authors' earlier work found that, when classes have few images and a highly varied number of particles per image, the split must be done carefully to keep every set representative. The number of particles per image, and whether particles are separate or clustered, are features that CNNs learn on this data.
2. **Procedure (per class).**
   - Sort the class's images by the number of virus particles they contain.
   - Assign images in a repeating sequence: **train, test, train, validation, train, …** until no images remain.
   - This gives roughly **60% / 20% / 20%** (train / val / test) **by image**. Each set gets some images with few particles and some with many.
3. **Image-level split.** Sets are built at image level, and images showing the same particles were removed. Patches cut from one image can overlap, but the authors state this **does not cause train/val/test leakage**, because a whole image always goes to a single set.
4. **"Bad split" control experiment.** Images in each class were sorted by particle count, and the first 60% went to training, the next 20% to validation, and the last 20% to test. Every model scored substantially lower. For example, the custom CNN fell from accuracy 0.901 / F1 0.893 to 0.765 / 0.640, and DenseNet201 (transfer) fell from 0.931 / 0.921 to 0.805 / 0.711.

---

## 4. Official preprocessing pipeline [MS21]

| Step | What is done |
|---|---|
| 1. Rescaling | All images rescaled with **Lanczos-3 kernel interpolation** to a common pixel size of **1 nm** |
| 2. Patch extraction | **256 × 256 px patches (= 256 × 256 nm)** cropped around annotation points: the centre point for spherical particles, and **every centre-line vertex** for elongated particles |
| 3. Short centre-lines | If an elongated particle's centre-line has only **2 points** (typically an oval particle such as Orf), a new **midway point** is used as the patch centre |
| 4. Border handling | **Mirror padding** for patches that extend past the image border |
| 5. Patch-vs-particle note | Elongated viruses yield **more patches than particles**, especially Marburg, Ebola, Influenza, Lassa, and Nipah. For example, the test set has only 24 Marburg particles but 173 Marburg patches. Patches from the same image can overlap |
| 6. Class balancing (training set only) | Training patches augmented by **flipping and multiple 90° rotations** so that **each class has 736 samples**. Classes with more than 736 patches were **randomly reduced** to 736 |
| 7. Intensity normalisation | Each patch is standardised by subtracting its mean and dividing by its standard deviation |

(Steps 1, 2 and 6 are also summarised on the Mendeley page [M].)

---

## 5. How later papers preprocessed and split the data

| Paper | Classes | Input / preprocessing | Split / protocol |
|---|---|---|---|
| [MS21] Matuszewski & Sintorn 2021 | 14 | 1 nm/px Lanczos-3; 256×256 patches; flips and 90° rotations to 736/class (train only); per-patch z-score | **Official** image-level ≈60/20/20 |
| [N23] Nanni et al. 2023 | 14 | Resized to 224×224; flipping and rescaling augmentation only when the training set has fewer than 5000 images; random flips/rotations during training | "80:20" (as listed in their dataset table, citing [MS21]); 0.25 validation split held out from training |
| [BB24] Bengtsson Bernander et al. 2024 | 14 | 256×256 patches at 1 nm/px; training **re-balanced to 93 particles per class** (augmented images removed first, then random removal; 1302 training samples in total); optional augmentation of 90/180/270° rotations and one-axis reflection (8 images per original) | **Official val/test kept unchanged**; image level "to prevent data leakage". Also 5-fold CV on merged train+test (100 train / 23 test images per class) |
| [S24] Sikder et al. 2024 | 14 | Resized to 128×128; branch 1 gets a **3×3 local standard-deviation filter** (symmetric padding); branch 2 gets a **2D DCT** | **Random 75/25 per class** of the 736-per-class samples, giving 184 test samples per class (2576 in total) |
| [U24] Usman et al. 2024 (DMLF-Net) | **22** | Not accessible (abstract only) | Not accessible |
| [H26] Hoque et al. 2026 (preprint) | 22 | Raw images resized to 224×224 (bilinear) with **no pixel-size normalisation**; grayscale replicated to 3 channels; scaled to [0, 1] then ImageNet mean/std; random flips, mild rotations, random resized crops; **weighted cross-entropy** (inverse frequency) | "Stratified split" into train/validation (ratio not stated) |

**Observation** (from the table above, not a claim made by any paper): the published results use **at least four different protocols**: the official split, a random 75/25 patch split, an unspecified 80:20 split, and 5-fold CV. Bengtsson Bernander et al. and the dataset authors both stress image-level splitting to avoid leakage. Numbers from different protocols should not be compared directly.

---

## 6. Pros (advantages reported in the papers)

1. **Real-world, imperfect data.** The dataset is built to represent real problems "with all their variations, complexity, limitations, and noise", which the authors argue is needed to judge whether deep learning can be trusted in clinical and diagnostic settings [MS21].
2. **Multi-task.** It provides full raw images, point/centre-line annotations, extracted patches, and metadata, so it supports **detection, segmentation, classification, and novelty detection** [MS21].
3. **Fills a gap.** Before its release, the only public TEM virus set suitable for ML was the Kylberg Virus Texture Dataset of fixed-size cut-outs. Those suit texture analysis but not semantic segmentation or combined segmentation and recognition [MS21].
4. **Higher resolution and more samples per class than the Kylberg Virus Texture Dataset** (1500 images of 41×41 px over 15 classes) [S24].
5. **Official, representative, leakage-safe split**, with overlapping images removed, which gives a common benchmark [MS21], [BB24].
6. **Physical scale is known.** Pixel sizes are known, so all images can be resampled to a common 1 nm/pixel [MS21], [BB24], [S24].
7. **Two microscope/camera set-ups**, which add acquisition variability [M], [MS21].
8. **Rotation invariance.** Samples show global rotation invariance, which suits rotation-equivariant networks: equivariant networks achieved higher accuracy, faster convergence, and better data efficiency than a CNN [BB24].
9. **Open licence** (CC BY 4.0), with the baseline code and trained models also released on Mendeley (`kxsvzhcfgs`) [M], [MS21].
10. **Many classes.** All 22 classes allow a harder task than earlier virus studies, which used at most 16 classes [U24].

---

## 7. Cons, limitations and challenges (as reported)

1. **Weak annotation.** Particles have only a centre point or centre-line, with **no full segmentation masks** [M], [MS21].
2. **Few images per class** [M], [MS21].
3. **Strong class imbalance.** There are 9–129 images and 38–1934 particles per class [M], [MS21]. Convolutional models were more sensitive to the imbalance; for example, ResNet50 reached accuracy 0.5887 on 22 classes [H26, preprint].
4. **Eight classes are too small to use.** Dengue, Guanarito, LCM, Machupo, Pseudocowpox, Sapovirus, TBE, and WestNile were excluded from the benchmark classification [MS21]. When all 22 classes are used, the reported accuracy is lower (DMLF-Net: 89.89% accuracy, 83.39% F1) [U24].
5. **Image-quality problems.** These include diffuse virus boundaries, imperfect focus, noise, different magnifications, and large variation in virus, background, and debris appearance [M], [MS21].
6. **Problematic test images.** Particle decomposition, clustered particles, staining artifacts, and imperfect acquisition caused most errors. These images were **deliberately kept in the test set** [MS21].
7. **Inter-class similarity.** Adenovirus and Rift Valley have similar size and shape (different textures). Ebola and Marburg are related and both very elongated [MS21]. Rift Valley, Marburg, and Ebola had the lowest per-class F1 in [S24]. The 2026 preprint cites strong inter-class similarity and scale variability [H26].
8. **Too small for large networks trained from scratch.** DenseNet201 dropped from 93.1% (transfer) to 81.1% (from scratch). Transfer learning is considered crucial for large models on this dataset [MS21].
9. **Results depend heavily on the split.** A non-representative split substantially reduces performance [MS21]. In 5-fold CV, test accuracy varied with SD 3.70 (CNN) and 4.18 (GCNN). The authors read this as a sign the dataset is highly varied, for example in illumination or in the number and class of particles, so training is sensitive to which images are included [BB24].
10. **Leakage risk if the official split is ignored.** Mixing the original image-level train and test sets may cause data leakage and inflated accuracy [BB24]. Patches from one image can overlap, and elongated viruses produce many patches per particle (for example, 24 Marburg test particles give 173 patches) [MS21].
11. **Patches only, not a full pipeline.** The benchmark covers classification of patches only. A complete system would also need sample handling, automatic imaging, image quality control, sliding-window operation, and pre/post-processing [MS21].
12. **General TEM limitation.** Morphology in a TEM image cannot identify a virus beyond genus level, so other (molecular) tests are needed for a definitive diagnosis [S24], [BB24].

---

## 8. Benchmark numbers to beat (by protocol)

| Protocol | Best published result | Paper |
|---|---|---|
| Official test set, 14 classes | DenseNet201 (ImageNet transfer): **Acc 93.1%, F1 0.921**; custom CNN from scratch: Acc 90.1%, F1 0.893 | [MS21] |
| Official val set, re-balanced training (93/class), 14 classes | GCNN (p4) + augmentation: **84.49% ± 1.63** (validation) | [BB24] |
| "80:20", 14 classes | CNN + transformer ensemble: **Acc 96.7%, F1 0.965** | [N23] |
| Random 75/25 of 736/class, 14 classes | Dual-branch functional CNN: **Acc 97.44%, F1 97.44%** | [S24] |
| 22 classes (protocol not accessible) | DMLF-Net: **Acc 89.89%, F1 83.39%, AUC 92.50%** | [U24] |

(Virus-FusionNet, *Engineering Research Express* 2025, reports 99.73% accuracy, but its class count and split could not be accessed, so it is left out of this table.)

---

## Sources

- Mendeley dataset page: https://data.mendeley.com/datasets/x4dwwfwtw3/3
- [MS21] https://doi.org/10.1016/j.cmpb.2021.106318 · code: https://data.mendeley.com/datasets/kxsvzhcfgs/2
- [N23] https://doi.org/10.1109/ACCESS.2023.3330442
- [BB24] https://www.nature.com/articles/s41598-024-65597-x
- [S24] https://www.nature.com/articles/s41598-024-80013-0
- [U24] https://doi.org/10.1016/j.engappai.2024.109348
- [H26] https://arxiv.org/abs/2606.14722 (preprint)
