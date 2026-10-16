# Published Journal Papers Using the TEM Virus Dataset (Mendeley Data `x4dwwfwtw3`, v3)

**Dataset:** Matuszewski, D. J. & Sintorn, I.-M. (2021). *TEM virus dataset*. Mendeley Data, V3. DOI: [10.17632/x4dwwfwtw3.3](https://data.mendeley.com/datasets/x4dwwfwtw3/3)

**Compiled:** 14 October 2026

---

## How this list was built (read first)

- **Finding papers.** I started from the papers that cite the dataset paper (Matuszewski & Sintorn, *CMPB* 2021), as listed by OpenAlex (43 citing works on the compile date). I then ran extra web searches. Each candidate was checked to see whether it **actually uses** the dataset, or only cites it. Many citing papers (for example, monkeypox skin-image papers) only cite it.
- **Rule: nothing invented.** Every number and claim below comes from the paper itself. Where only the abstract or the authors' public repository could be accessed, this is stated under **"Source access"**. Fields the paper does not report, or that I could not access, are marked **"not reported"** or **"not accessible"**.
- **Comparability warning.** The papers do **not** all use the same classes, split, or metric. Read the "Classes / split used" line of each paper before comparing numbers. See the summary table at the end.

---

## 1. Matuszewski & Sintorn (2021): the dataset paper (official benchmark)

| Field | Detail |
|---|---|
| Title | *TEM virus images: Benchmark dataset and deep learning classification* |
| Authors | Damian J. Matuszewski, Ida-Maria Sintorn (Uppsala University; Vironova AB) |
| Journal | *Computer Methods and Programs in Biomedicine*, vol. 209, 106318 (2021). DOI: [10.1016/j.cmpb.2021.106318](https://doi.org/10.1016/j.cmpb.2021.106318) |
| Source access | Full text read (open access, CC BY-NC-ND) |
| Classes / split used | The **14 most-represented classes**, with the authors' own **representative image-level split** (≈60/20/20 train/val/test) |

### Contribution (as stated by the authors)
1. Publishes and describes the full annotated TEM virus dataset: 1245 images, 22 classes. This is the first time the whole collection was released, after curation and partitioning.
2. Proposes a **representative train/validation/test split** for the dataset.
3. Reports benchmark results for virus classification using several established networks (large and small) plus a handcrafted small CNN.
4. Compares **transfer learning with training from scratch**. The authors hypothesise that, on a limited dataset, transfer learning is crucial for large networks, while their small custom network does relatively well from scratch.
5. Shows that a badly chosen data split substantially lowers performance. The authors use this to highlight the need for application-domain knowledge when building splits and interpreting results.

### Novelty
- Before this release, the authors note that the only public TEM virus set suitable for ML was the Kylberg Virus Texture Dataset. That set contains fixed-size cut-outs, so it suits texture analysis but not segmentation or combined segmentation and recognition. This dataset gives full raw images with point/centre-line annotations, image patches, and an official split.
- It is the first benchmark that explicitly studies how split strategy affects results on this kind of data (the "bad data split" experiment).

### Model architecture
- **Custom CNN (trained from scratch):** inspired by ResNet. It has 4 residual blocks of double convolutional layers with batch normalisation, max-pooling, and dropout. These are followed by a classification block: a final convolutional layer and 3 fully connected layers, with dropout between the first two. 14,651,401 trainable parameters.
- **Transfer-learning networks (ImageNet pre-trained):** DenseNet201, InceptionV3, MobileNetV2, ResNet50, ResNet50V2, VGG16, VGG19, and Xception. Each original top was replaced with 3 fully connected layers with dropouts in between. The best-performing networks were also trained from scratch.
- **Training:** 300 epochs from scratch and 50 epochs for fine-tuning, with categorical cross-entropy and early stopping on validation performance. The best models were reached after about 250 epochs (from scratch) and about 30 epochs (fine-tuned). Implemented in TensorFlow/Keras. Code and models are on Mendeley: [10.17632/kxsvzhcfgs.2](https://data.mendeley.com/datasets/kxsvzhcfgs/2).
- **Metrics:** accuracy, plus precision, recall, and F1 **averaged across classes, with each class weighted equally**.

### Performance (official test set, 14 classes; Table 2 of the paper)

| Model | Trainable params | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Custom (from scratch) | 14,651,401 | 0.901 | 0.895 | 0.898 | 0.893 |
| **DenseNet201 (transfer)**, best | 144,987,022 | **0.931** | 0.926 | 0.921 | **0.921** |
| DenseNet201 (from scratch) | 144,987,022 | 0.811 | 0.775 | 0.821 | 0.791 |
| InceptionV3 (transfer) | 98,330,798 | 0.895 | 0.887 | 0.894 | 0.889 |
| MobileNetV2 (transfer) | 87,174,926 | 0.846 | 0.803 | 0.849 | 0.813 |
| ResNet50 (transfer) | 158,817,294 | 0.898 | 0.883 | 0.881 | 0.877 |
| ResNet50V2 (transfer) | 158,802,062 | 0.887 | 0.862 | 0.885 | 0.869 |
| VGG16 (transfer) | 49,334,094 | 0.923 | 0.906 | 0.916 | 0.908 |
| VGG19 (transfer) | 54,643,790 | 0.918 | 0.909 | 0.924 | 0.912 |
| Xception (transfer) | 156,089,654 | 0.895 | 0.887 | 0.900 | 0.891 |
| Custom (from scratch), **bad split** | 14,649,867 | 0.765 | 0.751 | 0.619 | 0.640 |
| VGG16 (transfer), **bad split** | 49,334,094 | 0.832 | 0.822 | 0.726 | 0.737 |
| VGG16 (from scratch), **bad split** | 49,334,094 | 0.678 | 0.604 | 0.560 | 0.559 |
| DenseNet201 (transfer), **bad split** | 144,987,022 | 0.805 | 0.794 | 0.705 | 0.711 |
| DenseNet201 (from scratch), **bad split** | 144,987,022 | 0.689 | 0.639 | 0.556 | 0.558 |

### Key findings and error analysis (from the paper)
- The custom network from scratch beats DenseNet201 from scratch while being about 10× smaller. DenseNet201 does much better when fine-tuned from ImageNet. The authors read this as a sign the dataset is too small to train large networks from scratch.
- **Ebola** and **Adenovirus** instances were hard for both models. Adenovirus and Rift Valley have similar size and shape. Ebola and Marburg are related, very elongated viruses.
- Most misclassified particles came from a couple of problematic test images: poor quality, particle decomposition, clustering, staining artifacts. The authors deliberately **kept these images in the test set**, so the benchmark reflects real-world difficulty.

---

## 2. Nanni, Loreggia, Barcellona & Ghidoni (2023): ensembles of CNNs and transformers

| Field | Detail |
|---|---|
| Title | *Building Ensemble of Deep Networks: Convolutional Networks and Transformers* |
| Authors | Loris Nanni, Andrea Loreggia, Leonardo Barcellona, Stefano Ghidoni (Univ. of Padova; Univ. of Brescia) |
| Journal | *IEEE Access* (2023). DOI: [10.1109/ACCESS.2023.3330442](https://doi.org/10.1109/ACCESS.2023.3330442) |
| Source access | Full text read (accepted author version, CC BY 4.0) |
| Classes / split used | Their dataset table lists "TEM": **14 classes**, **1245 samples**, protocol **"80:20"**, citing Matuszewski & Sintorn. The TEM virus dataset is one of **9 benchmark datasets** (plankton, breast grading, laryngeal, pests, Kylberg virus "VIR", and TEM). |

### Contribution (as stated)
1. A set of **new Adam-based optimizer variants**. The three proposed are **Hyp**, **MinD**, and **AI**, alongside the DGrad, Cos, and Exp variants from earlier work.
2. Ensembles of CNNs (ResNet50, EfficientNetB0) trained with these optimizers, combined with ensembles of **transformers** (DeiT, ViT, Swin, CoAtNet).
3. An empirical evaluation across many small/medium datasets, plus a **testbed/baseline for transformer ensembles**.

### Novelty (as stated)
- The authors say that, to their knowledge, this is the first in-depth study of a set of transformers and CNNs over a large set of small/medium-sized image datasets.
- Fusion uses the weighted sum rule over separately trained CNNs and transformers. It needs minimal parameter tuning and no dataset-specific preprocessing.

### Model architecture and training
- **CNNs:** ImageNet-pretrained ResNet50 and EfficientNetB0, last layer replaced, no frozen layers. 224×224 input, 20 epochs, mini-batch 30, learning rate 0.001. The optimizer is chosen at random from the available variants. Data augmentation (flipping, rescaling) is applied only if the training set has fewer than 5000 images. A 0.25 validation split is held out from each training set.
- **Transformers (from timm, fine-tuned):** DeiT-Base/16, ViT-Base/16, Swin-Base/4, and CoAtNet.
- **Ensembling:** softmax on each model's scores, then sum/average-rule fusion. Training is repeated 10 times per model per dataset.

### Performance on the TEM virus dataset (Table 4 / Table 9 of the paper)

| Model / ensemble | Accuracy (%) | F1 |
|---|---|---|
| DeiT ×1, D(1) | 95.8 | 0.948 |
| Swin ×1, S(1) | 95.1 | 0.953 |
| ViT ×1, V(1) | 95.5 | 0.939 |
| CoAtNet ×1, C(1) | 95.8 | 0.954 |
| D(10) | 96.5 | 0.956 |
| (D+V)(5) / (D+V)(10) | 96.7 | 0.962 |
| (D+V+S+C)(2) | 96.6 | 0.964 |
| CNNs (Hyp+MinD+AI ensemble of ResNet50 + EffNetB0) | 96.3 | 0.965 |
| **CNNs + (D+V+S+C)(10)** | **96.7** | **0.965** |

- Error under the ROC curve (EUC, %, lower is better) on TEM: (D+V+S+C)(10) 0.241; CNNs 0.260; CNNs+(D+V+S+C)(10) 0.228.
- SOTA comparison table (accuracy, TEM column): this work 96.7; Sikder et al. preprint 96.4; Ali et al. (2022, TSP conference) 96.1; Matuszewski & Sintorn 93.1.

---

## 3. Owais, Cho & Park (2023): dataset used as one class in a 70-class medical database

| Field | Detail |
|---|---|
| Title | *Volumetric Model Genesis in Medical Domain for the Analysis of Multimodality 2-D/3-D Data Based on the Aggregation of Multilevel Features* |
| Authors | Muhammad Owais, Se Woon Cho, Kang Ryoung Park |
| Journal | *IEEE Transactions on Industrial Informatics* (2023). DOI: [10.1109/TII.2023.3252541](https://doi.org/10.1109/TII.2023.3252541) |
| Source access | Full text read (accepted author version) |
| How the dataset is used | All **1245 images** form a **single class**, "Microscopic Virus" (class A1, notation MS/I/1245). It is one of **70 classes** built by fusing **26 datasets** (151,095 samples in total). **It is not used for virus-type classification.** |

- **Contribution / novelty (as stated):** a deep volumetric classification network based on aggregating multilevel deep features (MDF-RN with DF-blocks, plus a second-stage STB-CN for 3D data) for large-scale multimodal 2D/3D medical data. 15 baseline methods were run under the same protocol.
- **Protocol:** two-fold cross-validation with 40% train (60,432), 10% validation (15,108), and 50% test (75,554) of the whole database. SGD, learning rate 0.001, MATLAB R2019a.
- **Performance:** overall AUC of 93.66% on the whole 70-class database. **No virus-specific result is reported.**

---

## 4. Bengtsson Bernander, Sintorn, Strand & Nyström (2024): rotation-equivariant CNNs

| Field | Detail |
|---|---|
| Title | *Classification of rotation-invariant biomedical images using equivariant neural networks* |
| Authors | Karl Bengtsson Bernander, Ida-Maria Sintorn, Robin Strand, Ingela Nyström (Uppsala University; Vironova AB) |
| Journal | *Scientific Reports* 14, 14995 (2024). DOI: [10.1038/s41598-024-65597-x](https://doi.org/10.1038/s41598-024-65597-x) |
| Source access | Full text read (open access, CC BY 4.0) |
| Classes / split used | **14 classes**, 256×256 patches at 1 nm/pixel. **Official validation and test sets unchanged.** The training set was **re-balanced to 93 particles per class** (the smallest class count), giving 1302 training samples. Sets were kept separate **at image level to prevent leakage**. Also uses BloodMNIST. |

### Contribution (as stated)
1. Builds a VGG16 baseline CNN and a version made **equivariant to the p4 group** (rotations by multiples of 90°) using group convolutions (GCNN).
2. Shows on TEM virus data that the GCNN gives higher top validation accuracy (+7.6% on average) and faster convergence (23.1% of the baseline's time to stability).
3. Concludes that rotation augmentation can be skipped when using the GCNN.
4. Studies **data efficiency**. Fitting a power law of error against training-set size gives a slope of −0.43 for the GCNN and −0.26 for the CNN, so the GCNN improves faster as data is added.

### Novelty
- Extends prior evidence on equivariant networks to TEM virus images (and BloodMNIST), adding a **predictable-scaling / power-law** analysis of data efficiency on this dataset.

### Model architecture and training
- Baseline: VGG16-style with 13 conv layers (channels 32-32-64-64-128-128-128-256-256-256-256-256-256). Conv layers 1–12 use (3,1,1) and layer 13 uses (4,0,0). Max-pool layers 1–4 use (2,2,0) and layer 5 uses (13,1,0). Linear layers (256,4096)-(4096,4096)-(4096,14).
- GCNN: every convolution is replaced with a p4 group convolution (E2CNN library on PyTorch), with a group-pooling layer before the linear layers for invariance. Channels are halved to keep the same number of weights.
- Cross-entropy loss, He initialisation, Adam with learning rate 1e-5, 100 epochs, ReLU, dropout 0.5 between linear layers. NAISS cloud, A40 GPU. Code: [github.com/kbbernander/TEM-equivariance](https://github.com/kbbernander/TEM-equivariance).
- Augmented training variant: each image rotated 90/180/270° and reflected over one axis, giving 8 images per original.

### Performance on TEM virus data (validation set, mean ± SD over 5 runs, Table 3)

| Model | No augmentation: best acc. (%) | Epochs / time to stability (s) | With augmentation: best acc. (%) | Epochs / time to stability (s) |
|---|---|---|---|---|
| CNN (VGG16) | 57.99 ± 2.27 | 36 / 2076 ± 336 | 76.59 ± 0.76 | 21 / 10,581 ± 2672 |
| GCNN (p4) | 82.43 ± 1.95 | 25 / 2440 ± 718 | **84.49 ± 1.63** | 7 / 3894 ± 717 |

- The GCNN without augmentation beats the CNN with augmentation (82.43% vs 76.59%, paired t-test p = 0.0081). The convergence-time difference is significant (p = 0.0021).
- Test-set confusion matrices (first run): average sensitivity 0.67 for the CNN and 0.71 for the GCNN; average specificity 0.97 and 0.98.
- With 714 training samples, top accuracy is 44.06% for the CNN and 74.88% for the GCNN.
- **5-fold cross-validation** (train and test merged; 100 training / 23 test images per class): mean test accuracy 87.39% for the GCNN and 85.03% for the CNN (SD 4.18 and 3.70). The authors note that these higher numbers may come from **data leakage** caused by mixing the original image-level train and test sets.

---

## 5. Sikder, Khan, Bairagi, Masud, Tiang & Nahid (2024): functional dual-branch CNN with DIP preprocessing

| Field | Detail |
|---|---|
| Title | *Heterogeneous virus classification using a functional deep learning model based on transmission electron microscopy images* |
| Authors | Niloy Sikder, Md. Al-Masrur Khan, Anupam Kumar Bairagi, Mehedi Masud, Jun Jiat Tiang, Abdullah-Al Nahid |
| Journal | *Scientific Reports* 14, 28954 (2024). DOI: [10.1038/s41598-024-80013-0](https://doi.org/10.1038/s41598-024-80013-0). Preprint: [arXiv:2405.15563](https://arxiv.org/abs/2405.15563) |
| Source access | Full text read (open access) |
| Classes / split used | **14 classes, 736 samples per class** (the balanced/augmented patch set). **Not the official split:** 75% of each class was randomly assigned to training and the rest to testing, giving **184 test samples per class (2576 in total)**. |

### Contribution (as stated)
1. A virus classification pipeline combining **two image-processing (DIP) steps** for noise reduction and sample preparation with a **functional (multi-input) CNN**.
2. Shows that the two branches are complementary through an ablation study.
3. Claims higher accuracy than the earlier works on the dataset, with **fewer trainable parameters (679,262)** and short training time.

### Novelty
- Feeds the network two transformed versions of each image: a **local standard deviation filtered** image (3×3 window, symmetric padding; edge enhancement) and a **2D Discrete Cosine Transform (DCT)** image. Each goes through its own convolutional branch, and the features are concatenated before an MLP.

### Model architecture and training
- Input: images resized to **128×128**.
- **Branch 1** (local SD filtered input): 3 Conv2D layers (2 with Sigmoid, 1 with ReLU).
- **Branch 2** (2D DCT input): 4 Conv2D layers (all ReLU).
- Both branches have 3 max-pooling layers, 1 batch normalisation, and several dropout layers, then a Flatten. All convolutions use 3×3 filters.
- **Neural network head:** concatenation, then an MLP with 5 dense layers and a dropout layer, then softmax.
- 100 epochs on an i9-12900KF, 128 GB RAM, 24 GB GPU. About 2 s per epoch on GPU and 34 s per epoch on CPU. About 0.05 s to classify one image, excluding preprocessing.

### Performance (their random 75/25 test subset)
- Peak test **accuracy 97.44%** (epoch 96); last epoch 96.43%. Macro **precision 97.82%**, **recall 97.07%**, **F1 97.44%**, average **AUC 99.95%**, **QWK 0.9719**. Test loss 0.0808, KLD 0.0819.
- Hardest classes: RV (Rift Valley; F1 92.10%), Mb (Marburg; 93.83%), Eb (Ebola; 94.15%). Orf reached 100%.

**Ablation (Table 4):**

| Preprocessing | Model | Acc. (%) | Prec. (%) | Rec. (%) | F1 (%) |
|---|---|---|---|---|---|
| Local SD filtering | Branch 1 + NN | 76.67 | 78.76 | 75.42 | 77.05 |
| 2D DCT | Branch 2 + NN | 92.20 | 92.35 | 91.89 | 92.12 |
| Both | Proposed | 97.44 | 97.82 | 97.07 | 97.44 |

**Comparison table reported by the authors (Table 5):**

| Study | Classes | Classifier | Trainable params | Acc. (%) | F1 (%) |
|---|---|---|---|---|---|
| Matuszewski & Sintorn 2021 | 14 | DenseNet201 | 144,987,022 | 93.10 | 92.10 |
| Ali et al. 2022 (TSP conference) | 14 | CNN | 1,047,246 | 96.10 | 95.29 |
| Bengtsson Bernander et al. 2024 | 14 | GCNN | 2,564,096 | 84.49 | 75.95 |
| **Sikder et al. 2024** | 14 | CNN | 679,262 | **97.44** | **97.44** |

> Note: the parameter count and F1 = 75.95 listed for Bengtsson Bernander et al. are the values Sikder et al. report. Those two numbers do not appear in the Bengtsson Bernander paper itself, which reports the 84.49% figure as a **validation** accuracy.

> **Comparability note (derived from the papers' own protocols, not stated by Sikder et al.):** the dataset authors build training, validation, and test sets **at image level** to avoid leakage, and the 736-per-class set is the **augmented training set** (flips and 90° rotations). Sikder et al. split the 736-per-class samples **randomly at sample level**. Bengtsson Bernander et al. explicitly warn that mixing the original sets may cause leakage. So the 97.44% result is **not measured on the official test set** and is not directly comparable to the 93.1% benchmark.

---

## 6. Usman, Sultan, Hong, Kim, Akram, Gondal, Tariq & Park (2024): DMLF-Net, all 22 classes

| Field | Detail |
|---|---|
| Title | *Dilated multilevel fused network for virus classification using transmission electron microscopy images* |
| Authors | Muhammad Usman, Haseeb Sultan, Jin Seong Hong, Seung Gu Kim, Rehan Akram, Hafiz Ali Hamza Gondal, Muhammad Hamza Tariq, Kang Ryoung Park |
| Journal | *Engineering Applications of Artificial Intelligence*, vol. 138, 109348 (2024). DOI: [10.1016/j.engappai.2024.109348](https://doi.org/10.1016/j.engappai.2024.109348) |
| Source access | **Abstract plus the authors' GitHub repository only.** The full text is open access (CC BY) on ScienceDirect but was blocked by a bot check, so the details below are limited to the abstract. |
| Classes / split used | **All 22 virus classes.** The repository ([github.com/MuhammadUsmanMalik/DMLF-Net](https://github.com/MuhammadUsmanMalik/DMLF-Net)) names "Dataset-1" as `10.17632/x4dwwfwtw3.3` (this dataset) and "Dataset-2" as the Kylberg Virus Texture dataset. The split protocol was **not accessible**. |

- **Contribution / novelty (from the abstract):** earlier studies classified at most 16 virus classes. This work targets **22 classes** with as few trainable parameters as possible. It proposes **DMLF-Net**, which uses **multilevel feature fusion within the network** to exploit more abstract features. A **multi-stage training strategy** is used to converge without overfitting, and there is a **classification-driven retrieval framework** for automatic virus identification. Evaluated on three open databases: two virus datasets and one bacteria dataset (DIBaS).
- **Architecture:** dilated, multilevel fused CNN with **25.5 M trainable parameters**, about 5.3× fewer than the second-best model, VGG16 (134.3 M). Layer-level details were not accessible.
- **Performance on the 1st virus dataset (this dataset, 22 classes):** **accuracy 89.89%, F1 83.39%, AUC 92.50%**. Second virus dataset (Kylberg): accuracy 80.70%, F1 81.20%, AUC 86.20%. Bacteria (DIBaS): accuracy 95.93%, F1 96.24%.

---

## 7. Bisht, Karthik, Ajay & Singh (2025): Virus-FusionNet

| Field | Detail |
|---|---|
| Title | *Virus-FusionNet: an explainable deep learning-based approach for virus classification from TEM images using DenseNet and lightweight hierarchical semi-local feature extraction network* |
| Authors | Akshaj Singh Bisht, R. Karthik, Armaano Ajay, Abhay Karan Singh (VIT Chennai) |
| Journal | *Engineering Research Express* 7(3), 035242 (2025). DOI: [10.1088/2631-8695/adf411](https://doi.org/10.1088/2631-8695/adf411) |
| Source access | **Abstract and data-availability statement only** (paywalled). The data statement points to Mendeley `x4dwwfwtw3` v3. |
| Classes / split used | **Not accessible** |

- **Architecture (from the abstract):** dual-track network.
  - **Track 1:** DenseNet, for local features.
  - **Track 2:** a custom **Lightweight Hierarchical Semi-Local Feature Extraction Network (LHSLFN)**. It contains a **Shuffled Flexi-Fusion Enhanced Convolutional (SFFEC)** block for global features and a **Dynamic Multi-path Semi-Local Enhancement (DMSLE)** block for local and semi-local features.
  - The concatenated features are refined by **Multidimensional Collaborative Attention (MCA)**.
- **Contribution / novelty (as stated):** combines local, semi-local, and global features with attention, and adds **interpretability through Grad-CAM and SHAP**.
- **Performance:** **accuracy 99.73%** on the TEM virus dataset. The authors claim it outperforms existing works and state-of-the-art networks. Other metrics, the class count, and the split are not accessible.

---

## Summary table (journal papers)

| # | Paper | Venue / year | Classes | Split / protocol | Best reported result on this dataset | Full text read? |
|---|---|---|---|---|---|---|
| 1 | Matuszewski & Sintorn | CMPB 2021 | 14 | **Official** image-level ≈60/20/20 | DenseNet201 (transfer): Acc 93.1%, F1 0.921 (test) | Yes |
| 2 | Nanni et al. | IEEE Access 2023 | 14 | "80:20" (as stated) | CNNs + transformers ensemble: Acc 96.7%, F1 0.965 | Yes |
| 3 | Owais et al. | IEEE TII 2023 | 1 (whole dataset = one class of 70) | 2-fold CV over the 26-dataset database | No virus-specific result | Yes |
| 4 | Bengtsson Bernander et al. | Sci. Rep. 2024 | 14 | Official val/test; training re-balanced to 93/class | GCNN + aug.: 84.49% (validation); 5-fold CV test 87.39% | Yes |
| 5 | Sikder et al. | Sci. Rep. 2024 | 14 | Random 75/25 of 736/class | Acc 97.44%, F1 97.44% | Yes |
| 6 | Usman et al. (DMLF-Net) | EAAI 2024 | **22** | Not accessible | Acc 89.89%, F1 83.39%, AUC 92.50% | No (abstract and repo) |
| 7 | Bisht et al. (Virus-FusionNet) | Eng. Res. Express 2025 | Not accessible | Not accessible | Acc 99.73% | No (abstract) |

---

## Related non-journal works (listed for completeness, not included above)

- **Ali, Joshi, Dutta, Burget & Mezina (2022)**, *Deep learning-based classification of viruses using transmission electron microscopy images*, 45th Int. Conf. on Telecommunications and Signal Processing (TSP), pp. 174–178 (**conference**). As reported by Sikder et al. (2024): a custom CNN with six convolutional layers, 14 classes, **96.10% accuracy / 95.29% F1**, 1,047,246 parameters, compared with XceptionNet, DenseNet201, and MobileNet. Full text not read.
- **Bylander, Nyström & Bengtsson Bernander (2024)**, *Equivariant Neural Networks for TEM Virus Images Improves Data Efficiency*, LNCS (**conference**), DOI 10.1007/978-3-031-78128-5_2. Abstract not accessible.
- **Hoque et al. (2026)**, *A Comparative and Hybrid Study of CNN and Transformer Models for Multi-Class Virus Classification in TEM*, [arXiv:2606.14722](https://arxiv.org/abs/2606.14722) (**preprint, not peer-reviewed**). 22 classes. Raw images resized to 224×224 (bilinear) with **no pixel-size normalisation**, ImageNet normalisation, weighted cross-entropy, and a stratified split (ratio not stated). Results: Swin accuracy 0.8831 / macro-F1 0.8444; DeiT accuracy 0.8669; ResNet50 accuracy 0.5887; performance-weighted hybrid accuracy 0.8831 / macro-F1 0.8528.
- **Nithiyanandam et al. (2025)**, *Enhancing Classification Accuracy in Virology Through Deep Learning for Accurate Virus Identification from TEM Imagery*, in *Medical Virology* (Springer **book chapter**), DOI 10.1007/978-981-96-8919-4_16. Content not accessible.

## Screened and excluded (they cite the dataset paper but do not use the dataset)

- Vivek et al., *Computational Intelligence and Neuroscience* (2022), DOI 10.1155/2022/3531308. Full text checked: it uses the PMID2019, CEM500K, and BSST265 datasets, not the TEM virus dataset.
- Xiao et al., *Briefings in Bioinformatics* (2023), VISN. Per its abstract, it uses its own new TEM dataset (SARS-CoV-2, H1N1, RSV, HSV-1, HAdV-5, Vaccinia) for instance segmentation.
- Xiao et al., *Int. J. Wavelets Multiresolution Inf. Process.* (2023). Per its abstract, it uses its own pre-training virus dataset for few-shot SARS-CoV-2 segmentation.
- Wei et al., *Applied Sciences* (2025). Per its abstract, it does YOLOv11 detection on unprocessed bulk biopharmaceutical samples. Full text not accessible.
- Many monkeypox, CT, polymer, angiography, and review papers in the citation list cite the paper without using the data.

---

## Sources

- Mendeley dataset page: https://data.mendeley.com/datasets/x4dwwfwtw3/3
- Matuszewski & Sintorn 2021: https://doi.org/10.1016/j.cmpb.2021.106318 (full text via the Uppsala DiVA repository, archived copy)
- Code for the 2021 paper: https://data.mendeley.com/datasets/kxsvzhcfgs/2
- Nanni et al. 2023: https://doi.org/10.1109/ACCESS.2023.3330442
- Owais et al. 2023: https://doi.org/10.1109/TII.2023.3252541
- Bengtsson Bernander et al. 2024: https://www.nature.com/articles/s41598-024-65597-x
- Sikder et al. 2024: https://www.nature.com/articles/s41598-024-80013-0 · https://arxiv.org/abs/2405.15563
- Usman et al. 2024: https://www.sciencedirect.com/science/article/pii/S0952197624015069 · https://github.com/MuhammadUsmanMalik/DMLF-Net
- Bisht et al. 2025: https://iopscience.iop.org/article/10.1088/2631-8695/adf411
- Hoque et al. 2026 (preprint): https://arxiv.org/abs/2606.14722
- Citation list: OpenAlex, works citing W3171728179
