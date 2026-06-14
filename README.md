# Deep Learning-Based CNNs for Oral Lesion Diagnosis: A Comparative Study

> **Authors:** Gabriel Lucas S. Moura & Felipe D. Cunha  
> **Institution:** Department of Computer Science, PUC Minas — Belo Horizonte, Brazil  
> **Contact:** gl81922@gmail.com · felipe@pucminas.br

---

## Overview

Oral cancer accounts for approximately **390,000 new cases per year** worldwide, with a five-year survival rate below 50% in many countries — largely due to late diagnosis. Early oral lesions are subtle, often asymptomatic, and frequently missed during routine clinical examination.

This repository contains the code and experiments from a comparative study that develops and evaluates **five CNN architectures** for binary classification of oral lesions (Cancer vs. Non-Cancer) using a real-world clinical image dataset. All models were trained under identical conditions to ensure a fair and reproducible comparison.

**Key finding:** MobileNetV3-Small achieved the best overall performance (88.75% accuracy, 99.79% sensitivity, AUC = 0.9942), demonstrating that lightweight architectures can be highly effective — and even superior to heavier models — in medical imaging tasks.

---

## Repository Structure

```
.
├── Arquiteturas/Modelos/
│   ├── EfficientNetB4/
│   ├── InceptionV3/
│   ├── MobileNetV3/
│   ├── MobileNetV3Small/
│   └── ResNet-50/
├── .gitignore
└── README.md
```

- **`Arquiteturas/Modelos/`** — Training scripts and evaluation notebooks for each evaluated CNN architecture, developed and run on Google Colab.

> **Note:** The dataset is not included in this repository. Download it from [Kaggle — Oral Cancer Dataset](https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset) and [Kaggle — Oral Lesions Malignancy Detection Dataset](https://www.kaggle.com/datasets/mohamedgobara/oral-lesions-malignancy-detection-dataset) and place them locally before running the notebooks.

---

## Architectures Evaluated

| Architecture | Input Size | Notes |
|---|---|---|
| **MobileNetV3-Small** | 224 × 224 | Best overall performer; near-perfect sensitivity (99.79%) |
| **MobileNetV3-Large** | 224 × 224 | High sensitivity (97.26%); slightly lower accuracy than Small |
| **ResNet-50** | 224 × 224 | Most stable across runs; competitive balanced baseline |
| **EfficientNet-B4** | 380 × 380 | Very high sensitivity but low precision; tends to over-predict Cancer |
| **InceptionV3** | 299 × 299 | Weakest performer; highest variance across runs |

All models were initialized with **ImageNet pre-trained weights** and fully fine-tuned on the oral cancer dataset using transfer learning.

---

## Results

Performance metrics are reported as **mean ± standard deviation** over 3 independent runs with different random seeds (42, 123, 456).

| Architecture | Accuracy (%) | Precision (%) | Sensitivity (%) | F1-Score (%) | AUC |
|---|---|---|---|---|---|
| **MobileNetV3-Small** | **88.75 ± 4.79** | **81.93 ± 6.31** | **99.79 ± 0.30** | **89.84 ± 3.82** | **0.9942** |
| ResNet-50 | 87.10 ± 2.29 | 81.37 ± 3.14 | 95.78 ± 2.85 | 87.92 ± 1.97 | 0.9917 |
| MobileNetV3-Large | 85.14 ± 4.18 | 78.44 ± 6.13 | 97.26 ± 1.66 | 86.65 ± 3.11 | 0.9990 |
| InceptionV3 | 78.74 ± 4.36 | 73.13 ± 5.13 | 90.93 ± 7.91 | 80.71 ± 3.67 | 0.8971 |
| EfficientNet-B4 | 73.37 ± 1.58 | 65.09 ± 1.54 | 98.52 ± 1.08 | 78.37 ± 0.94 | 0.9951 |

> **Note on sensitivity:** In cancer screening, sensitivity (the ability to correctly identify true cancer cases) is the most clinically critical metric. MobileNetV3-Small achieved only **~1 false negative per run** across ~158 cancer test cases (miss rate: ~0.21%).

---

## Dataset

The dataset used is composed of two publicly available sources on Kaggle:
- [Oral Cancer Dataset](https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset)
- [Oral Lesions Malignancy Detection Dataset](https://www.kaggle.com/datasets/mohamedgobara/oral-lesions-malignancy-detection-dataset)

| Property | Value |
|---|---|
| Total training images | ~4,000 (~2,000/class) |
| Test set | ~323 images (~158 Cancer / ~165 Non-Cancer) |
| Image source | Real dental clinic environments |
| Variability | Lighting, camera angle, resolution, patient positioning |

The test set consists exclusively of **original, unmodified clinical photographs** from an independent subset — no augmented images were included — providing a clean and unbiased evaluation benchmark. Training and test directories are fully separated at the directory level prior to any training step.

---

## Methodology

The pipeline follows five sequential stages:

```
1. Image Preparation  →  2. Data Augmentation  →  3. Stratified Split
                                  ↓
              4. CNN Training (Transfer Learning from ImageNet)
                                  ↓
              5. Metric Evaluation & Architecture Selection
```

### Data Augmentation (applied on-the-fly during training)

- **Random Horizontal Flip** (p = 0.5) — simulates natural variation in patient positioning
- **Random Rotation** (±10°) — accounts for camera angle differences in clinical photography
- **Resize** — 224×224 for MobileNetV3-Small, MobileNetV3-Large, ResNet-50, and InceptionV3; 380×380 for EfficientNet-B4
- **Normalization** — ImageNet channel-wise mean and standard deviation

> Augmentation was applied **on-the-fly** at every epoch using PyTorch's `torchvision.transforms` pipeline. No augmented images were stored on disk. The test set received **no augmentation**.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 8 |
| Epochs | 10 |
| Loss Function | Cross-Entropy Loss |
| Framework | PyTorch (GPU-accelerated via Google Colab) |
| Pre-training | ImageNet weights (full fine-tuning) |
| Runs per architecture | 3 (seeds: 42, 123, 456) |

---

## Requirements

```bash
pip install torch torchvision
pip install numpy pandas matplotlib scikit-learn
```

---

## Key Takeaways

- **Lightweight ≠ weak.** MobileNetV3-Small outperformed all larger architectures across every metric, including precision, sensitivity, F1-score, and AUC.
- **Sensitivity matters most.** In cancer screening, false negatives are far more dangerous than false positives. MobileNetV3-Small achieved a miss rate of only ~0.21%.
- **Transfer learning works across domains.** Fine-tuning ImageNet pre-trained weights on clinical dental photography consistently yielded strong results across all five architectures.
- **EfficientNet-B4 over-predicts Cancer.** Its low precision (65.09%) indicates a systematic bias toward the positive class, likely due to limited training data relative to its capacity.
- **InceptionV3 showed the highest instability.** With standard deviations up to ±7.91% in sensitivity, its multi-scale inception modules appear less suited to the localized texture patterns of intraoral lesion photographs.
- **ResNet-50 is the most stable baseline.** It achieved competitive accuracy (87.10%) with the lowest variance across runs, making it a reliable choice when consistency matters.

---

## Future Work

- **Multiclass classification** — extending the model to distinguish specific lesion types such as leukoplakia, erythroplakia, fibroma, and oral squamous cell carcinoma (OSCC).
- **Explainability (Grad-CAM)** — highlighting the image regions most influential to each model's decision, enabling clinician trust and auditability.
- **Dataset expansion** — collecting images across diverse patient populations, institutions, and imaging conditions to improve generalization.
- **External validation** — evaluating on independent clinical datasets from distinct institutions to confirm generalization beyond the experimental conditions of this study.
- **Federated Learning** — enabling collaborative training across multiple clinical institutions without sharing sensitive patient data.

---

*This work was developed at the Department of Computer Science, PUC Minas, Belo Horizonte, Brazil.*  
*Supported by the Brazilian Ministry of Science, Technology and Innovations, within the scope of PPI-SOFTEX / Arquitetura Cognitiva (Phase 3).*
