# Reproducibility

## Scope

This document records the environment, inputs, preprocessing assumptions and reported training configuration for reproducing the PhishFusion experiments.

The existing Python scripts remain the authoritative implementation. The YAML files in `configs/` document the configuration but do not automatically control the current scripts unless YAML parsing is explicitly added.

## Software Environment

The dissertation reports:

- Python 3.11
- Pandas
- NumPy
- scikit-learn
- XGBoost
- PyTorch
- Torchvision
- Hugging Face Transformers
- NLTK
- PIL / Pillow
- CUDA-enabled GPU environment

## Hardware

The reported environment uses:

- NVIDIA RTX 3050 or higher
- 16 GB RAM

## Training Configuration

| Modality / component | Learning rate |
|---|---:|
| Text models | 2e-5 |
| ResNet50 CNN | 1e-4 |
| ViT | 3e-5 |
| Metadata models | 1e-3 |

Reported batch-size range:

```text
8–32
```

Actual values used for an individual run remain those present in the corresponding Python script.

## Text Reproduction

Expected inputs:

```text
spam.csv
Nazario_5 - Copy.csv
emails.csv
phishtank.csv
SpamAssassin/
```

The pipeline also loads:

```text
hf://datasets/zefang-liu/phishing-email-dataset/Phishing_Email.csv
```

Processing includes cleaning, deduplication, phishing-class augmentation using WordNet synonyms and character swapping, metadata feature extraction and scaling.

Generated checkpoints:

```text
bert_best.pt
distil_best.pt
meta_best.pt
```

The weighted ensemble is calculated in memory.

## Metadata Reproduction

Expected inputs:

```text
Phishing-Dataset-master/dataset_full.csv
Phishing-Dataset-master/dataset_small.csv
phiusiil+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv
archive/phishing_email.csv
archive/Nigerian_Fraud.csv
archive/SpamAssasin.csv
archive/Ling.csv
archive/Nazario.csv
archive/Enron.csv
```

Processing includes:

- merging,
- cleaning,
- duplicate removal,
- missing-value handling,
- label detection/encoding,
- categorical feature encoding.

Generated artifacts:

```text
label_encoder.pkl
feature_encoders.pkl
rf_metadata.pkl
xgb_metadata.pkl
feature_importance.csv
```

## Image Reproduction

Expected root:

```text
Images/
```

with the required dataset subdirectories documented in [datasets.md](datasets.md).

Processing includes:

- resize to 224×224,
- normalization,
- horizontal flip,
- color jitter,
- class-weighted loss,
- early stopping.

Generated checkpoints:

```text
cnn_best.pt
vit_best.pt
```

The image ensemble is calculated in memory.

## Evaluation

Reported evaluation includes:

- text validation and unseen datasets,
- image validation and ensemble evaluation,
- metadata validation,
- metadata cross-dataset evaluation on CEAS 08 and UCI.

## Reproducibility Limitations

Exact reproduction of the reported metrics depends on:

- obtaining the same datasets,
- preserving the expected dataset structure,
- maintaining the same feature definitions,
- using the same preprocessing,
- using the same training code and parameters.

Cross-dataset metadata performance is particularly sensitive to feature availability and representation.

## Public Repository vs Local Research Workspace

The public GitHub repository should contain source code, documentation, configuration descriptions and selected final results.

Large datasets and generated model artifacts should remain outside the repository.
