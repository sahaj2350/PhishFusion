# PhishFusion Datasets

This document describes the datasets required by the PhishFusion pipelines, their expected local directory structures, processing roles, and reproducibility requirements.

## Important

The raw and sample datasets are **not included in the public GitHub repository**.

This is primarily due to dataset size and the fact that PhishFusion uses third-party datasets whose individual licensing and redistribution terms must be respected.

Obtain each dataset from its original source and place it in the local directory structure expected by the corresponding pipeline.

Users are responsible for complying with the applicable licensing, usage, and redistribution terms of each dataset.

---

## 1. Metadata Modality

The metadata training pipeline expects:

```text
data/raw/metadata/

├── Phishing-Dataset-master/
│   ├── dataset_full.csv
│   └── dataset_small.csv
│
├── phiusiil+phishing+url+dataset/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
│
└── archive/
    ├── phishing_email.csv
    ├── Nigerian_Fraud.csv
    ├── SpamAssasin.csv
    ├── Ling.csv
    ├── Nazario.csv
    └── Enron.csv
```

These datasets are merged, cleaned, deduplicated, processed for missing values, and used for categorical encoding before Random Forest and XGBoost training.

### Metadata dataset roles

| Dataset                 | Role                                      |
| ----------------------- | ----------------------------------------- |
| Phishing Dataset Master | URL / structural phishing features        |
| PhiUSIIL                | Structured phishing URL features          |
| Phishing Email          | Email and metadata features               |
| Nigerian Fraud          | Fraud/scam email patterns                 |
| SpamAssasin             | Spam/ham and structural email information |
| Ling                    | Spam and legitimate email information     |
| Nazario                 | Phishing URL/email-related attributes     |
| Enron                   | Legitimate corporate email patterns       |

---

## 2. Text Modality

The text pipeline expects:

```text
data/raw/text/

├── spam.csv
├── Nazario_5 - Copy.csv
├── emails.csv
├── phishtank.csv
│
└── SpamAssassin/
    ├── easy_ham/
    ├── easy_ham_2/
    ├── hard_ham/
    ├── spam/
    └── spam_2/
```

The pipeline also uses the Enron dataset stored under the metadata directory:

```text
data/raw/metadata/archive/Enron.csv
```

A separate copy of the Enron dataset under `data/raw/text/` is therefore not required.

The pipeline additionally loads the Zefang-Liu phishing email dataset using:

```text
hf://datasets/zefang-liu/phishing-email-dataset/Phishing_Email.csv
```

The text pipeline performs cleaning and deduplication, phishing-class augmentation, metadata extraction/scaling, and training of the BERT-CNN-LSTM, DistilBERT-CNN-BiLSTM, and Metadata MLP components.

### Text dataset roles

* **SpamAssassin** provides legitimate and spam email examples.
* **Enron** provides legitimate corporate communication and is also used by the text pipeline.
* **Nazario** provides phishing-related email/URL examples.
* **PhishTank** provides phishing URL data.
* **Zefang-Liu Phishing Email Dataset** provides phishing email examples.
* The auxiliary datasets are used to broaden the training and evaluation distribution.

---

## 3. Image Modality

The image pipeline expects:

```text
data/raw/image/

└── Images/
    ├── genuine_site_0/
    ├── phishing_site_1/
    ├── phishing/
    │   └── .../screenshots/
    ├── not-phishing/
    │   └── .../screenshots/
    └── phishIRIS_DL_Dataset/
        ├── train/
        │   └── <brand>/
        └── val/
            └── <brand>/
```

### Supported image formats

* PNG
* JPG / JPEG
* BMP
* WEBP

The image pipeline uses webpage screenshots for visual classification of genuine and phishing websites.

### Dataset-directory rule

Only the directory structures actually required by the image pipeline should be recreated.

Unrelated files that happen to appear inside dataset screenshot directories should not be treated as part of the required PhishFusion dataset structure.

---

## 4. Raw vs. Processed Data

### Raw data

Original downloaded datasets are stored locally under:

```text
data/raw/
```

with modality-specific subdirectories:

```text
data/raw/
├── metadata/
├── text/
└── image/
```

### Processed data

Generated cleaned, transformed, or intermediate datasets may be stored locally under:

```text
data/processed/
```

Processed datasets should normally remain excluded from Git unless a small, redistributable processed sample is intentionally included.

---

## 5. Local vs. GitHub

The recommended workspace arrangement is:

```text
PhishFusion_Workspace/

├── PhishFusion/              # GitHub repository
│   ├── data/
│   │   └── README.md
│   ├── docs/
│   │   └── datasets.md
│   └── ...
│
└── data/
    ├── raw/
    │   ├── text/
    │   ├── metadata/
    │   └── image/
    │
    └── processed/
```

The public repository contains dataset documentation and expected directory structures, but not the full raw datasets.

The actual datasets should remain in the local workspace and should not be committed to Git unless a dataset or sample has been specifically verified to be redistributable.

---

## 6. Reproducibility

Before running a PhishFusion pipeline:

1. Obtain the required datasets from their respective original sources.
2. Review and comply with the applicable dataset licenses and terms.
3. Place the datasets in the directory structures documented above.
4. Confirm that filenames match the paths referenced by the existing Python code.
5. Confirm that the required preprocessing inputs are available.
6. Run the corresponding preprocessing and training pipeline.

Dataset versions, preprocessing procedures, library versions, and other experimental conditions can affect the resulting model performance.

---

## 7. Dataset Licensing and Redistribution

PhishFusion does not redistribute the complete third-party datasets used during experimentation.

Dataset users should obtain the datasets from their respective sources and independently review their licensing and usage conditions.

The inclusion of a dataset name or expected directory structure in this documentation does not imply ownership of, or redistribution rights to, that dataset.
