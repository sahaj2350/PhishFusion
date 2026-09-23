# PhishFusion Data

This directory documents the datasets required by the PhishFusion pipelines.

## Important

The raw datasets are **not included in the public GitHub repository**.

Reasons include dataset size and the fact that the project should not redistribute
third-party datasets without confirming their individual terms and licenses.

Obtain each dataset from its original source and place it in the local directory
structure expected by the corresponding pipeline.

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

These datasets are merged, cleaned, deduplicated, processed for missing values,
and used for categorical encoding before Random Forest and XGBoost training.

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

The pipeline also loads the Zefang-Liu phishing email dataset using:

```text
hf://datasets/zefang-liu/phishing-email-dataset/Phishing_Email.csv
```

The text pipeline performs cleaning and deduplication, phishing-class augmentation,
metadata extraction/scaling, and training of the BERT-CNN-LSTM, DistilBERT-CNN-BiLSTM,
and Metadata MLP components.

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

Supported image formats:

- PNG
- JPG / JPEG
- BMP
- WEBP

Only the directory structures actually required by the image pipeline should be
recreated. Unrelated files that happen to appear inside dataset screenshot
directories should not be treated as project data.

---

## Local vs GitHub

Recommended local structure:

```text
PhishFusion_Workspace/
├── PhishFusion/          # GitHub repository
│   └── data/
│       └── README.md
│
└── data/
    ├── raw/
    │   ├── text/
    │   ├── metadata/
    │   └── image/
    │
    └── processed/
```

The repository tracks the documentation above, not the raw datasets themselves.

## Reproducibility

Before running a pipeline, place the required datasets in the paths documented
above and confirm that the filenames match the paths referenced by the existing
Python code.
