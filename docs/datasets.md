# Datasets

## Important

The raw datasets are **not included in the public GitHub repository**.

They should be obtained from their respective sources and placed locally using the structures below.

Users are responsible for complying with the licensing and redistribution terms of each dataset.

---

## 1. Metadata Datasets

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

### Metadata dataset roles

| Dataset | Role |
|---|---|
| Phishing Dataset Master | URL / structural phishing features |
| PhiUSIIL | Structured phishing URL features |
| Phishing Email | Email and metadata features |
| Nigerian Fraud | Fraud/scam email patterns |
| SpamAssasin | Spam/ham and structural email information |
| Ling | Spam and legitimate email information |
| Nazario | Phishing URL/email-related attributes |
| Enron | Legitimate corporate email patterns |

---

## 2. Text Datasets

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

The pipeline additionally loads:

```text
hf://datasets/zefang-liu/phishing-email-dataset/Phishing_Email.csv
```

### Text dataset roles

- SpamAssassin provides legitimate and spam email examples.
- Enron provides legitimate corporate communication.
- Nazario provides phishing-related email/URL examples.
- PhishTank provides phishing URL data.
- The Zefang-Liu phishing email dataset provides modern phishing email examples.
- The supplied auxiliary datasets are used to broaden the evaluation/training distribution.

---

## 3. Image Dataset

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

Accepted image formats:

```text
.png
.jpg
.jpeg
.bmp
.webp
```

The image datasets provide genuine and phishing webpage screenshots for visual classification.

### Dataset-directory rule

Only the directory structures required by the image pipeline should be considered part of the dataset specification. Unrelated files that happen to occur inside screenshot folders should not be turned into repository structure.

---

## 4. Raw vs Processed Data

### Raw data

Original downloaded datasets are stored locally under:

```text
data/raw/
```

### Processed data

Any generated cleaned/transformed datasets may be stored locally under:

```text
data/processed/
```

These should normally remain excluded from Git unless a small, redistributable processed sample is intentionally included.

## 5. GitHub Data Policy

This repository contains:

- dataset descriptions,
- expected directory structures,
- preprocessing documentation.

It does not contain the full raw datasets.
