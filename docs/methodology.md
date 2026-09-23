# Methodology

## 1. Research Approach

PhishFusion investigates phishing detection using multiple information sources: textual content, webpage visual appearance, and structural/metadata features.

The project follows a modality-specific development strategy. Each branch is trained and evaluated independently before considering future unified fusion.

## 2. Text Modality

### Input Data

The text pipeline uses:

```text
spam.csv
Nazario_5 - Copy.csv
emails.csv
phishtank.csv
SpamAssassin/
```

It also loads the Zefang-Liu phishing-email dataset through:

```text
hf://datasets/zefang-liu/phishing-email-dataset/Phishing_Email.csv
```

### Preprocessing

The pipeline:

1. cleans the collected text/data,
2. removes duplicates,
3. augments phishing-class text using WordNet synonyms and character swapping,
4. extracts metadata features from text,
5. scales the metadata features.

### BERT-CNN-LSTM

BERT provides contextual text representations. CNN layers are used to capture local patterns, and LSTM layers model sequential dependencies.

### DistilBERT-CNN-BiLSTM

DistilBERT provides a lighter transformer representation. CNN layers capture local patterns, while BiLSTM models bidirectional sequence dependencies.

### Metadata MLP in the Text Pipeline

The text pipeline also extracts metadata-style numerical features from the text data and scales them before training an MLP component.

### Ensemble

Predictions from the text components are combined using weighted soft voting.

The ensemble is calculated in memory rather than being saved as an independent checkpoint.

## 3. Image Modality

### Input

The image pipeline uses webpage screenshots from the supplied dataset directory structure.

Supported image formats are:

```text
.png
.jpg
.jpeg
.bmp
.webp
```

### Preprocessing

Images are:

- resized to 224×224,
- normalized,
- augmented using horizontal flipping and color jittering.

Class-weighted loss and early stopping are used during training.

### ResNet50 CNN

The ResNet50-based CNN is used to learn local visual characteristics such as:

- edges,
- textures,
- interface components,
- localized structural irregularities.

### Vision Transformer

The Vision Transformer divides images into patches and uses attention to model broader relationships across the webpage image, including global layout and visual structure.

### Image Ensemble

The reported image ensemble uses soft voting. The two model predictions are combined with equal weights of 0.5 and 0.5 in the reported configuration.

The ensemble is calculated in memory and is not saved as a separate model artifact.

## 4. Metadata Modality

### Input

The metadata training pipeline combines data from several URL, email and structured phishing datasets.

### Data Preparation

The pipeline:

1. merges datasets,
2. cleans the combined data,
3. removes duplicates,
4. handles missing values,
5. detects/encodes labels,
6. label-encodes categorical features.

### Feature Engineering

Approximately 174 engineered features are used.

Feature groups include:

#### Lexical

- character distributions,
- digit counts,
- token patterns.

#### Structural

- URL length,
- link redirects,
- special characters,
- header-based attributes.

#### Domain

- domain type,
- IP-address presence,
- domain-related tokens.

#### Statistical / Empirical

- entropy,
- frequency-based indicators.

### Models

Two classical ensemble classifiers are used:

- Random Forest
- XGBoost

The models are validated on the consolidated feature space and subsequently evaluated on external datasets.

## 5. Cross-Dataset Evaluation

The metadata component is evaluated on CEAS 08 and UCI data.

The reported performance demonstrates that metadata models can perform well when the feature space is consistent with training data, but performance can degrade substantially when feature definitions and availability differ.

## 6. Training Configuration

Reported learning rates:

| Component | Learning rate |
|---|---:|
| Text models | 2e-5 |
| ResNet50 CNN | 1e-4 |
| ViT | 3e-5 |
| Metadata models | 1e-3 |

Reported batch-size range:

```text
8–32
```

The exact model-specific training behavior remains defined by the existing Python scripts.
