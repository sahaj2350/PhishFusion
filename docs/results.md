# Results

## Overview

The reported experiments evaluate the text, image and metadata modalities separately.

The following results should be interpreted as **modality-specific or within-modality ensemble results**, not as the performance of a completed end-to-end three-modal fusion system.

## 1. Text Modality

### BERT Classifier

| Epoch | Loss | Validation F1 |
|---:|---:|---:|
| 1 | 0.1027 | 0.6644 |
| 2 | 0.0824 | 0.9953 |
| 3 | 0.0073 | 0.9990 |

### DistilBERT Classifier

| Epoch | Loss | Validation F1 |
|---:|---:|---:|
| 1 | 0.0102 | 0.9992 |
| 2 | 0.0045 | 0.9993 |
| 3 | 0.0019 | 0.9995 |

### Metadata MLP in Text Pipeline

| Epoch | Loss | Validation F1 |
|---:|---:|---:|
| 1 | 0.1381 | 0.9682 |
| 2 | 0.1304 | 0.9698 |
| 3 | 0.1301 | 0.9686 |

### Hybrid Text Ensemble

| Metric | Value |
|---|---:|
| Accuracy | 0.9992968 |
| Precision | 0.9989748 |
| Recall | 0.9996122 |
| F1-score | 0.9992934 |

### Unseen Text Evaluation

#### Nigerian Fraud

| Metric / Class | Result |
|---|---:|
| Accuracy | 0.98 |
| Phish precision | 1.00 |
| Phish recall | 0.98 |
| Phish F1 | 0.99 |

Reported confusion matrix:

```text
                Predicted
               Ham   Phish
Actual Ham       0      0
Actual Phish    75   3257
```

#### Ling

| Metric / Class | Result |
|---|---:|
| Accuracy | 0.99 |
| Ham F1 | 1.00 |
| Phish precision | 0.96 |
| Phish recall | 1.00 |
| Phish F1 | 0.98 |

Reported confusion matrix:

```text
                Predicted
               Ham   Phish
Actual Ham    2383     18
Actual Phish     1    457
```

## 2. Image Modality

### CNN / ResNet50

Validation F1 by epoch:

| Epoch | Val F1 |
|---:|---:|
| 1 | 0.9169 |
| 2 | 0.9272 |
| 3 | 0.9355 |
| 4 | 0.9318 |
| 5 | 0.9463 |
| 6 | 0.9478 |
| 7 | 0.9409 |
| 8 | 0.9495 |
| 9 | 0.9497 |
| 10 | 0.9494 |
| 11 | 0.9470 |
| 12 | 0.9494 |

### Vision Transformer

| Epoch | Val F1 |
|---:|---:|
| 1 | 0.9126 |
| 2 | 0.9268 |
| 3 | 0.9367 |
| 4 | 0.9399 |
| 5 | 0.9491 |
| 6 | 0.9434 |
| 7 | 0.9476 |
| 8 | 0.9483 |

### CNN + ViT Ensemble

| Metric | Value |
|---|---:|
| Precision | 0.9891 |
| Recall | 0.9181 |
| F1-score | 0.9523 |

The reported ensemble uses equal 0.5 / 0.5 model weights.

## 3. Metadata Modality

### Validation

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.92 | 0.93 | 0.92 | 0.92 |
| XGBoost | 0.91 | 0.91 | 0.91 | 0.91 |

### CEAS 08

| Metric | Value |
|---|---:|
| Accuracy | 0.4569 |
| Precision | 0.9882 |
| Recall | 0.0268 |
| F1-score | 0.0522 |

The class-wise report shows that the model produced very high precision for the phishing class but very low recall on this external dataset.

### UCI

| Metric | Value |
|---|---:|
| Accuracy | 0.5569 |
| Precision | 0.5569 |
| Recall | 1.0000 |
| F1-score | 0.7154 |

The external UCI evaluation shows almost complete phishing recall but substantially lower precision.

## 4. Main Observations

### Text

The reported text models converge rapidly and achieve very high validation performance. The hybrid text ensemble also performs strongly on the reported unseen text datasets.

### Image

Both ResNet50 and ViT learn useful webpage-visual representations, with validation F1 values around 0.95. Their combination produces an image-ensemble F1 of approximately 0.95.

### Metadata

Metadata models perform well in the controlled validation setting but are much more sensitive to cross-dataset feature compatibility. Differences between the approximately 174-feature training representation and external datasets with fewer/different features resulted in substantial degradation.

## 5. Interpretation Boundary

These results support the value of investigating complementary modalities, but they do **not** constitute the result of a completed single model that simultaneously consumes text, image and metadata.
