# PhishFusion Architecture

## Overview

PhishFusion investigates phishing detection through three complementary modalities:

* **Text**
* **Image**
* **Metadata**

Each modality was developed and evaluated as an **independent component** in the reported study.

The current implementation therefore consists of three modality-specific pipelines rather than a single end-to-end three-modal model.

---

## High-Level Architecture

```text
                              PhishFusion
                                  |
              +-------------------+-------------------+
              |                   |                   |
              v                   v                   v
            TEXT                IMAGE              METADATA
              |                   |                   |
       +------+-------+       +---+---+         +-----+-----+
       |              |       |       |         |           |
       v              v       v       v         v           v
 BERT-CNN-LSTM   DistilBERT- ResNet50   ViT   Random Forest XGBoost
                 CNN-BiLSTM
       |              |       |       |         |           |
       |              |       +---+---+         +-----+-----+
       |              |           |                   |
       +------+-------+           v                   v
              |             Image Ensemble      Metadata Results
              v
         Metadata MLP
              |
              v
     Weighted Soft Voting
              |
              v
        Text Ensemble


                     FUTURE RESEARCH DIRECTION
                                |
                                v
                 +-------------------------------+
                 | Unified Multimodal Fusion     |
                 |                               |
                 | Text + Image + Metadata       |
                 +-------------------------------+
```

The unified text + image + metadata fusion layer shown above is a **future research direction** and was not implemented in the reported study.

---

# 1. Text Modality

The text-oriented pipeline combines transformer-based contextual representations with convolutional and recurrent layers.

It contains two primary neural ensemble branches and a Metadata MLP component.

---

## 1.1 BERT-CNN-LSTM

The first text branch uses BERT followed by convolutional and recurrent layers:

```text
Text Input
    |
    v
  BERT
    |
    v
   CNN
    |
    v
  LSTM
    |
    v
Classification
```

### Components

**BERT**

Produces contextualized representations of the input text.

**CNN**

Extracts local patterns and higher-level n-gram-like representations from the contextualized embeddings.

**LSTM**

Models sequential dependencies in the resulting representation.

---

## 1.2 DistilBERT-CNN-BiLSTM

The second text branch uses DistilBERT together with CNN and bidirectional LSTM layers:

```text
Text Input
    |
    v
DistilBERT
    |
    v
   CNN
    |
    v
 BiLSTM
    |
    v
Classification
```

### Components

**DistilBERT**

Provides contextualized text representations using a lighter transformer architecture.

**CNN**

Extracts local feature patterns from the transformer representation.

**BiLSTM**

Models sequential dependencies in both forward and backward directions.

---

## 1.3 Metadata MLP

The text-oriented pipeline also incorporates a Metadata MLP based on extracted and scaled metadata information:

```text
Extracted Metadata
        |
        v
 Feature Processing
        |
        v
       MLP
        |
        v
Prediction
```

The MLP provides an additional learned prediction component alongside the two transformer-based branches.

---

## 1.4 Text Ensemble

The reported text pipeline combines the predictions from its implemented components using **weighted soft voting**.

Conceptually:

```text
       BERT-CNN-LSTM
              |
              v
         Prediction
              |
              |
     DistilBERT-CNN-BiLSTM
              |
              v
         Prediction
              |
              |
         Metadata MLP
              |
              v
         Prediction
              |
              v
      Weighted Soft Voting
              |
              v
        Text Ensemble
```

The ensemble prediction is calculated **in memory** during inference/evaluation and is not saved as a separate model artifact.

---

# 2. Image Modality

The image module analyses screenshots of websites for visual phishing detection.

The preprocessing workflow consists of resizing, normalization, and augmentation before model inference.

```text
Website Screenshot
        |
        v
 Resize to 224 × 224
        |
        v
 Normalization
        |
        v
   Augmentation
        |
        +----------------------+
        |                      |
        v                      v
  Horizontal Flip          Color Jitter
        |                      |
        +----------+-----------+
                   |
                   v
          +--------+--------+
          |                 |
          v                 v
       ResNet50            ViT
          |                 |
          |                 |
          +--------+--------+
                   |
                   v
              Soft Voting
                   |
                   v
             Image Ensemble
```

## 2.1 ResNet50

ResNet50 is used as the CNN-based visual classifier.

It provides hierarchical visual representations and captures spatial patterns within webpage screenshots.

## 2.2 Vision Transformer

A Vision Transformer (ViT) is used as the second visual classifier.

Unlike the convolutional architecture, ViT processes image patches and models relationships between them to capture broader visual and layout information.

## 2.3 Image Ensemble

The predictions from the ResNet50 and ViT models are combined using **soft voting**:

```text
ResNet50 Prediction
        |
        +----------+
                   |
                   v
               Soft Voting
                   ^
                   |
        +----------+
        |
ViT Prediction
        |
        v
 Image Ensemble
```

The resulting ensemble provides the reported image-modality prediction.

---

# 3. Metadata Modality

The metadata module operates on manually engineered numerical and categorical information derived from multiple datasets.

The preprocessing workflow is:

```text
Multiple Datasets
        |
        v
   Merge / Clean
        |
        v
    Deduplicate
        |
        v
Missing-Value Handling
        |
        v
Label Detection / Encoding
        |
        v
 Categorical Encoding
        |
        v
Approx. 174 Features
        |
        +-------------------+
        |                   |
        v                   v
 Random Forest          XGBoost
        |                   |
        +---------+---------+
                  |
                  v
             Evaluation
```

## 3.1 Dataset Processing

The metadata workflow performs:

* Dataset merging
* Data cleaning
* Duplicate removal
* Missing-value handling
* Label detection and encoding
* Categorical feature encoding
* Feature preparation

The resulting representation contains approximately **174 engineered features**.

## 3.2 Random Forest

Random Forest is used as one of the metadata classifiers.

It provides an ensemble of decision trees operating on the engineered feature representation.

## 3.3 XGBoost

XGBoost provides the second tree-based metadata classifier.

It is evaluated independently against the Random Forest model.

## 3.4 Metadata Evaluation

The metadata models are evaluated using the relevant validation and cross-dataset evaluation procedures.

The reported study includes cross-dataset evaluation involving **CEAS 08** and **UCI** datasets.

---

# 4. Relationship Between the Modalities

The three modalities provide complementary information:

```text
                  PhishFusion
                      |
       +--------------+--------------+
       |              |              |
       v              v              v
     TEXT           IMAGE         METADATA
       |              |              |
       v              v              v
Textual          Visual          Structural /
Signals          Signals         Metadata Signals
       |              |              |
       v              v              v
Text Ensemble   Image Ensemble   Metadata Models
```

The modalities are **not fused into a single prediction model** in the reported implementation.

Instead, each modality produces its own experimental result.

---

# 5. Implemented vs. Future Architecture

It is important to distinguish the architecture that was actually implemented from the broader multimodal architecture motivating the project.

### Implemented

```text
Text
 └── BERT-CNN-LSTM
 └── DistilBERT-CNN-BiLSTM
 └── Auxiliary Metadata MLP
 └── Weighted soft-voting Text Ensemble

Image
 └── ResNet50
 └── ViT
 └── Soft-voting Image Ensemble

Metadata
 └── Random Forest
 └── XGBoost
```

### Future Direction

```text
Text Ensemble
       \
        \
         +--------------------------+
        /                           |
       /                            v
Image Ensemble          Unified Multimodal Fusion
       \                            ^
        \                           |
         +---------------------------+
                    ^
                    |
             Metadata Models
```

A unified model combining text, image, and metadata was not implemented in the reported study.

---

# 6. Important Project Boundary

Although PhishFusion is motivated by multimodal phishing detection, the final **Text + Image + Metadata** fusion layer was not implemented in the reported study.

The principal reason was the lack of a suitable **aligned dataset containing text, image, and metadata for the same instances**.

Accordingly, the project should be presented as:

> **A research framework with independently developed and evaluated text, image, and metadata phishing-detection modules, with unified three-modal fusion identified as future work.**

This distinction should be preserved throughout the repository, including the README, methodology, results, and future-scope documentation.

---

# 7. Summary

The current PhishFusion architecture can therefore be summarized as:

```text
TEXT
 ├── BERT → CNN → LSTM
 ├── DistilBERT → CNN → BiLSTM
 ├── Auxiliary Metadata MLP
 └── Weighted Soft Voting
             |
             v
       Text Ensemble


IMAGE
 ├── ResNet50
 ├── ViT
 └── Soft Voting
             |
             v
       Image Ensemble


METADATA
 ├── Random Forest
 └── XGBoost
             |
             v
      Metadata Results
```

These three independently developed pipelines form the current PhishFusion research framework.

Unified fusion of their outputs remains a future research direction.
