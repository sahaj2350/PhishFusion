# PhishFusion Architecture

## Overview

PhishFusion investigates phishing detection through three complementary modalities:

- **Text**
- **Image**
- **Metadata**

Each modality was developed and evaluated as an independent component in the reported study.

## High-Level Architecture

```text
                           PhishFusion
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
           Text              Image             Metadata
             |                  |                  |
       +-----+------+       +---+---+       +------+------+
       |            |       |       |       |             |
       v            v       v       v       v             v
     BERT       DistilBERT ResNet50 ViT  Random Forest  XGBoost
       |            |       |       |       |             |
       +-----+------+       +---+---+       +------+------+
             |                  |                  |
             v                  v                  v
       Text Ensemble      Image Ensemble     Metadata Results

                         Future Direction
                                |
                                v
                  Unified Text + Image + Metadata
                             Fusion
```

## Text Modality

The text-oriented pipeline combines transformer-based contextual representations with convolutional and recurrent layers.

### BERT-CNN-LSTM

```text
Text
  ↓
BERT
  ↓
CNN
  ↓
LSTM
  ↓
Classification
```

### DistilBERT-CNN-BiLSTM

```text
Text
  ↓
DistilBERT
  ↓
CNN
  ↓
BiLSTM
  ↓
Classification
```

The two model families provide complementary approaches to contextual and sequential text modelling.

The text pipeline also uses a Metadata MLP component based on extracted/scaled metadata information.

### Text Ensemble

The reported system combines model predictions using weighted soft voting. The ensemble is calculated in memory and is not saved as a separate model artifact.

## Image Modality

The image module analyses website screenshots.

```text
Screenshot
    ↓
Resize to 224×224
    ↓
Normalization
    ↓
Augmentation
    ├── Horizontal Flip
    └── Color Jitter
    ↓
 +------------------+
 |                  |
 v                  v
ResNet50           ViT
 |                  |
 +--------+---------+
          ↓
     Soft Voting
       Ensemble
```

ResNet50 focuses on local spatial patterns, while ViT models relationships across image patches and broader webpage layout.

## Metadata Modality

The metadata module operates on manually engineered numerical and categorical information.

```text
Multiple datasets
       ↓
Merge / Clean
       ↓
Deduplicate
       ↓
Missing-value handling
       ↓
Label detection / encoding
       ↓
Categorical encoding
       ↓
Approx. 174 features
       ↓
 +----------------------+
 |                      |
 v                      v
Random Forest         XGBoost
 |                      |
 +----------+-----------+
            ↓
        Validation
```

## Important Project Boundary

Although PhishFusion is motivated by multimodal phishing detection, the final text + image + metadata fusion layer was not implemented in the reported study.

The project therefore should be presented as:

> a research framework with independently developed and evaluated text, image and metadata detection modules, with unified three-modal fusion identified as future work.

This distinction should be preserved throughout the repository.
