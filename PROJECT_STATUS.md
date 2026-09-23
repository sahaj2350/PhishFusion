# PhishFusion — Project Status

## Current Status

PhishFusion is a research framework consisting of three separately developed and evaluated phishing-detection modalities:

1. Text
2. Image
3. Metadata

The current repository represents the implementation and experiments reported in the Bachelor's thesis.

## Implemented

### Text

- Data cleaning and deduplication
- Phishing-class text augmentation using WordNet synonyms and character swapping
- Metadata feature extraction from text
- Metadata scaling
- BERT + CNN + LSTM
- DistilBERT + CNN + BiLSTM
- Metadata MLP
- Weighted soft-voting ensemble
- Testing on unseen datasets including Nigerian Fraud, Ling and Spam-related data

### Image

- Screenshot discovery and loading
- Image resizing to 224×224
- Normalization
- Horizontal-flip and color-jitter augmentation
- Class-weighted training
- Early stopping
- ResNet50 CNN classifier
- Vision Transformer classifier
- CNN + ViT soft-voting ensemble

### Metadata

- Multiple-dataset merging
- Cleaning and deduplication
- Missing-value handling
- Label detection/encoding
- Categorical feature encoding
- Approximately 174 engineered features
- Random Forest
- XGBoost
- Validation
- Cross-dataset testing on CEAS 08 and UCI

## Generated Artifacts

### Text

```text
bert_best.pt
distil_best.pt
meta_best.pt
```

### Image

```text
cnn_best.pt
vit_best.pt
```

### Metadata

```text
label_encoder.pkl
feature_encoders.pkl
rf_metadata.pkl
xgb_metadata.pkl
feature_importance.csv
```

The text and image ensembles are calculated in memory and are not currently saved as separate ensemble model files.

## Not Implemented in the Reported Study

- Final end-to-end fusion of text + image + metadata
- Unified aligned multimodal training/evaluation dataset
- Production deployment
- Real-time continuous deployment

## Important Interpretation

The reported scores are modality-specific or ensemble-within-modality results. They should not be described as the accuracy of a completed end-to-end three-modal PhishFusion model.

## Future Work

- Fully integrated multimodal fusion
- Aligned multimodal datasets
- Metadata feature standardization
- Transfer learning
- Deep-learning-based metadata models
- Real-time detection
- Advanced multimodal architectures
- Continual learning and concept-drift handling
- SMS phishing, voice/vishing and network behavioural signals
- Collaboration with organizations/proprietary datasets
- Explainability/XAI

Explainability is listed here only as future work and is not presented as an implemented component of the reported system.
