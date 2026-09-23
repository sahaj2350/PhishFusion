# PhishFusion Model and Generated Artifacts

The PhishFusion pipelines generate trained model checkpoints and preprocessing
artifacts. These are **generated files**, not source code.

Large model files and generated artifacts should normally be excluded from Git.

## Generated Artifacts

### Text Modality

```text
artifacts/text/
├── bert_best.pt
├── distil_best.pt
└── meta_best.pt
```

- `bert_best.pt`: checkpoint for the BERT + CNN + LSTM component.
- `distil_best.pt`: checkpoint for the DistilBERT + CNN + BiLSTM component.
- `meta_best.pt`: checkpoint for the Metadata MLP component used by the text pipeline.

The text weighted soft-voting ensemble is calculated in memory and is not saved
as a separate model file.

---

### Image Modality

```text
artifacts/image/
├── cnn_best.pt
└── vit_best.pt
```

- `cnn_best.pt`: ResNet50-based CNN checkpoint.
- `vit_best.pt`: Vision Transformer checkpoint.

The image ensemble is calculated in memory and is not saved as a separate model file.

---

### Metadata Modality

```text
artifacts/metadata/
├── label_encoder.pkl
├── feature_encoders.pkl
├── rf_metadata.pkl
├── xgb_metadata.pkl
└── feature_importance.csv
```

#### Preprocessing artifacts

```text
label_encoder.pkl
feature_encoders.pkl
```

These preserve the label and categorical-feature encoding information required
by the metadata pipeline.

#### Trained model artifacts

```text
rf_metadata.pkl
xgb_metadata.pkl
```

These contain the trained Random Forest and XGBoost metadata models.

#### Analysis output

```text
feature_importance.csv
```

This is an analysis/result artifact containing feature-importance information.
It is not a trained model.

---

## Why these files are excluded from Git

The repository should contain source code and documentation while trained
weights/checkpoints and generated binary artifacts are kept outside the normal
Git history.

This keeps the repository lightweight and reproducible.

The `.gitignore` should exclude:

```text
*.pt
*.pth
*.ckpt
*.safetensors
*.pkl
*.joblib
```

as well as training checkpoints and local artifacts.

## Reproducing the Artifacts

A user should:

1. Obtain the required datasets.
2. Place them in the documented local paths.
3. Run the corresponding training pipeline.
4. The pipeline will generate the model/preprocessing artifacts locally.

The exact training behavior remains defined by the existing Python scripts.
