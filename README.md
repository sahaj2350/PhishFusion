# PhishFusion

## A Hybrid Multimodal Framework for Intelligent Phishing Detection

PhishFusion is a research framework for phishing detection that investigates three complementary modalities:

- **Text**
- **Image**
- **Metadata**

The project develops and evaluates modality-specific deep-learning and machine-learning approaches, together with ensemble-based prediction strategies.

> **Project status:** The text, image, and metadata modules were developed and evaluated independently. The final end-to-end fusion of all three modalities was **not implemented in the reported study** because a suitable aligned dataset containing text, image, and metadata for the same instances was not available.

## Research Problem

Modern phishing attacks can combine deceptive textual content, misleading website visuals, and structural or metadata-level indicators. A detector that considers only one information source may miss signals available in another.

PhishFusion investigates how complementary information from text, visual appearance, and metadata can be incorporated into a broader phishing-detection framework.

## Implemented Components

### Text Modality

- BERT + CNN + LSTM
- DistilBERT + CNN + BiLSTM
- Metadata MLP within the text-oriented pipeline
- Weighted soft-voting ensemble
- Evaluation on unseen text datasets including Nigerian Fraud and Ling

### Image Modality

- Website screenshot preprocessing
- ResNet50-based CNN classifier
- Vision Transformer (ViT)
- Soft-voting ensemble of CNN and ViT

### Metadata Modality

- Dataset merging and cleaning
- Duplicate removal
- Missing-value handling
- Label detection and encoding
- Categorical feature encoding
- Approximately 174 engineered features
- Random Forest
- XGBoost
- Cross-dataset evaluation on CEAS 08 and UCI datasets

## Architecture

```text
                         PhishFusion
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
           TEXT             IMAGE           METADATA
             |                |                |
     +-------+-------+     +--+--+       +-----+------+
     |               |     |     |       |            |
     v               v     v     v       v            v
   BERT         DistilBERT ResNet50   ViT      Random Forest  XGBoost
     |               |     |     |       |            |
     +-------+-------+     +--+--+       +-----+------+
             |                |                |
             v                v                v
       Text ensemble     Image ensemble    Metadata results

                    Future direction:
             Unified text + image + metadata fusion
```

The final three-way fusion shown above is a **future direction**, not a completed experimental component.

## Reported Results

### Text Ensemble

| Metric | Reported value |
|---|---:|
| Accuracy | 0.9992968 |
| Precision | 0.9989748 |
| Recall | 0.9996122 |
| F1-score | 0.9992934 |

### Image Ensemble

| Metric | Reported value |
|---|---:|
| Precision | 0.9891 |
| Recall | 0.9181 |
| F1-score | 0.9523 |

### Metadata Validation

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.92 | 0.93 | 0.92 | 0.92 |
| XGBoost | 0.91 | 0.91 | 0.91 | 0.91 |

The metadata results changed substantially during cross-dataset evaluation, highlighting sensitivity to differences in feature availability and representation.

## Repository Structure

```text
PhishFusion/
├── README.md
├── PROJECT_STATUS.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── configs/
│   ├── text.yaml
│   ├── metadata.yaml
│   └── image.yaml
│
├── src/
│   └── phishfusion/
│       ├── text/
│       ├── metadata/
│       └── image/
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── results/
│   ├── figures/
│   └── tables/
│
└── docs/
    ├── architecture.md
    ├── methodology.md
    ├── datasets.md
    ├── results.md
    ├── limitations-and-future-scope.md
    ├── literature-review.md
    ├── reproducibility.md
    └── references.md
```

## Documentation

- [Architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Datasets](docs/datasets.md)
- [Results](docs/results.md)
- [Limitations and Future Scope](docs/limitations-and-future-scope.md)
- [Literature Review](docs/literature-review.md)
- [Reproducibility](docs/reproducibility.md)
- [References](docs/references.md)

## Data

Raw datasets are intentionally not included in the public repository. See [data/README.md](data/README.md) for the expected local directory structure and dataset requirements.

## Model Artifacts

Trained checkpoints and generated preprocessing artifacts are excluded from Git. See [models/README.md](models/README.md) for the artifact inventory.

## Installation

The reported environment uses Python 3.11 together with PyTorch, Torchvision, Hugging Face Transformers, Pandas, NumPy, scikit-learn, XGBoost, NLTK and PIL.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the Project

The repository preserves the project's existing pipeline style: one main training pipeline for each modality and separate inference programs for the text and metadata test datasets.

Place the required datasets in the paths described in `data/README.md`, then run the relevant Python program from `src/phishfusion/`.

The exact command-line interface should follow the current scripts; this repository does not claim a redesigned training API.

## Limitations

The principal limitations reported in the study include:

- lack of a unified aligned dataset containing all three modalities,
- feature-space mismatch in cross-dataset metadata evaluation,
- dependence on static datasets,
- difficulty with closely duplicated phishing webpages,
- possible difficulty with very short or obscure text.

## Future Scope

Future work includes unified multimodal fusion, metadata feature standardization and alignment, transfer learning, improved metadata models, real-time detection, advanced multimodal architectures, concept-drift handling, additional phishing modalities, and appropriate explainability techniques.

## Dissertation

The complete dissertation provides the detailed research methodology, literature review, experiments, results, limitations and future scope for PhishFusion.

## License

See [LICENSE](LICENSE).
