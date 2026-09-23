# PhishFusion

## A Hybrid Multimodal Framework for Intelligent Phishing Detection

PhishFusion is a research framework for phishing detection that investigates three complementary modalities:

* **Text**
* **Image**
* **Metadata**

The project develops and evaluates modality-specific deep-learning and machine-learning approaches, together with ensemble-based prediction strategies.

> **Project status:** The text, image, and metadata modules were developed and evaluated independently. The final end-to-end fusion of all three modalities was **not implemented in the reported study** because a suitable aligned dataset containing text, image, and metadata for the same instances was not available.

IMPORTANT:
PhishFusion currently consists of three independently evaluated
modality-specific pipelines. The unified Text + Image + Metadata
fusion architecture is proposed future work and is not part of the
reported implementation.

---

## Research Problem

Modern phishing attacks can combine deceptive textual content, misleading website visuals, and structural or metadata-level indicators. A detector that considers only one information source may miss signals available in another.

PhishFusion investigates how complementary information from **text, visual appearance, and metadata** can be incorporated into a broader phishing-detection framework.

---

## Implemented Components

### Text Modality

The text-oriented pipeline consists of:

* **BERT-CNN-LSTM**
* **DistilBERT-CNN-BiLSTM**
* **Metadata MLP**
* **Weighted soft-voting ensemble**
* Evaluation on unseen text datasets including **Nigerian Fraud** and **Ling**

The two transformer-based branches are:

```text
BERT → CNN → LSTM
```

and

```text
DistilBERT → CNN → BiLSTM
```

The **Metadata MLP** provides an additional learned component within the text-oriented pipeline.

### Image Modality

The image pipeline consists of:

* Website screenshot preprocessing
* **ResNet50-based CNN classifier**
* **Vision Transformer (ViT)**
* **Soft-voting ensemble of ResNet50 and ViT**

### Metadata Modality

The metadata pipeline consists of:

* Dataset merging and cleaning
* Duplicate removal
* Missing-value handling
* Label detection and encoding
* Categorical feature encoding
* Approximately **174 engineered features**
* **Random Forest**
* **XGBoost**
* Cross-dataset evaluation on **CEAS 08** and **UCI** datasets

---

## Architecture

```text
                              PhishFusion
                                  |
                 +----------------+----------------+
                 |                |                |
                 v                v                v
               TEXT             IMAGE           METADATA
                 |                |                |
        +--------+--------+       |          +-----+------+
        |                 |       |          |            |
        v                 v       v          v            v
   BERT Branch      DistilBERT   ResNet50    ViT    Random Forest
        |              Branch       |         |            |
    +---+---+        +---+---+       |         |            |
    |       |        |       |       |         |            |
    v       v        v       v       |         |            |
   CNN     LSTM     CNN    BiLSTM    |         |          XGBoost
    |       |        |       |       |         |            |
    +---+---+        +---+---+       +----+----+            |
        |                 |                |                 |
        v                 v                v                 |
 BERT-CNN-LSTM    DistilBERT-CNN-BiLSTM  Image Ensemble      |
        |                 |                |                 |
        +--------+--------+                |                 |
                 |                         |                 |
                 v                         |                 v
          Metadata MLP                     |          Metadata Results
                 |                         |
                 v                         |
       Weighted Soft Voting               |
                 |                         |
                 v                         |
           Text Ensemble           ResNet50 + ViT
                                           |
                                           v
                                   Soft-Voting Ensemble
                                           |
                                           v
                                     Image Ensemble


                         Future Direction
                                |
                                v
              Unified Text + Image + Metadata Fusion
```

### Text Pipeline

The text modality contains three implemented components:

**BERT-CNN-LSTM branch**

```text
Input Text
    |
   BERT
    |
   CNN
    |
  LSTM
    |
Branch Prediction
```

**DistilBERT-CNN-BiLSTM branch**

```text
Input Text
    |
DistilBERT
    |
   CNN
    |
  BiLSTM
    |
Branch Prediction
```

**Metadata MLP**

```text
Metadata Features
       |
      MLP
       |
MLP Prediction
```

The outputs are combined using the implemented **weighted soft-voting ensemble** to produce the reported text-ensemble result.

### Image Pipeline

The image modality contains two complementary visual models:

```text
Website Screenshot
        |
   +----+----+
   |         |
   v         v
ResNet50     ViT
   |         |
   +----+----+
        |
 Soft Voting
        |
        v
Image Ensemble
```

### Metadata Pipeline

The metadata modality evaluates the following models independently:

```text
              Metadata Features
                     |
              +------+------+
              |             |
              v             v
        Random Forest     XGBoost
              |             |
              +------+------+
                     |
                     v
            Metadata Results
```

### Overall Framework

The three modalities are implemented as **separate experimental pipelines**.

The following unified architecture represents a **future direction only**:

```text
Text Ensemble
      \
       \
        +----> Unified Multimodal Fusion
       /
Image Ensemble
      /
Metadata Results
```

A final three-way fusion model was **not implemented in the reported study** due to the lack of a suitable aligned dataset containing text, image, and metadata for the same instances.

---

## Reported Results

### Text Ensemble

| Metric    | Reported Value |
| --------- | -------------: |
| Accuracy  |      0.9992968 |
| Precision |      0.9989748 |
| Recall    |      0.9996122 |
| F1-score  |      0.9992934 |

### Image Ensemble

| Metric    | Reported Value |
| --------- | -------------: |
| Precision |         0.9891 |
| Recall    |         0.9181 |
| F1-score  |         0.9523 |

### Metadata Validation

| Model         | Accuracy | Precision | Recall | F1-score |
| ------------- | -------: | --------: | -----: | -------: |
| Random Forest |     0.92 |      0.93 |   0.92 |     0.92 |
| XGBoost       |     0.91 |      0.91 |   0.91 |     0.91 |

The metadata results changed substantially during cross-dataset evaluation, highlighting sensitivity to differences in feature availability and representation.

> **Note:** Reported results are specific to the datasets, preprocessing procedures, model configurations, and experimental settings used in the study. They should not be interpreted as universal real-world phishing-detection performance.

---

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

---

## Documentation

* [Architecture](docs/architecture.md)
* [Methodology](docs/methodology.md)
* [Datasets](docs/datasets.md)
* [Results](docs/results.md)
* [Limitations and Future Scope](docs/limitations-and-future-scope.md)
* [Literature Review](docs/literature-review.md)
* [Reproducibility](docs/reproducibility.md)
* [References](docs/references.md)

---

## Data

Raw datasets are intentionally **not included** in the public repository.

See [data/README.md](data/README.md) for an overview of the local data directory and [docs/datasets.md](docs/datasets.md) for the detailed dataset requirements, directory structures, and preparation information.

The repository does not redistribute the complete third-party datasets used during experimentation.

Users are responsible for obtaining the required datasets from their respective sources and complying with their applicable licensing and usage terms.

---

## Model Artifacts

Trained model checkpoints and generated preprocessing artifacts are excluded from Git.

See [models/README.md](models/README.md) for the model-artifact inventory and related information.

---

## Installation

The reported environment uses **Python 3.11** together with PyTorch, Torchvision, Hugging Face Transformers, Pandas, NumPy, scikit-learn, XGBoost, NLTK, and PIL.

Install the required packages with:

```bash
pip install -r requirements.txt
```

For a clean environment, a Python 3.11 virtual environment is recommended.

### Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The repository preserves the project's existing pipeline style: one main training pipeline for each modality and separate inference programs for the text and metadata test datasets.

Place the required datasets in the paths described in [data/README.md](data/README.md) and [docs/datasets.md](docs/datasets.md), then run the relevant Python program from:

```text
src/phishfusion/
```

The exact command-line interface should follow the current scripts; this repository does not claim a redesigned training API.

---

## Experimental Workflow

The general workflow of the project is:

```text
Dataset Preparation
        |
        v
Preprocessing
        |
        v
Feature / Representation Generation
        |
        v
Model Training
        |
        v
Ensemble Evaluation
        |
        v
Cross-Dataset / Unseen-Dataset Evaluation
        |
        v
Results Analysis
```

The text, image, and metadata workflows are evaluated independently.

---

## Limitations

The principal limitations reported in the study include:

* Lack of a unified aligned dataset containing all three modalities
* Feature-space mismatch in cross-dataset metadata evaluation
* Dependence on static datasets
* Difficulty with closely duplicated phishing webpages
* Possible difficulty with very short or obscure text

Additional limitations may arise from differences in dataset composition, preprocessing, feature availability, and experimental configuration.

---

## Future Scope

Future work includes:

* Unified multimodal fusion
* Metadata feature standardization and alignment
* Transfer learning
* Improved metadata models
* Real-time phishing detection
* Advanced multimodal architectures
* Concept-drift handling
* Additional phishing modalities
* Appropriate explainability techniques

A major research direction is the development of a unified architecture capable of processing **text, image, and metadata jointly** when aligned multimodal data are available.

---

## Dissertation

The complete dissertation provides the detailed research methodology, literature review, experiments, results, limitations, and future scope for PhishFusion.

---

## Project Status

**Research Prototype**

PhishFusion was developed as a research project and represents the experimental implementation of the reported study.

The current repository should therefore be considered a **research prototype rather than a production-ready phishing protection system**.

---

## Disclaimer

PhishFusion is intended for **research and educational purposes**.

The system is designed to investigate phishing-detection techniques and should not be considered a complete security solution. Detection results should be interpreted within the context of the datasets and experimental conditions used in the study.

---

## Citation

If you use PhishFusion in academic work, please cite the associated research work or dissertation when applicable.

```bibtex
@software{phishfusion,
  title  = {PhishFusion: A Hybrid Multimodal Framework for Intelligent Phishing Detection},
  author = {Sahaj Khurana},
  year   = {2026},
  url    = {https://github.com/sahaj2350/PhishFusion}
}
```
---

## Publication

### Associated Research Publication

**Sahi, S. K., Kalra, V., & Khurana, S. (2026).** *PhishFusion: A Hybrid Multimodal Framework for Intelligent Phishing Detection.* *DMPedia Advances in Science, Technology and Innovation*, 1–24.

**Chapter citation:**

> Chapter 1: PhishFusion: A Hybrid Multimodal Framework for Intelligent Phishing Detection. (2026). *DMPedia Advances in Science, Technology and Innovation*, *1*(SPE), 1–24.

**Publication:** [Read the published chapter](https://digitalmanuscriptpedia.com/book/index.php/DMP-DASTI/article/view/1)

For additional bibliographic information, see [References](docs/references.md).


---

## License

See [LICENSE](LICENSE) for the applicable terms of use.
