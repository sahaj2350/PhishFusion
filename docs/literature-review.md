# Literature Review

## Review Scope

The literature review examined research on phishing detection using machine learning, deep learning, transformers, ensemble learning, multimodal methods, robustness and interpretability.

The dissertation reports a review of **54 research papers**.

## Search Sources

The reported search included:

- IEEE Xplore
- ScienceDirect / Elsevier
- SpringerLink
- ACM Digital Library
- MDPI
- Frontiers Media
- Tech Science Press
- ResearchGate

## Search Strings

The reported search strings included:

```text
phishing detection
phishing and deep learning
phishing detection and multimodal
phishing and ensemble and deep learning
Phishing activity trends report
(phishing and multimodal) and ensemble
ensemble and phishing detection
```

## Major Themes

### Traditional Machine Learning

Research commonly uses manually engineered URL, lexical, domain and structural features such as URL length, tokens, entropy, special characters, IP-address presence and WHOIS-related information.

Ensemble methods including Random Forest, boosting and stacking are frequently investigated for phishing classification.

### Deep Learning

CNNs and recurrent networks are used to learn patterns directly from text, URLs and related representations, reducing dependence on manual feature engineering.

### Transformers

BERT and related transformer architectures are used to capture contextual and semantic relationships in phishing emails and URLs.

DistilBERT and other lighter transformer variants are investigated where computational efficiency matters.

### Ensemble Learning

The reviewed work frequently combines several learners to improve predictive performance over individual models.

### Multimodal Detection

Multimodal research combines textual, visual and structural information because phishing attacks may manipulate several information sources simultaneously.

The literature also identifies practical challenges around computational cost, dataset standardization and fusion design.

### Robustness and Generalization

A recurring issue in the literature is the gap between strong benchmark performance and stable performance under distribution changes, adversarial manipulation or cross-dataset testing.

## Research Gap Identified for PhishFusion

The review motivated a framework that investigates:

- semantic text information,
- visual webpage information,
- structural metadata,
- ensemble-based prediction,
- evaluation beyond a single dataset.

The study also identifies the need for better multimodal datasets and more robust cross-domain evaluation.

## Representative Papers

The full reference list is provided in [references.md](references.md).

| Area | Representative work |
|---|---|
| URL / classical ML | Ahammad et al. (2022) |
| Deep learning | Altwaijry et al. (2024) |
| Transformer-based URL analysis | Asiri et al. (2024) |
| Multimodal text + structural analysis | Asliyuksek et al. (2025) |
| Ensemble ML | Basit et al. (2020) |
| Multimodal phishing detection | Murhej & Nallasivan (2025) |
| Multimodal review | Wangchuk & Gonsalves (2025) |
| CNN + Random Forest | Yang et al. (2021) |
| BERT / transformer analysis | Patra et al. (2024), Elsadig et al. (2022) |
| Hybrid phishing detection | van Geest et al. (2024) |

## Role of the Literature Review

The literature review served as the basis for selecting the modalities, model families, ensemble approach and evaluation directions used in the PhishFusion study.
