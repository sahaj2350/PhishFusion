# Limitations and Future Scope

## Limitations

### 1. No Completed Three-Modal Fusion

The final end-to-end fusion of text, image and metadata was not implemented in the reported study.

The primary reason identified was the absence of a suitable curated dataset in which text, image and metadata are aligned for the same instances.

### 2. Metadata Feature-Space Mismatch

The metadata module was trained using approximately 174 engineered features, while some external test datasets contained fewer or differently defined attributes.

Missing attributes were aligned using zero values, which introduced sparsity and information loss and substantially affected cross-dataset performance.

### 3. Static Datasets

The experiments rely on static and historical datasets. Such data cannot fully represent the dynamic evolution of phishing attacks.

### 4. Image-Modality Limitations

Very closely duplicated or visually similar phishing websites may remain difficult to distinguish from genuine websites when visual differences are minimal.

### 5. Text-Modality Limitations

Very short, obscure or otherwise limited text content may provide insufficient contextual information for reliable detection.

### 6. Dataset Standardization

The work highlights the broader difficulty of maintaining consistent feature definitions and representations across cybersecurity datasets.

## Future Scope

### Unified Multimodal Fusion

Develop a fully integrated text + image + metadata architecture using a curated dataset in which the three modalities are aligned for each instance.

### Metadata Improvements

- Feature standardization and alignment
- Transfer learning across datasets
- Deep-learning-based metadata models

### Real-Time Detection

Investigate real-time phishing detection capabilities for practical security systems.

### Advanced Architectures

Explore newer transformer-based and multimodal architectures.

### Concept Drift

Investigate continual learning and mechanisms for handling the evolution of phishing techniques.

### Additional Modalities

Extend the framework to:

- SMS phishing,
- voice/vishing,
- network-based behavioural signals.

### External / Proprietary Data

Collaboration with organizations and access to proprietary datasets could improve realism, scalability and evaluation under real-world conditions.

### Explainability

Appropriate explainable-AI methods can be investigated in future work to improve the interpretability and trust of predictions across the different modalities.

> Explainability is a **future direction** in this repository. SHAP, LIME and related methods are not presented as implemented components of the reported PhishFusion experimental system.
