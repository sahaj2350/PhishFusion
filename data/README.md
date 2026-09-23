# PhishFusion Data

This directory is reserved for datasets used by the PhishFusion pipelines.

## Important

The raw and sample datasets are **not included in the public GitHub repository**.

The datasets are excluded because of their size and because they are obtained from third-party sources with individual licensing and redistribution terms.

Datasets should be obtained from their respective original sources and stored locally according to the directory structures required by the PhishFusion pipelines.

## Dataset Documentation

For complete information about:

* required datasets,
* dataset roles,
* expected directory structures,
* metadata, text, and image modalities,
* raw and processed data,
* preprocessing requirements,
* reproducibility, and
* dataset licensing considerations,

see:

**[Dataset Documentation](../docs/datasets.md)**

## Local Data Structure

The expected local organization is broadly:

```text
data/

├── raw/
│   ├── metadata/
│   ├── text/
│   └── image/
│
└── processed/
```

The exact directory structures and filenames required by each pipeline are documented in [`docs/datasets.md`](../docs/datasets.md).

## GitHub Policy

Only documentation is maintained in this directory in the public repository.

The actual datasets under `data/raw/` and generated processed data under `data/processed/` should remain local unless a specific dataset or sample has been verified as legally redistributable and intentionally added to the repository.

Users are responsible for complying with the applicable licensing and usage terms of the datasets they obtain.
