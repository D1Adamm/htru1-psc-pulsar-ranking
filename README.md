# HTRU1–PSC Pulsar Candidate Ranking

Train a convolutional neural network (CNN) to rank pulsar candidates on the **HTRU1** survey dataset, then evaluate how well that model transfers to **PSC AO327** candidates.

## Project goals

1. **Train** a CNN classifier/ranker on HTRU1 labeled candidates.
2. **Evaluate** in-domain performance on held-out HTRU1 data.
3. **Transfer** the trained model to PSC AO327 candidates and measure cross-survey generalization.

## Repository layout

```
configs/          # Experiment and training configs (YAML)
data/
  raw/            # Original datasets (not tracked)
  processed/      # Preprocessed arrays / splits (not tracked)
notebooks/        # Exploratory analysis and demos
outputs/
  checkpoints/    # Saved model weights (not tracked)
  figures/        # Plots and visualizations
  metrics/        # Evaluation metrics and logs
src/
  data/           # Dataset loading and preprocessing
  models/         # Network architectures (to be added)
tests/            # Unit and integration tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Datasets are not included in this repository. Place raw files under `data/raw/` once they are available.

## Status

Scaffolding only — model code and dataset download scripts are not included yet.
