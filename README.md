# ECE 285 Homework 2

Two unconditional DCGANs generate 64 × 64 RGB paintings from a subset of
Kaggle WikiArt (`steubk/wikiart`, version 1). The subset consists of
Abstract Expressionism, Color Field Painting, and Minimalism. Both models
train on the same split. Their upsampling blocks use either transposed
convolution or nearest-neighbor resizing followed by convolution.

## Reproduce the experiment

From this directory, run the following in a Python environment with PyTorch.

```bash
python -m pip install -r requirements.txt
python fetch_wikiart.py --cache cache --out data
python gan.py --data data --out reproduced/transpose --upsampling transpose
python gan.py --data data --out reproduced/resize --upsampling resize
python evaluate.py --data data --run reproduced/transpose --device cpu
python evaluate.py --data data --run reproduced/resize --device cpu
```

The data preparation command fetches the selected WikiArt v1 directories
from the original Kaggle ZIP and writes 64 × 64 RGB images under `data/`.
It uses `curl` and about 2.7 GB for cached ranges and their assembled copies.
The first evaluation may download the ImageNet Inception-v3 weights.

Training selects CUDA, Apple MPS, or CPU based on availability. The reported
runs used Apple MPS. Use a new output directory for each run. Defaults are
16 epochs, batch size 64, learning rate 0.0002, and seed 42 for the
split and training. Results can vary across devices and library versions.

## Experiment files

- `gan.py` contains the two generators, shared discriminator architecture,
  train/holdout split, and training loop.
- `evaluate.py` uses Ignite FID and Inception Score on `initial.pt` and
  `last.pt`. It evaluates the two checkpoints on matching latent draws.
- Each generated run directory contains the checkpoints, sample grids,
  `history.csv`, `metrics.json`, and the exact `split.json` manifest.
