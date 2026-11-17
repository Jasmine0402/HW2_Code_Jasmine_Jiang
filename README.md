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
python evaluate.py --data data --run reproduced/transpose --run reproduced/resize --last-only --device cpu
```

The data preparation command fetches the selected WikiArt v1 directories
from the original Kaggle ZIP and writes 64 × 64 RGB images under `data/`.
It uses `curl` and about 2.7 GB for cached ranges and their assembled copies.
The first evaluation may download the ImageNet Inception-v3 weights.
The local homework folder already contains the prepared `data/` directory.

Training selects CUDA, Apple MPS, or CPU based on availability. The reported
runs used Apple MPS. Use a new output directory for each run. Defaults are
16 epochs, batch size 64, learning rate 0.0002, and seed 42 for the
split and training. Before splitting, the trainer collapses identical and
near-identical 64 × 64 images across style directories. The threshold is
a mean absolute RGB difference of 0.35 on the 0–255 scale. Results can vary
across devices and library versions.

## Experiment files

- `gan.py` contains the two generators, shared discriminator architecture,
  train/holdout split, and training loop.
- `evaluate.py` uses Ignite FID and Inception Score. By default it reads
  `initial.pt` and `last.pt`. `--last-only` evaluates `last.pt`. Pass
  multiple `--run` directories with identical holdouts to share a single
  real-image Inception pass. The latent draws are fixed before batching.
- Each generated run directory contains the checkpoints, sample grids,
  `history.csv`, `metrics.json`, and the exact `split.json` manifest.

## Re-evaluate the report checkpoints

The report uses the two original 16-epoch models. They were trained with
the exact-pixel deduplication in commit `4791914`. The revised evaluation
excludes six original holdout files that have near-identical copies in the
training set and scores both models on the same 506 reference images. The
list is in `report_holdout_exclusions.json`. The original `last.pt` and
`split.json` for each model are needed for this command.

```bash
python evaluate.py --data data --run runs/transpose --run runs/resize --last-only --exclude-holdout report_holdout_exclusions.json --device cpu
```

The above command re-evaluates the original weights. Runs trained with the
current deduplication use their own split and omit `--exclude-holdout`.

## Tested environment

The local Apple Silicon environment used Python 3.13.9, PyTorch 2.15.0,
torchvision 0.29.2, PyTorch-Ignite 0.5.5, SciPy 1.18.1, Matplotlib 3.11.2,
Pillow 13.0.0, and NumPy 2.5.4. MPS was used for original training and CPU
for report metrics.
