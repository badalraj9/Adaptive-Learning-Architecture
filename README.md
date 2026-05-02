# ALA Experimental Architecture

This repository contains experimental implementations of the Adaptive Learning
Architecture.

## Layout

- `model/` - model implementation files
- `model/legacy/` - older v7-style demo implementation files
- `docs/` - architecture specifications
- `tests/` - experiments and older compression tests
- `runtime/` - generated states, checkpoints, and caches
- `archives/` - backup zip files

## Current Trainable Target

Use `model/ALA_v8.py` for the v8.1 trainable prototype. It implements a
prediction-error-driven loop, bounded primitive memory, and an adaptive strategy
layer. The default configuration is intended to be roughly a 5M-parameter model.

## Colab Smoke Train

```bash
pip install -r requirements.txt
python train_v8_colab.py --steps 1000 --batch-size 64
```

Checkpoints and metrics are written to `runtime/v8_runs/`.

The older files are kept in `model/legacy/` for reference and comparison, but
they are closer to the v7.1 demo architecture than the v8.1 training target.
