# Paper Reproduction Guide

This directory contains scripts to reproduce the experiments from the *Neural Cryptanalyst* paper. Running these scripts will generate the figures and metrics reported in the publication.

## Requirements
- Python 3.8 or later
- TensorFlow 2.8+
- ASCAD dataset (`./data/ASCAD.h5`)
- A GPU is recommended but not required

Install the project requirements from the repository root:

```bash
pip install -r requirements.txt
```

## Running the experiments
1. Download the ASCAD dataset and place it at `./data/ASCAD.h5` (or edit the path in `reproduce_results.py`).
2. Execute the reproduction script:

```bash
python reproduce_results.py
```

All plots and result files will be written to the `results/` folder.
