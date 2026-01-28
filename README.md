# Neural Cryptanalyst

**Machine Learning-Powered Side-Channel Attacks**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository accompanies the paper:

> **The Neural Cryptanalyst: Machine Learning-Powered Side-Channel Attacks - A Comprehensive Survey**
> Ahmed Taha, Johns Hopkins University, 2025
> [[PDF]](The%20Neural%20Cryptanalyst.pdf) [[Zenodo]](https://zenodo.org/records/18407224)

---

## Citation

```bibtex
@misc{taha2025neural,
  author       = {Taha, Ahmed},
  title        = {The Neural Cryptanalyst: Machine Learning-Powered
                  Side-Channel Attacks - A Comprehensive Survey},
  year         = {2025},
  doi          = {10.5281/zenodo.18407224},
  url          = {https://github.com/ahmedtaha100/Neural-Cryptanalyst}
}
```

---

## Overview

Deep learning reduces the number of power traces required to compromise cryptographic implementations by 80-90% compared to traditional statistical methods. This repository provides:

- **Architectures:** CNN, LSTM, CNN-LSTM, Transformer, and GPAM models for side-channel analysis
- **Attacks:** Profiled and non-profiled attacks against AES, RSA, and ECC
- **Preprocessing:** Trace alignment, filtering, and POI selection (SOST, mutual information, PCA)
- **Countermeasures:** Boolean masking, hiding, blinding, constant-time implementations
- **Evaluation:** Guessing entropy, success rate, mutual information metrics

---

## Installation

```bash
git clone https://github.com/ahmedtaha100/Neural-Cryptanalyst.git
cd Neural-Cryptanalyst
pip install -e .
```

---

## Dataset

Download the ASCAD dataset from the [ANSSI-FR/ASCAD repository](https://github.com/ANSSI-FR/ASCAD) and place `ASCAD.h5` in `./ASCAD_data/`.

---

## Quick Start

```python
from neural_cryptanalyst.attacks.profiled import ProfiledAttack
from neural_cryptanalyst.models import SideChannelCNN
from neural_cryptanalyst.datasets import ASCADDataset

dataset = ASCADDataset()
traces, labels = dataset.load_ascad_v1("ASCAD_data/ASCAD.h5")

attack = ProfiledAttack(model=SideChannelCNN(trace_length=700))
attack.train_model(traces[:45000], labels[:45000], epochs=50)
predictions = attack.attack(traces[45000:45100])
```

See [examples/](examples/) for additional workflows including preprocessing, ensemble attacks, and countermeasure evaluation.

---

## Reproducing Paper Results

Scripts to reproduce the experimental results are in [paper_reproduction/](paper_reproduction/).

---

## Project Structure

```
Neural-Cryptanalyst/
|-- src/neural_cryptanalyst/
|   |-- models/           # Neural network architectures
|   |-- attacks/          # Attack implementations
|   |-- preprocessing/    # Trace processing pipeline
|   |-- countermeasures/  # Defense implementations
|   |-- evaluation/       # Metrics and benchmarking
|-- examples/             # Usage examples
|-- paper_reproduction/   # Scripts to reproduce paper results
|-- tests/                # Test suite
```

---

## Security Notice

This software is for research and educational purposes. Users are responsible for compliance with applicable laws.

---

## License

MIT License. See [LICENSE](LICENSE).
