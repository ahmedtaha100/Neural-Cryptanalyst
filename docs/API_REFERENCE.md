# Neural Cryptanalyst API Reference

## Preprocessing

### TracePreprocessor
Main preprocessing class for power traces.

**Methods:**
- `fit()`: Fit preprocessing parameters
- `preprocess_traces()`: Apply preprocessing pipeline
- `align_traces_correlation()`: Align using cross-correlation
- `align_traces_dtw()`: Align using Dynamic Time Warping
- `apply_filter()`: Apply various filters
- `detect_and_remove_outliers()`: Remove anomalous traces

### FeatureSelector
Feature selection and dimensionality reduction.

**Methods:**
- `select_poi_sost()`: SOST-based selection
- `select_poi_ttest()`: T-test based selection
- `select_poi_cpa()`: CPA-based selection
- `apply_pca()`: Principal Component Analysis
- `combine_methods()`: Combine multiple selection methods

## Models

### SideChannelCNN
Convolutional Neural Network for SCA.

### SideChannelLSTM
Recurrent Neural Network for temporal analysis.

### SideChannelTransformer
Transformer architecture with attention mechanism.

### Custom Loss Functions
- `ranking_loss()`: Optimized for key ranking
- `focal_loss_ratio()`: Handle class imbalance
- `cross_entropy_ratio()`: Combined CE and ranking

## Attacks

### ProfiledAttack
Template-based attacks with known device.

### NonProfiledAttack
DPA/CPA attacks without profiling.

## Visualization

Complete set of plotting functions for analysis and paper figures.

## Dataset Loading

### ASCADDataset
```python
from neural_cryptanalyst.datasets import ASCADDataset

dataset = ASCADDataset()
traces, labels = dataset.load_ascad_v1('path/to/ASCAD.h5')
```

## Model Persistence

### Saving Models
```python
model.save_model('path/to/model')
```

### Loading Models
```python
model = SideChannelCNN.load_model('path/to/model')
```
