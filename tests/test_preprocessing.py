import numpy as np
import pytest
from neural_cryptanalyst.preprocessing import TracePreprocessor, FeatureSelector, TraceAugmenter

def test_trace_preprocessor_fit():

    traces = np.random.randn(100, 1000)
    preprocessor = TracePreprocessor()

    preprocessor.fit(traces)
    assert preprocessor._fitted
    assert preprocessor.reference_trace is not None
    assert preprocessor.scaler is not None

def test_preprocessing_pipeline():

    traces = np.random.randn(100, 1000) + 5.0
    preprocessor = TracePreprocessor()
    preprocessor.fit(traces)

    processed = preprocessor.preprocess_traces(traces)

    assert processed.shape == traces.shape

def test_feature_selector_all_methods():

    traces = np.random.randn(100, 500)
    labels = np.random.randint(0, 2, 100)

    selector = FeatureSelector()

    idx1, selected1 = selector.select_poi_sost(traces, labels, num_poi=50)
    assert len(idx1) == 50
    assert selected1.shape == (100, 50)

    idx2, selected2 = selector.select_poi_ttest(traces, labels, num_poi=50)
    assert len(idx2) == 50

    idx3, selected3 = selector.select_poi_mutual_information(traces, labels, num_poi=50)
    assert len(idx3) == 50

    transformed = selector.transform(traces)
    assert transformed.shape == (100, 50)

def test_augmentation():

    traces = np.random.randn(10, 100)
    labels = np.random.randint(0, 256, 10)

    augmenter = TraceAugmenter(random_state=42)

    noisy = augmenter.add_gaussian_noise(traces)
    assert noisy.shape == traces.shape
    assert not np.array_equal(noisy, traces)

    shifted = augmenter.random_shift(traces)
    assert shifted.shape == traces.shape

    scaled = augmenter.random_scale(traces)
    assert scaled.shape == traces.shape

    aug_traces, aug_labels = augmenter.augment_batch(traces, labels, num_augmented=2)
    assert aug_traces.shape == (30, 100)
    assert aug_labels.shape == (30,)
