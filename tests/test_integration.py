import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from neural_cryptanalyst import (
    TracePreprocessor, FeatureSelector, ProfiledAttack,
    SideChannelCNN, calculate_guessing_entropy
)

def test_full_attack_pipeline():
    traces = np.random.randn(1000, 5000)
    labels = np.random.randint(0, 256, 1000)

    preprocessor = TracePreprocessor()
    selector = FeatureSelector()

    preprocessor.fit(traces)
    processed = preprocessor.preprocess_traces(traces)

    poi_idx, selected = selector.select_poi_sost(processed, labels, num_poi=1000)

    attack = ProfiledAttack(model=SideChannelCNN(trace_length=1000))
    attack.model.compile_model()
    attack.train_model(selected, labels, epochs=2, batch_size=32)

    test_traces = np.random.randn(10, 5000)
    test_processed = preprocessor.preprocess_traces(test_traces)
    test_selected = selector.transform(test_processed)

    predictions = attack.attack(test_selected, num_attack_traces=10)

    ge = calculate_guessing_entropy(predictions, labels[0], [1, 5, 10])
    assert len(ge) == 3
