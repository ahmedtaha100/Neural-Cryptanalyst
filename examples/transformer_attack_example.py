import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neural_cryptanalyst import (
    TracePreprocessor, FeatureSelector, TraceAugmenter,
    SideChannelTransformer, PowerConsumptionModel,
    ranking_loss, calculate_guessing_entropy,
    plot_power_traces, plot_guessing_entropy_evolution
)

def generate_realistic_traces(num_traces=10000, trace_length=5000):
    model = PowerConsumptionModel(sigma=0.5)
    traces = []
    keys = np.random.randint(0, 256, num_traces)
    plaintexts = np.random.randint(0, 256, num_traces)
    for i in range(num_traces):
        trace = model.calculate_power_trace(plaintexts[i], keys[i], num_samples=trace_length, operations=['sbox'])
        traces.append(trace)
    return np.array(traces), keys, plaintexts

def main():
    print("=== Neural Cryptanalyst: Transformer Attack Example ===\n")
    traces, keys, plaintexts = generate_realistic_traces(num_traces=5000)
    train_traces = traces[:4000]
    train_keys = keys[:4000]
    test_traces = traces[4000:]
    test_keys = keys[4000:]
    preprocessor = TracePreprocessor()
    selector = FeatureSelector()
    preprocessor.fit(train_traces)
    processed_train = preprocessor.preprocess_traces(train_traces)
    processed_test = preprocessor.preprocess_traces(test_traces)
    _, selected_train = selector.select_poi_sost(processed_train, train_keys, num_poi=1000)
    selected_test = selector.transform(processed_test)
    selected_train = selected_train.reshape(selected_train.shape[0], selected_train.shape[1], 1)
    selected_test = selected_test.reshape(selected_test.shape[0], selected_test.shape[1], 1)
    transformer = SideChannelTransformer(trace_length=1000, d_model=128, num_heads=8)
    transformer.compile_model()
    import tensorflow as tf
    train_keys_cat = tf.keras.utils.to_categorical(train_keys, num_classes=256)
    print("Training Transformer model...")
    history = transformer.model.fit(selected_train, train_keys_cat, validation_split=0.2, epochs=20, batch_size=64, verbose=1)
    print("\nPerforming attack...")
    num_traces_list = [1, 5, 10, 20, 50, 100]
    ge_values = []
    for n in num_traces_list:
        predictions = transformer.model.predict(selected_test[:n])
        ge = calculate_guessing_entropy(predictions, test_keys[0], list(range(1, n + 1)))
        ge_values.append(ge[-1])
        print(f"Guessing entropy with {n} traces: {ge[-1]:.2f}")
    plot_guessing_entropy_evolution(np.array(ge_values), num_traces_list, title="Transformer Attack - Guessing Entropy Evolution")
    print("\nExtracting attention weights...")
    attention_weights = transformer.get_attention_weights(selected_test[:1])
    print(f"Attention shape: {attention_weights[0].shape}")

if __name__ == "__main__":
    main()
