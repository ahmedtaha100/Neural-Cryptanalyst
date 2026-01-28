import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_cryptanalyst import (
    TracePreprocessor, FeatureSelector, TraceAugmenter,
    SideChannelCNN, calculate_guessing_entropy
)
from neural_cryptanalyst.attacks.profiled import ProfiledAttack
from neural_cryptanalyst.utils.crypto import aes_sbox

def simulate_aes_traces(num_traces=10000, trace_length=5000, snr=1.0):

    traces = np.random.normal(0, 1, (num_traces, trace_length))
    keys = np.random.randint(0, 256, num_traces)
    plaintexts = np.random.randint(0, 256, num_traces)

    for i in range(num_traces):
        sbox_output = aes_sbox(plaintexts[i] ^ keys[i])
        hw = bin(sbox_output).count("1")
        leakage_point = 1000 + (i % 100)
        traces[i, leakage_point:leakage_point + 50] += hw * snr

    return traces, keys, plaintexts

def main():
    print("=== Neural Cryptanalyst: AES Attack Example ===\n")
    traces, keys, _ = simulate_aes_traces(num_traces=5000)
    train_traces = traces[:4000]
    train_keys = keys[:4000]
    test_traces = traces[4000:]
    test_keys = keys[4000:]

    attack = ProfiledAttack(
        preprocessor=TracePreprocessor(),
        feature_selector=FeatureSelector(),
        augmenter=TraceAugmenter(noise_level=0.1)
    )

    print("Training model...")
    attack.train_model(train_traces, train_keys, num_features=500, epochs=5, batch_size=64)

    print("\nPerforming attack...")
    for n in [1, 5, 10, 20, 50, 100]:
        preds = attack.attack(test_traces, num_attack_traces=n)
        ge = calculate_guessing_entropy(preds, test_keys[0], list(range(1, n + 1)))
        print(f"Guessing entropy with {n} traces: {ge[-1]:.2f}")

if __name__ == "__main__":
    main()
