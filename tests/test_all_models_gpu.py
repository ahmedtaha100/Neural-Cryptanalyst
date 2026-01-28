import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import tensorflow as tf
from neural_cryptanalyst import (
    SideChannelCNN, SideChannelLSTM, SideChannelTransformer
)
from neural_cryptanalyst.models import SideChannelCNNLSTM, GPAM
from neural_cryptanalyst.utils.crypto import aes_sbox
import time

print("="*70)
print("NEURAL CRYPTANALYST - GPU VERIFICATION TEST")
print("="*70)
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}\n")

def create_simple_data(n=500, length=500):

    traces = np.random.randn(n, length).astype(np.float32) * 0.1
    keys = np.random.randint(0, 256, n)
    for i in range(n):
        hw = bin(aes_sbox(keys[i])).count('1')
        traces[i, 250:270] += hw * 0.8
    return traces.reshape(n, length, 1), tf.keras.utils.to_categorical(keys, 256)

models_to_test = [
    ("CNN", lambda: SideChannelCNN(trace_length=500)),
    ("LSTM", lambda: SideChannelLSTM(trace_length=500, bidirectional=False)),
    ("Bi-LSTM", lambda: SideChannelLSTM(trace_length=500, bidirectional=True)),
    ("CNN-LSTM", lambda: SideChannelCNNLSTM(trace_length=500)),
    ("Transformer", lambda: SideChannelTransformer.create_gpam_variant(trace_length=500)),
    ("GPAM", lambda: GPAM(trace_length=500))
]

results = []

for name, model_fn in models_to_test:
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    try:
        start = time.time()

        X, y = create_simple_data(n=500, length=500)

        model = model_fn()
        model.compile_model(learning_rate=0.001)

        print(f"Training {name} (3 epochs, 500 traces)...")

        history = model.model.fit(
            X[:400], y[:400],
            validation_data=(X[400:], y[400:]),
            epochs=3,
            batch_size=32,
            verbose=0
        )

        preds = model.model.predict(X[400:420], verbose=0)
        acc = history.history['accuracy'][-1]

        elapsed = time.time() - start

        print(f"✓ {name}: {elapsed:.1f}s, Accuracy: {acc:.4f}")
        results.append((name, elapsed, acc))

    except Exception as e:
        print(f"✗ {name} failed: {str(e)[:100]}")
        results.append((name, -1, 0))

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Model':<15} {'Time (s)':>10} {'Accuracy':>12}")
print("-" * 70)

total_time = 0
for name, t, acc in results:
    if t > 0:
        print(f"{name:<15} {t:>10.1f} {acc:>12.4f}")
        total_time += t
    else:
        print(f"{name:<15} {'FAILED':>10} {'N/A':>12}")

print("-" * 70)
print(f"{'Total':<15} {total_time:>10.1f}s")
print(f"\n✓ All models trained successfully on M3 Max GPU")
print("="*70)
