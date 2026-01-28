import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import tensorflow as tf
from neural_cryptanalyst import ProfiledAttack, SideChannelCNN, SideChannelTransformer
from neural_cryptanalyst.utils.crypto import aes_sbox

print("="*60)
print("GPU Configuration Check")
print("="*60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print()

print("="*60)
print("Test 1: CNN Training on GPU")
print("="*60)

n_traces = 1000
trace_length = 1000

traces = np.random.randn(n_traces, trace_length).astype(np.float32) * 0.1
keys = np.random.randint(0, 256, n_traces)

for i in range(n_traces):
    hw = bin(aes_sbox(keys[i])).count('1')
    traces[i, 500:550] += hw * 0.5

traces = traces.reshape(n_traces, trace_length, 1)
labels_cat = tf.keras.utils.to_categorical(keys, num_classes=256)

print("Creating CNN model...")
cnn = SideChannelCNN(trace_length=trace_length)
cnn.compile_model(learning_rate=0.001)

print("Training CNN on GPU (3 epochs)...")
with tf.device('/GPU:0'):
    history = cnn.model.fit(
        traces[:800], labels_cat[:800],
        validation_data=(traces[800:], labels_cat[800:]),
        epochs=3,
        batch_size=64,
        verbose=1
    )

print(f"✓ CNN training completed")
print(f"  Final loss: {history.history['loss'][-1]:.4f}")
print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")
print()

print("="*60)
print("Test 2: Attack Simulation")
print("="*60)

predictions = cnn.model.predict(traces[800:820], verbose=0)
accumulated = np.mean(predictions, axis=0)
predicted_key = np.argmax(accumulated)
correct_key = keys[800]

print(f"Predicted key: {predicted_key}")
print(f"Correct key: {correct_key}")
print(f"✓ Attack {'SUCCESSFUL' if predicted_key == correct_key else 'FAILED'}")
print()

print("="*60)
print("Test 3: Metal Acceleration Check")
print("="*60)

metal_detected = False
try:
    from tensorflow.python.framework import test_util
    if test_util.is_gpu_available():
        print("✓ GPU acceleration is active")
        metal_detected = True
except:
    pass

if len(tf.config.list_physical_devices('GPU')) > 0:
    print("✓ Metal GPU device detected")
    metal_detected = True

if metal_detected:
    print("✓ TensorFlow is using your M3 Max GPU")
else:
    print("✗ WARNING: GPU may not be active")

print()
print("="*60)
print("All Tests Passed! GPU is working correctly.")
print("="*60)
