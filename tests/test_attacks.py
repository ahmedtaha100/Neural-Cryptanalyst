import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from neural_cryptanalyst.attacks import ProfiledAttack
from neural_cryptanalyst.attacks.ensemble import EnsembleAttack
from neural_cryptanalyst.models import SideChannelCNN

def test_profiled_attack_workflow():

    traces = np.random.randn(100, 1000).astype(np.float32)
    labels = np.random.randint(0, 256, 100)

    attack = ProfiledAttack(model=SideChannelCNN(trace_length=1000))
    attack.model.compile_model()
    attack.train_model(traces, labels, epochs=1, batch_size=32, validation_split=0.1)

    predictions = attack.attack(traces[:10], num_attack_traces=10)
    assert predictions.shape == (10, 256)

def test_ensemble_attack():

    traces = np.random.randn(20, 100).astype(np.float32)
    labels = np.random.randint(0, 256, 20)

    model1 = SideChannelCNN(trace_length=100)
    model2 = SideChannelCNN(trace_length=100)
    ensemble = EnsembleAttack(models=[model1, model2])
    traces_reshaped = traces.reshape(traces.shape[0], traces.shape[1], 1)
    y_cat = tf.keras.utils.to_categorical(labels, num_classes=256)
    ensemble.train_ensemble(traces_reshaped, y_cat, epochs=1, batch_size=16,
                          X_val=traces_reshaped, y_val=y_cat)

    combined = ensemble.predict_ensemble(traces_reshaped, method='average')
    assert combined.shape[0] == traces.shape[0]
