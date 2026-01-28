import numpy as np
import pytest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from neural_cryptanalyst.models import (
    SideChannelCNN, SideChannelLSTM, SideChannelCNNLSTM,
    SideChannelTransformer, PowerConsumptionModel
)

def test_all_models_creation():
    trace_length = 1000
    cnn = SideChannelCNN(trace_length=trace_length)
    assert cnn.model is not None
    lstm = SideChannelLSTM(trace_length=trace_length)
    assert lstm.model is not None
    cnn_lstm = SideChannelCNNLSTM(trace_length=trace_length)
    assert cnn_lstm.model is not None
    transformer = SideChannelTransformer(trace_length=trace_length)
    assert transformer.model is not None

def test_power_consumption_model():
    model = PowerConsumptionModel()
    trace = model.calculate_power_trace(data=100, key=42, num_samples=1000)
    assert trace.shape == (1000,)
    assert not np.all(trace == 0)
    assert model.hamming_weight(0) == 0
    assert model.hamming_weight(255) == 8
    assert model.hamming_weight(15) == 4
    assert model.hamming_distance(0, 0) == 0
    assert model.hamming_distance(0, 255) == 8
    assert model.hamming_distance(15, 240) == 8

def test_custom_losses():
    from neural_cryptanalyst.models.losses import ranking_loss, focal_loss_ratio, cross_entropy_ratio
    model = SideChannelCNN(trace_length=100)
    model.model.compile(optimizer='adam', loss=ranking_loss, metrics=['accuracy'])
    flr = focal_loss_ratio(alpha=0.25, gamma=2.0)
    model.model.compile(optimizer='adam', loss=flr, metrics=['accuracy'])
    model.model.compile(optimizer='adam', loss=cross_entropy_ratio, metrics=['accuracy'])
