import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

class SideChannelDetector:

    def __init__(self, feature_dim: int = 400, hidden_units: tuple = (64, 32)):
        self.feature_dim = feature_dim
        self.hidden_units = hidden_units
        self.model = self._create_detection_model()
        self._countermeasure_callback: Optional[Callable] = None

    def _create_detection_model(self) -> Sequential:
        layers = [Input(shape=(self.feature_dim,))]

        for units in self.hidden_units:
            layers.append(Dense(units, activation='relu'))
            layers.append(Dropout(0.2))

        layers.append(Dense(1, activation='sigmoid'))

        model = Sequential(layers)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def extract_features(self, power_measurements: np.ndarray) -> np.ndarray:
        if power_measurements.ndim == 1:
            power_measurements = power_measurements.reshape(1, -1)

        features = []

        features.append(np.mean(power_measurements, axis=0))
        features.append(np.std(power_measurements, axis=0))
        features.append(np.max(power_measurements, axis=0) - np.min(power_measurements, axis=0))

        fft_values = np.abs(np.fft.fft(power_measurements, axis=-1))
        features.append(np.mean(fft_values, axis=0))

        return np.concatenate([f.flatten() for f in features])

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        features = features.reshape(1, -1)
        current_dim = features.shape[1]

        if current_dim < self.feature_dim:
            features = np.pad(
                features,
                ((0, 0), (0, self.feature_dim - current_dim)),
                mode='constant'
            )
        elif current_dim > self.feature_dim:
            indices = np.linspace(0, current_dim - 1, self.feature_dim, dtype=int)
            features = features[:, indices]

        return features

    def detect_attack(self, power_measurements: np.ndarray,
                      threshold: float = 0.8) -> bool:
        features = self.extract_features(power_measurements)
        features = self._normalize_features(features)

        attack_probability = float(self.model.predict(features, verbose=0)[0][0])

        if attack_probability > threshold:
            self.trigger_countermeasures()
            return True

        return False

    def get_attack_probability(self, power_measurements: np.ndarray) -> float:
        features = self.extract_features(power_measurements)
        features = self._normalize_features(features)
        return float(self.model.predict(features, verbose=0)[0][0])

    def set_countermeasure_callback(self, callback: Callable) -> None:
        self._countermeasure_callback = callback

    def trigger_countermeasures(self) -> None:
        logger.warning("ALERT: Side-channel attack detected!")
        logger.info("Activating countermeasures: random delays, masking, etc.")

        if self._countermeasure_callback is not None:
            self._countermeasure_callback()

    def train(self, normal_traces: np.ndarray, attack_traces: np.ndarray,
              epochs: int = 50, validation_split: float = 0.2) -> tf.keras.callbacks.History:
        normal_features = np.array([
            self._normalize_features(self.extract_features(t.reshape(1, -1)))[0]
            for t in normal_traces
        ])
        attack_features = np.array([
            self._normalize_features(self.extract_features(t.reshape(1, -1)))[0]
            for t in attack_traces
        ])

        X = np.vstack([normal_features, attack_features])
        y = np.concatenate([
            np.zeros(len(normal_features)),
            np.ones(len(attack_features))
        ])

        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

        return self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1
        )
