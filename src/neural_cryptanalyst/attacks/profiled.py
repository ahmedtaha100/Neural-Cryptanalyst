import numpy as np
from typing import Optional, Union, Tuple, List
import tensorflow as tf

from ..preprocessing import TracePreprocessor, FeatureSelector, TraceAugmenter
from ..models import SideChannelCNN, SideChannelLSTM

class ProfiledAttack:
    def __init__(self, model: Optional[Union[SideChannelCNN, SideChannelLSTM]] = None,
                 preprocessor: Optional[TracePreprocessor] = None,
                 feature_selector: Optional[FeatureSelector] = None,
                 augmenter: Optional[TraceAugmenter] = None,
                 num_classes: Optional[int] = None):
        self.model = model
        self.preprocessor = preprocessor or TracePreprocessor()
        self.feature_selector = feature_selector or FeatureSelector()
        self.augmenter = augmenter
        if num_classes is not None:
            self.num_classes = num_classes
        elif model is not None and hasattr(model, 'num_classes'):
            self.num_classes = model.num_classes
        else:
            self.num_classes = 256
        self.key_predictions = None
        self._fitted_num_features = None

    def prepare_data(self, traces: np.ndarray, labels: np.ndarray,
                     num_features: int = 1000, augment: bool = False,
                     fit_selector: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.preprocessor._fitted:
            self.preprocessor.fit(traces)
        processed = self.preprocessor.preprocess_traces(traces)

        if fit_selector:
            _, selected = self.feature_selector.select_poi_sost(
                processed, labels, num_poi=num_features
            )
            self._fitted_num_features = num_features
        else:
            if self.feature_selector.selected_indices is None:
                raise ValueError(
                    "Feature selector must be fitted during training before attack. "
                    "Call train_model() first."
                )
            selected = self.feature_selector.transform(processed)

        if augment and self.augmenter:
            selected, labels = self.augmenter.augment_batch(selected, labels)

        if len(selected.shape) == 2:
            selected = selected.reshape(selected.shape[0], selected.shape[1], 1)

        return selected, labels

    def train_model(self, traces: np.ndarray, labels: np.ndarray,
                    validation_split: float = 0.2, num_features: int = 1000,
                    epochs: int = 100, batch_size: int = 64):
        X, y = self.prepare_data(
            traces, labels, num_features=num_features, augment=True, fit_selector=True
        )
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_onehot[:split_idx], y_onehot[split_idx:]

        if self.model is None:
            self.model = SideChannelCNN(
                trace_length=X.shape[1],
                num_classes=self.num_classes
            )

        self.model.compile_model()

        history = self.model.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size
        )
        return history

    def attack(self, traces: np.ndarray,
               num_attack_traces: Optional[int] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained before attack")

        if num_attack_traces is None:
            num_attack_traces = len(traces)

        attack_traces = traces[:num_attack_traces]

        dummy_labels = np.zeros(len(attack_traces))
        X, _ = self.prepare_data(
            attack_traces, dummy_labels,
            num_features=self._fitted_num_features or 1000,
            augment=False,
            fit_selector=False
        )

        predictions = self.model.model.predict(X)
        self.key_predictions = predictions
        return predictions

    def analyze_attack_quality(self, traces: np.ndarray, predictions: np.ndarray,
                               correct_key: int) -> dict:
        from ..attacks.metrics import calculate_mutual_information_analysis

        metrics = {}

        mi = calculate_mutual_information_analysis(traces, predictions, correct_key)
        metrics['mutual_information'] = mi

        correct_probs = predictions[:, correct_key]
        other_probs = np.delete(predictions, correct_key, axis=1)

        signal = np.mean(correct_probs)
        noise = np.std(other_probs.flatten())
        metrics['prediction_snr'] = signal / (noise + 1e-10)

        metrics['discrimination_ratio'] = np.mean(correct_probs) / (np.mean(other_probs) + 1e-10)

        max_probs = np.max(predictions, axis=1)
        metrics['average_confidence'] = np.mean(max_probs)
        metrics['correct_key_confidence'] = np.mean(correct_probs)

        predicted_keys = np.argmax(predictions, axis=1)
        metrics['success_rate'] = np.mean(predicted_keys == correct_key)

        return metrics

    def attack_with_analysis(self, traces: np.ndarray,
                             correct_key: Optional[int] = None,
                             num_attack_traces: Optional[int] = None) -> Tuple[np.ndarray, Optional[dict]]:
        if num_attack_traces is None:
            num_attack_traces = len(traces)

        predictions = self.attack(traces, num_attack_traces)

        analysis = None
        if correct_key is not None:
            analysis = self.analyze_attack_quality(
                traces[:num_attack_traces],
                predictions,
                correct_key
            )

        return predictions, analysis

    def recover_key(self, predictions: np.ndarray,
                    aggregation: str = 'sum_log') -> Tuple[int, np.ndarray]:
        if aggregation == 'sum_log':
            log_probs = np.log(predictions + 1e-36)
            key_scores = np.sum(log_probs, axis=0)
        elif aggregation == 'product':
            key_scores = np.prod(predictions, axis=0)
        elif aggregation == 'majority':
            votes = np.argmax(predictions, axis=1)
            key_scores = np.bincount(votes, minlength=self.num_classes).astype(float)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        best_key = int(np.argmax(key_scores))
        return best_key, key_scores
