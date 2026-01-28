import numpy as np
import tensorflow as tf
from typing import List, Union
from ..models import SideChannelCNN, SideChannelLSTM, SideChannelTransformer

class EnsembleAttack:
    def __init__(self, models: List[Union[SideChannelCNN, SideChannelLSTM, SideChannelTransformer]]):
        self.models = models

    def train_ensemble(self, traces, labels, validation_split=0.2,
                       X_val=None, y_val=None, **kwargs):
        if len(traces.shape) == 2:
            traces = traces.reshape(traces.shape[0], traces.shape[1], 1)
        if len(labels.shape) == 1:
            num_classes = getattr(self.models[0], 'num_classes', 256)
            labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        if (X_val is None) != (y_val is None):
            raise ValueError("Provide both X_val and y_val or neither")
        if X_val is None:
            split_idx = int(len(traces) * (1 - validation_split))
            X_train, X_val = traces[:split_idx], traces[split_idx:]
            y_train, y_val = labels[:split_idx], labels[split_idx:]
        else:
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            if len(y_val.shape) == 1:
                num_classes = getattr(self.models[0], 'num_classes', 256)
                y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
            X_train, y_train = traces, labels
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}")
            model.compile_model()
            model.train(X_train, y_train, X_val, y_val, **kwargs)

    def predict_ensemble(self, traces, method: str = 'average'):
        all_predictions = []
        for model in self.models:
            preds = model.model.predict(traces)
            all_predictions.append(preds)
        all_predictions = np.array(all_predictions)

        if method == 'average':
            return np.mean(all_predictions, axis=0)
        elif method == 'vote':
            votes = np.argmax(all_predictions, axis=-1)
            return np.array([np.bincount(votes[:, i]).argmax()
                             for i in range(traces.shape[0])])
        else:
            raise ValueError("Unknown combination method")
