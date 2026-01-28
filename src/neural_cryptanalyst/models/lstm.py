import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

class SideChannelLSTM:

    def __init__(self, trace_length: int, num_features: int = 1,
                 num_classes: int = 256, bidirectional: bool = False):
        self.trace_length = trace_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.model = self._build_bidirectional_model() if bidirectional else self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.trace_length, self.num_features)),
            Dropout(0.25),
            BatchNormalization(),

            LSTM(256, return_sequences=True),
            Dropout(0.25),
            BatchNormalization(),

            LSTM(512, return_sequences=False),
            Dropout(0.25),
            BatchNormalization(),

            Dense(1024, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),

            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def _build_bidirectional_model(self) -> Sequential:
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.trace_length, self.num_features)),
            Dropout(0.25),
            BatchNormalization(),

            Bidirectional(LSTM(256, return_sequences=True)),
            Dropout(0.25),
            BatchNormalization(),

            Bidirectional(LSTM(512, return_sequences=False)),
            Dropout(0.25),
            BatchNormalization(),

            Dense(1024, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),

            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def guessing_entropy(self, y_true, y_pred):
        y_true_class = tf.cast(K.argmax(y_true, axis=-1), tf.int32)
        sorted_indices = tf.argsort(y_pred, axis=-1, direction='DESCENDING')

        batch_size = tf.shape(y_true_class)[0]
        row_indices = tf.range(batch_size)

        ranks = tf.map_fn(
            lambda i: tf.cast(
                tf.where(tf.equal(sorted_indices[i], y_true_class[i]))[0, 0],
                tf.float32
            ),
            row_indices,
            dtype=tf.float32
        )
        return K.mean(ranks)

    def compile_model(self, learning_rate: float = 0.0001, clipnorm: float = 1.0):
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', self.guessing_entropy]
        )

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 64):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def save_model(self, filepath: str) -> None:
        self.model.save(filepath)
        import json
        config = {
            'trace_length': self.trace_length,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'bidirectional': self.bidirectional
        }
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config, f)

    @classmethod
    def load_model(cls, filepath: str):
        import json
        with open(filepath + '_config.json', 'r') as f:
            config = json.load(f)
        instance = cls(**config)
        instance.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'guessing_entropy': instance.guessing_entropy}
        )
        return instance
