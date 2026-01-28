import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from typing import Optional, Tuple

class ShiftInvariantPositionalEncoding(layers.Layer):

    def __init__(self, max_length: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.base_encoding = self.add_weight(
            name='base_pos_encoding',
            shape=(1, self.max_length, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.shift_mlp = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(self.embed_dim)
        ])

    def call(self, x, detected_shift=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        pos_encoding = self.base_encoding[:, :seq_len, :]
        if detected_shift is not None:
            shift_adjustment = self.shift_mlp(tf.cast(detected_shift, tf.float32))
            shift_adjustment = tf.reshape(shift_adjustment, [batch_size, 1, self.embed_dim])
            pos_encoding = pos_encoding + shift_adjustment
        return x + pos_encoding

class ShiftDetectionModule(layers.Layer):

    def __init__(self, hidden_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(64, 7, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(128, 5, padding='same', activation='relu')
        self.pool = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(self.hidden_dim, activation='relu')
        self.fc2 = layers.Dense(1)

    def call(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.pool(h)
        h = self.fc1(h)
        shift_estimate = self.fc2(h)
        return shift_estimate

class AlignmentAwareAttention(layers.Layer):

    def __init__(self, num_heads: int, key_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = layers.MultiHeadAttention(num_heads, key_dim)

    def build(self, input_shape):
        self.alignment_score = layers.Dense(1, activation='sigmoid')

    def call(self, x, training=None):
        attn_output = self.attention(x, x, training=training)
        alignment_confidence = self.alignment_score(x)
        weighted_output = attn_output * alignment_confidence
        return weighted_output

class TransNet(Model):

    def __init__(self,
                 trace_length: int,
                 num_classes: int = 256,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 use_shift_detection: bool = True,
                 max_shift: int = 1000):
        super().__init__()

        self.trace_length = trace_length
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_shift_detection = use_shift_detection
        self.max_shift = max_shift

        self.input_projection = layers.Dense(embed_dim)
        self.input_norm = layers.LayerNormalization()

        if use_shift_detection:
            self.shift_detector = ShiftDetectionModule()

        self.pos_encoding = ShiftInvariantPositionalEncoding(
            trace_length + max_shift, embed_dim
        )

        self.transformer_blocks = []
        for i in range(num_layers):
            block = {
                'attention': AlignmentAwareAttention(num_heads, embed_dim // num_heads),
                'norm1': layers.LayerNormalization(),
                'ffn': tf.keras.Sequential([
                    layers.Dense(embed_dim * 4, activation='gelu'),
                    layers.Dense(embed_dim)
                ]),
                'norm2': layers.LayerNormalization(),
                'dropout': layers.Dropout(0.1)
            }
            self.transformer_blocks.append(block)

        self.global_pool = layers.GlobalAveragePooling1D()
        self.output_norm = layers.LayerNormalization()
        self.classifier = layers.Dense(num_classes, activation='softmax')

        if use_shift_detection:
            self.shift_predictor = layers.Dense(1, name='shift_output')

    def detect_shift(self, x):
        if self.use_shift_detection:
            return self.shift_detector(x)
        return None

    def call(self, inputs, training=None):
        if len(inputs.shape) == 2:
            x = tf.expand_dims(inputs, -1)
        else:
            x = inputs
        detected_shift = self.detect_shift(x) if self.use_shift_detection else None
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.pos_encoding(x, detected_shift)
        for block in self.transformer_blocks:
            attn_out = block['attention'](x, training=training)
            x = x + attn_out
            x = block['norm1'](x)
            ffn_out = block['ffn'](x)
            x = x + ffn_out
            x = block['norm2'](x)
            x = block['dropout'](x, training=training)
        x = self.global_pool(x)
        x = self.output_norm(x)
        key_output = self.classifier(x)
        if self.use_shift_detection and training:
            shift_output = self.shift_predictor(x)
            return {'key': key_output, 'shift': shift_output}
        return key_output

    def compile_with_shift_loss(self, optimizer='adam', alpha=0.1):
        losses = {
            'key': 'categorical_crossentropy',
            'shift': 'mse'
        }
        loss_weights = {
            'key': 1.0,
            'shift': alpha
        }
        metrics = {
            'key': ['accuracy'],
            'shift': ['mae']
        }
        self.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

class ShiftAugmentation(layers.Layer):

    def __init__(self, max_shift: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.max_shift = max_shift

    def call(self, x, training=None):
        if not training:
            return x
        batch_size = tf.shape(x)[0]
        trace_length = tf.shape(x)[1]
        shifts = tf.random.uniform(
            (batch_size,),
            -self.max_shift,
            self.max_shift,
            dtype=tf.int32
        )
        shifted = []
        for i in range(batch_size):
            shifted_trace = tf.roll(x[i], shift=shifts[i], axis=0)
            shifted.append(shifted_trace)
        return tf.stack(shifted)

def create_shift_invariant_model(trace_length: int,
                                num_classes: int = 256,
                                max_expected_shift: int = 1000) -> TransNet:
    model = TransNet(
        trace_length=trace_length,
        num_classes=num_classes,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        use_shift_detection=True,
        max_shift=max_expected_shift
    )
    return model

def create_shift_augmented_training_data(traces: np.ndarray,
                                         labels: np.ndarray,
                                         max_shift: int = 1000,
                                         augmentation_factor: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    augmented_traces = []
    augmented_labels = []
    shift_values = []
    for _ in range(augmentation_factor):
        for trace, label in zip(traces, labels):
            shift = np.random.randint(-max_shift, max_shift)
            shifted_trace = np.roll(trace, shift)
            augmented_traces.append(shifted_trace)
            augmented_labels.append(label)
            shift_values.append(shift)
    return (np.array(augmented_traces),
            np.array(augmented_labels),
            np.array(shift_values))
