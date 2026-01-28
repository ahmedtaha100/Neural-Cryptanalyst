import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D, Layer
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

class WarmupSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model_float = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step + 1)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model_float) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }

class MultiHeadAttentionWithWeights(Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = None

    def build(self, input_shape):
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim
        )
        super().build(input_shape)

    def call(self, inputs, return_attention_scores=False, training=None):
        if return_attention_scores:
            output, weights = self.attention(
                inputs, inputs,
                return_attention_scores=True,
                training=training
            )
            return output, weights
        return self.attention(inputs, inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config

class RandomShiftAugmentation(Layer):
    def __init__(self, shift_range=100, **kwargs):
        super().__init__(**kwargs)
        self.shift_range = shift_range

    def call(self, inputs, training=None):
        if training:
            return self._apply_shift(inputs)
        return inputs

    def _apply_shift(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        shifts = tf.random.uniform(
            (batch_size,), -self.shift_range, self.shift_range + 1, dtype=tf.int32
        )

        def shift_single(args):
            trace, shift = args
            return tf.roll(trace, shift, axis=0)

        shifted = tf.map_fn(
            shift_single,
            (x, shifts),
            fn_output_signature=tf.TensorSpec(shape=x.shape[1:], dtype=x.dtype)
        )
        return shifted

    def get_config(self):
        config = super().get_config()
        config.update({'shift_range': self.shift_range})
        return config

class SideChannelTransformer:
    def __init__(self, trace_length, d_model=256, num_heads=8, num_classes=256):
        if trace_length <= 0 or d_model <= 0 or num_heads <= 0 or num_classes <= 0:
            raise ValueError("Invalid input dimensions")
        self.trace_length = trace_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_classes = num_classes
        self._attention_layers = []
        self.model = self._build_model()

    def positional_encoding(self, length, depth):
        positions = np.arange(length)[:, np.newaxis].astype('float32')
        depths = (np.arange(depth)[np.newaxis, :] // 2 * 2).astype('float32')
        angle_rates = 1 / (10000**(depths / depth))
        angle_rads = positions * angle_rates
        pos_encoding = np.zeros((length, depth), dtype='float32')
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.constant(pos_encoding, dtype=tf.float32)

    def transformer_block(self, x, ff_dim, num_heads, dropout_rate=0.1):
        attn_layer = MultiHeadAttentionWithWeights(
            num_heads=num_heads,
            key_dim=self.d_model // num_heads
        )
        self._attention_layers.append(attn_layer)

        attn_output = attn_layer(x)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

        ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dropout(dropout_rate),
            Dense(self.d_model)
        ])
        ffn_output = ffn(out1)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        return out2

    def _build_model(self):
        self._attention_layers = []

        inputs = Input(shape=(self.trace_length, 1))
        x = Dense(self.d_model)(inputs)
        pos_encoding = self.positional_encoding(self.trace_length, self.d_model)
        x = x + pos_encoding

        x = self.transformer_block(x, ff_dim=2048, num_heads=self.num_heads)
        x = self.transformer_block(x, ff_dim=2048, num_heads=self.num_heads)
        x = self.transformer_block(x, ff_dim=2048, num_heads=self.num_heads)

        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax', dtype='float32')(x)

        return Model(inputs=inputs, outputs=outputs)

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

    def compile_model(self, learning_rate=0.0001, warmup_steps=4000):
        lr_schedule = WarmupSchedule(self.d_model, warmup_steps)
        optimizer = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
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

    def get_attention_weights(self, traces):
        traces = traces.astype('float32')
        if len(traces.shape) == 2:
            traces = traces.reshape(traces.shape[0], traces.shape[1], 1)

        if not self._attention_layers:
            raise ValueError("No attention layers found in model")

        inputs = self.model.input
        embedding_layer = self.model.layers[1]
        x = embedding_layer(inputs)
        pos_encoding = self.positional_encoding(self.trace_length, self.d_model)
        x = x + pos_encoding

        attention_outputs = []
        layer_idx = 2

        for i, attn_layer in enumerate(self._attention_layers):
            _, weights = attn_layer(x, return_attention_scores=True)
            attention_outputs.append(weights)

            attn_out = attn_layer(x)
            dropout_layer = self.model.layers[layer_idx + 1]
            attn_out = dropout_layer(attn_out, training=False)

            norm_layer1 = self.model.layers[layer_idx + 2]
            x = norm_layer1(x + attn_out)

            ffn_layer = self.model.layers[layer_idx + 3]
            ffn_out = ffn_layer(x)

            norm_layer2 = self.model.layers[layer_idx + 4]
            x = norm_layer2(x + ffn_out)

            layer_idx += 5

        attention_model = Model(inputs=inputs, outputs=attention_outputs)
        return attention_model.predict(traces)

    def save_model(self, filepath: str) -> None:
        self.model.save(filepath)
        import json
        config = {
            'trace_length': self.trace_length,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_classes': self.num_classes
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
            custom_objects={
                'guessing_entropy': instance.guessing_entropy,
                'WarmupSchedule': WarmupSchedule,
                'MultiHeadAttentionWithWeights': MultiHeadAttentionWithWeights,
                'RandomShiftAugmentation': RandomShiftAugmentation
            }
        )
        return instance

    @classmethod
    def create_gpam_variant(cls, trace_length: int, num_classes: int = 256):
        return cls(trace_length=trace_length, d_model=512, num_heads=16, num_classes=num_classes)

    @classmethod
    def create_transnet_variant(cls, trace_length: int, num_classes: int = 256):
        return cls(trace_length=trace_length, d_model=256, num_heads=8, num_classes=num_classes)

    def add_shift_invariance_augmentation(self, shift_range: int = 100):
        self._attention_layers = []

        inputs = Input(shape=(self.trace_length, 1))

        x = RandomShiftAugmentation(shift_range=shift_range)(inputs)

        x = Dense(self.d_model)(x)
        pos_encoding = self.positional_encoding(self.trace_length, self.d_model)
        x = x + pos_encoding

        x = self.transformer_block(x, ff_dim=2048, num_heads=self.num_heads)
        x = self.transformer_block(x, ff_dim=2048, num_heads=self.num_heads)
        x = self.transformer_block(x, ff_dim=2048, num_heads=self.num_heads)

        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax', dtype='float32')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
