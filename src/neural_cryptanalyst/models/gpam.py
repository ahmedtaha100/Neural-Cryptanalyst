import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from typing import List, Tuple, Optional

class TemporalPatchEmbedding(layers.Layer):

    def __init__(self, patch_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = None

    def build(self, input_shape):
        trace_length = input_shape[1]
        self.num_patches = (trace_length + self.patch_size - 1) // self.patch_size
        self.padded_length = self.num_patches * self.patch_size

        self.projection = layers.Dense(self.embed_dim)
        self.position_embedding = self.add_weight(
            name='pos_embed',
            shape=(1, self.num_patches, self.embed_dim),
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        trace_length = tf.shape(x)[1]

        pad_length = self.padded_length - trace_length
        if pad_length > 0:
            x = tf.pad(x, [[0, 0], [0, pad_length], [0, 0]])

        patches = tf.reshape(x, [batch_size, self.num_patches, self.patch_size])
        x = self.projection(patches)
        x += self.position_embedding
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim
        })
        return config

class MultiScaleAttention(layers.Layer):

    def __init__(self, num_heads: int, embed_dim: int, scales: List[int] = [1, 2, 4], **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.scales = scales

    def build(self, input_shape):
        self.attention_layers = []
        for scale in self.scales:
            self.attention_layers.append(
                layers.MultiHeadAttention(
                    num_heads=self.num_heads // len(self.scales),
                    key_dim=self.embed_dim // self.num_heads
                )
            )
        self.combine = layers.Dense(self.embed_dim)

    def call(self, x, training=None):
        outputs = []
        for scale, attn in zip(self.scales, self.attention_layers):
            if scale > 1:
                pooled = tf.nn.avg_pool1d(x, scale, scale, 'SAME')
                attn_out = attn(pooled, pooled, training=training)
                upsampled = tf.repeat(attn_out, scale, axis=1)
                upsampled = upsampled[:, :tf.shape(x)[1], :]
                outputs.append(upsampled)
            else:
                outputs.append(attn(x, x, training=training))
        combined = tf.concat(outputs, axis=-1)
        return self.combine(combined)

class GPAM(Model):

    def __init__(self,
                 trace_length: int,
                 num_classes: int = 256,
                 patch_size: int = 100,
                 embed_dim: int = 512,
                 num_heads: int = 16,
                 num_layers: int = 6,
                 scales: List[int] = [1, 2, 4],
                 use_multi_task: bool = True):
        super().__init__()

        self.trace_length = trace_length
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.scales = scales
        self.use_multi_task = use_multi_task

        self.input_norm = layers.LayerNormalization()
        self.patch_embed = TemporalPatchEmbedding(patch_size, embed_dim)

        self.transformer_blocks = []
        for _ in range(num_layers):
            block = [
                MultiScaleAttention(num_heads, embed_dim, scales),
                layers.LayerNormalization(),
                layers.Dense(embed_dim * 4, activation='gelu'),
                layers.Dense(embed_dim),
                layers.LayerNormalization(),
                layers.Dropout(0.1)
            ]
            self.transformer_blocks.append(block)

        self.global_pool = layers.GlobalAveragePooling1D()
        self.key_head = layers.Dense(num_classes, activation='softmax', name='key_output')

        if use_multi_task:
            self.hw_head = layers.Dense(9, activation='softmax', name='hw_output')
            self.mask_head = layers.Dense(256, activation='softmax', name='mask_output')

    def call(self, inputs, training=None):
        if len(inputs.shape) == 2:
            x = tf.expand_dims(inputs, -1)
        else:
            x = inputs
        x = self.input_norm(x)
        x = self.patch_embed(x)
        for block in self.transformer_blocks:
            attn_out = block[0](x, training=training)
            x = x + attn_out
            x = block[1](x)
            ffn_out = block[2](x)
            ffn_out = block[3](ffn_out)
            x = x + ffn_out
            x = block[4](x)
            x = block[5](x, training=training)
        x = self.global_pool(x)
        outputs = {'key': self.key_head(x)}
        if self.use_multi_task and training:
            outputs['hw'] = self.hw_head(x)
            outputs['mask'] = self.mask_head(x)
        return outputs if self.use_multi_task else outputs['key']

    def compile_multi_task(self, optimizer='adam', loss_weights=None):
        if loss_weights is None:
            loss_weights = {'key': 1.0, 'hw': 0.3, 'mask': 0.2}
        losses = {
            'key': 'categorical_crossentropy',
            'hw': 'categorical_crossentropy',
            'mask': 'categorical_crossentropy'
        }
        metrics = {
            'key': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)],
            'hw': ['accuracy'],
            'mask': ['accuracy']
        }
        self.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

    @classmethod
    def create_gpam_small(cls, trace_length: int, num_classes: int = 256):
        return cls(
            trace_length=trace_length,
            num_classes=num_classes,
            patch_size=50,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            scales=[1, 2],
            use_multi_task=True
        )

    @classmethod
    def create_gpam_large(cls, trace_length: int, num_classes: int = 256):
        return cls(
            trace_length=trace_length,
            num_classes=num_classes,
            patch_size=100,
            embed_dim=768,
            num_heads=24,
            num_layers=8,
            scales=[1, 2, 4, 8],
            use_multi_task=True
        )

def create_multi_task_labels(traces: np.ndarray, keys: np.ndarray,
                             plaintexts: Optional[np.ndarray] = None) -> dict:
    from ..utils.crypto import aes_sbox, hamming_weight
    labels = {}
    labels['key'] = tf.keras.utils.to_categorical(keys, num_classes=256)
    if plaintexts is not None:
        sbox_outputs = [aes_sbox(p ^ k) for p, k in zip(plaintexts, keys)]
        hw_values = [hamming_weight(s) for s in sbox_outputs]
        labels['hw'] = tf.keras.utils.to_categorical(hw_values, num_classes=9)
    else:
        hw_values = [hamming_weight(k) for k in keys]
        labels['hw'] = tf.keras.utils.to_categorical(hw_values, num_classes=9)
    masks = np.random.randint(0, 256, len(keys))
    labels['mask'] = tf.keras.utils.to_categorical(masks, num_classes=256)
    return labels
