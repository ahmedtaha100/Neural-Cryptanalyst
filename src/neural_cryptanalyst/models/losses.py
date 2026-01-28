import tensorflow as tf
from tensorflow.keras import backend as K

def ranking_loss(y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_true_idx = tf.argmax(y_true, axis=-1)
    batch_size = tf.shape(y_pred)[0]
    indices = tf.stack([tf.range(batch_size), tf.cast(y_true_idx, tf.int32)], axis=1)
    true_scores = tf.gather_nd(y_pred, indices)
    loss = -tf.math.log(true_scores / tf.reduce_sum(y_pred, axis=-1))
    return tf.reduce_mean(loss)

def focal_loss_ratio(alpha=0.25, gamma=2.0):
    def flr(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true_idx = tf.argmax(y_true, axis=-1)
        batch_size = tf.shape(y_pred)[0]
        indices = tf.stack([tf.range(batch_size), tf.cast(y_true_idx, tf.int32)], axis=1)
        p_t = tf.gather_nd(y_pred, indices)

        focal_pos = -alpha * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)

        num_classes = tf.shape(y_pred)[1]
        class_indices = tf.range(num_classes)
        mask = tf.cast(tf.not_equal(
            tf.expand_dims(y_true_idx, 1),
            tf.expand_dims(class_indices, 0)
        ), tf.float32)

        focal_neg_all = alpha * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred) * mask
        focal_neg = tf.reduce_sum(focal_neg_all, axis=1)

        return tf.reduce_mean(focal_pos / (focal_neg + epsilon))
    return flr

def cross_entropy_ratio(y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_true_idx = tf.argmax(y_true, axis=-1)
    batch_size = tf.shape(y_pred)[0]
    indices = tf.stack([tf.range(batch_size), tf.cast(y_true_idx, tf.int32)], axis=1)
    p_true = tf.gather_nd(y_pred, indices)
    ce_true = -tf.math.log(p_true)

    num_classes = tf.shape(y_pred)[1]
    class_indices = tf.range(num_classes)
    mask = tf.cast(tf.not_equal(
        tf.expand_dims(y_true_idx, 1),
        tf.expand_dims(class_indices, 0)
    ), tf.float32)

    ce_false_all = -tf.math.log(1 - y_pred) * mask
    ce_false = tf.reduce_sum(ce_false_all, axis=1)

    return tf.reduce_mean(ce_true / (ce_false + epsilon))
