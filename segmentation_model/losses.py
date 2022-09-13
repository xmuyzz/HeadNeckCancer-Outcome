from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf
# https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
# https://gist.github.com/jerheff/8cf06fe1df0695806456



epsilon = 1e-5
smooth = 1

# overlap measures
def precision_loss(y_true, y_pred):
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    precision = (tp + smooth)/(tp+fp+smooth)
    return 1 - precision

def recall_loss(y_true, y_pred):
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    recall = (tp+smooth)/(tp+fn+smooth)
    return 1 - recall

def dsc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(float(y_true_f) * float(y_pred_f))
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(float(y_true), float(y_pred))
    return loss

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky_loss(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

# cross entropy
def bce_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

def wce(beta=0.5):
    # To decrease the number of false negatives, set β>1. To decrease the number of false positives, set β<1.
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))
    def wce_loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=beta)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)
    return wce_loss

def balanced_cross_entropy(beta=1+K.epsilon()):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))
    def balanced_cross_entropy_loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))
    return balanced_cross_entropy_loss

def focal(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))
        loss = focal_loss_with_logits(logits=logits, labels=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)
    return focal_loss

# combination
def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def wce_dice_loss(y_true, y_pred):
    wce_loss_func = wce()
    loss = wce_loss_func(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
