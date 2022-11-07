from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
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
    return 1 - tversky(y_true, y_pred)

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
        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)
    return focal_loss

# combination
def bce_dice_loss(y_true, y_pred):
    #loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def wce_dice_loss(y_true, y_pred):
   #wce_loss_func = wce()
   loss = wce()(y_true, y_pred) + dice_loss(y_true, y_pred)
   return loss

#def wce_dice_loss(beta=0.5):
#    def loss_function(y_true, y_pred):
#        loss1 = wce(beta=beta)(y_true, y_pred)
#        loss2 = dice_loss(y_true, y_pred)
#        return loss1 + loss2
#    return loss_function


##-----------------------------------------------------------------------
## unified focal loss
## https://github.com/mlyg/unified-focal-loss
##-----------------------------------------------------------------------
#"""
#The Unified Focal loss is a new compound loss function that unifies Dice-based and cross 
#entropy-based loss functions into a single framework. By incorporating ideas from focal and asymmetric losses, the        Unified Focal loss is designed to handle class imbalance. It can be shown that all Dice and cross entropy based loss        functions described above are special cases of the Unified Focal loss
#"""
#
## Helper function to enable loss function to be flexibly used for 
## both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
#def identify_axis(shape):
#    # Three dimensional
#    if len(shape) == 5 : return [1,2,3]
#    # Two dimensional
#    elif len(shape) == 4 : return [1,2]
#    # Exception - Unknown
#    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
#
#################################
##           Dice loss          #
#################################
#def dice_loss(delta=0.5, smooth=0.000001):
#    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.
#    
#    Parameters
#    ----------
#    delta : float, optional
#        controls weight given to false positive and false negatives, by default 0.5
#    smooth : float, optional
#        smoothing constant to prevent division by zero errors, by default 0.000001
#    """
#    def loss_function(y_true, y_pred):
#        axis = identify_axis(y_true.get_shape())
#        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
#        tp = K.sum(y_true * y_pred, axis=axis)
#        fn = K.sum(y_true * (1-y_pred), axis=axis)
#        fp = K.sum((1-y_true) * y_pred, axis=axis)
#        # Calculate Dice score
#        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
#        # Average class scores
#        dice_loss = K.mean(1-dice_class)
#
#        return dice_loss
#        
#    return loss_function
#
#
#################################
##         Tversky loss         #
#################################
#def tversky_loss(delta=0.7, smooth=0.000001):
#    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
#	Link: https://arxiv.org/abs/1706.05721
#    Parameters
#    ----------
#    delta : float, optional
#        controls weight given to false positive and false negatives, by default 0.7
#    smooth : float, optional
#        smoothing constant to prevent division by zero errors, by default 0.000001
#    """
#    def loss_function(y_true, y_pred):
#        axis = identify_axis(y_true.get_shape())
#        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
#        tp = K.sum(y_true * y_pred, axis=axis)
#        fn = K.sum(y_true * (1-y_pred), axis=axis)
#        fp = K.sum((1-y_true) * y_pred, axis=axis)
#        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
#        # Average class scores
#        tversky_loss = K.mean(1-tversky_class)
#
#        return tversky_loss
#
#    return loss_function
#
#################################
##       Dice coefficient       #
#################################
#def dice_coefficient(delta=0.5, smooth=0.000001):
#    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, is a statistical tool which measures the similarity between two sets of data.
#    Parameters
#    ----------
#    delta : float, optional
#        controls weight given to false positive and false negatives, by default 0.5
#    smooth : float, optional
#        smoothing constant to prevent division by zero errors, by default 0.000001
#    """
#    def loss_function(y_true, y_pred):
#        axis = identify_axis(y_true.get_shape())
#        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
#        tp = K.sum(y_true * y_pred, axis=axis)
#        fn = K.sum(y_true * (1-y_pred), axis=axis)
#        fp = K.sum((1-y_true) * y_pred, axis=axis)
#        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
#        # Average class scores
#        dice = K.mean(dice_class)
#
#        return dice
#
#    return loss_function
#
#################################
##          Combo loss          #
#################################
#def combo_loss(alpha=0.5, beta=0.5):
#    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
#    Link: https://arxiv.org/abs/1805.02798
#    Parameters
#    ----------
#    alpha : float, optional
#        controls weighting of dice and cross-entropy loss., by default 0.5
#    beta : float, optional
#        beta > 0.5 penalises false negatives more than false positives., by default 0.5
#    """
#    def loss_function(y_true, y_pred):
#        dice = dice_coefficient()(y_true, y_pred)
#        axis = identify_axis(y_true.get_shape())
#        # Clip values to prevent division by zero error
#        epsilon = K.epsilon()
#        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#        cross_entropy = -y_true * K.log(y_pred)
#
#        if beta is not None:
#            beta_weight = np.array([beta, 1-beta])
#            cross_entropy = beta_weight * cross_entropy
#        # sum over classes
#        cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
#        if alpha is not None:
#            combo_loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
#        else:
#            combo_loss = cross_entropy - dice
#        return combo_loss
#
#    return loss_function
#
#
#def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
#    """
#    Focal Tversky loss 
#    A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
#    Link: https://arxiv.org/abs/1810.07842
#    Parameters
#    ----------
#    gamma : float, optional
#        focal parameter controls degree of down-weighting of easy examples, by default 0.75
#    """
#    def loss_function(y_true, y_pred):
#        # Clip values to prevent division by zero error
#        epsilon = K.epsilon()
#        y_pred = K.clip(y_pred, epsilon, 1. - epsilon) 
#        axis = identify_axis(y_true.get_shape())
#        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
#        tp = K.sum(y_true * y_pred, axis=axis)
#        fn = K.sum(y_true * (1-y_pred), axis=axis)
#        fp = K.sum((1-y_true) * y_pred, axis=axis)
#        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
#        # Average class scores
#        focal_tversky_loss = K.mean(K.pow((1-tversky_class), gamma))
#	
#        return focal_tversky_loss
#
#    return loss_function
#
#
#def focal_loss(alpha=None, gamma_f=2.):
#    """
#    Focal loss  
#    Focal loss is used to address the issue of the class imbalance problem. A modulation term applied to the Cross-Entropy loss function.
#    Parameters
#    ----------
#    alpha : float, optional
#        controls relative weight of false positives and false negatives. alpha > 0.5 penalises false negatives more than false positives, by default None
#    gamma_f : float, optional
#        focal parameter controls degree of down-weighting of easy examples, by default 2.
#    """
#    def loss_function(y_true, y_pred):
#        axis = identify_axis(y_true.get_shape())
#        # Clip values to prevent division by zero error
#        epsilon = K.epsilon()
#        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#        cross_entropy = -y_true * K.log(y_pred)
#
#        if alpha is not None:
#            alpha_weight = np.array(alpha, dtype=np.float32)
#            focal_loss = alpha_weight * K.pow(1 - y_pred, gamma_f) * cross_entropy
#        else:
#            focal_loss = K.pow(1 - y_pred, gamma_f) * cross_entropy
#
#        focal_loss = K.mean(K.sum(focal_loss, axis=[-1]))
#        return focal_loss
#        
#    return loss_function
#
#
#def symmetric_focal_loss(delta=0.7, gamma=2.):
#    """
#    Symmetric Focal loss
#    Parameters
#    ----------
#    delta : float, optional
#        controls weight given to false positive and false negatives, by default 0.7
#    gamma : float, optional
#        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
#    """
#    def loss_function(y_true, y_pred):
#
#        axis = identify_axis(y_true.get_shape())  
#
#        epsilon = K.epsilon()
#        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#        cross_entropy = -y_true * K.log(y_pred)
#        #calculate losses separately for each class
#        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
#        back_ce =  (1 - delta) * back_ce
#
#        fore_ce = K.pow(1 - y_pred[:,:,:,1], gamma) * cross_entropy[:,:,:,1]
#        fore_ce = delta * fore_ce
#
#        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))
#
#        return loss
#
#    return loss_function
#
#
#def symmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
#    """
#    Symmetric Focal Tversky loss 
#    This is the implementation for binary segmentation.
#    Parameters
#    ----------
#    delta : float, optional
#        controls weight given to false positive and false negatives, by default 0.7
#    gamma : float, optional
#        focal parameter controls degree of down-weighting of easy examples, by default 0.75
#    """
#    def loss_function(y_true, y_pred):
#        # Clip values to prevent division by zero error
#        epsilon = K.epsilon()
#        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#
#        axis = identify_axis(y_true.get_shape())
#        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
#        tp = K.sum(y_true * y_pred, axis=axis)
#        fn = K.sum(y_true * (1-y_pred), axis=axis)
#        fp = K.sum((1-y_true) * y_pred, axis=axis)
#        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)
#
#        #calculate losses separately for each class, enhancing both classes
#        back_dice = (1-dice_class[:,0]) * K.pow(1-dice_class[:,0], -gamma) 
#        fore_dice = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], -gamma) 
#
#        # Average class scores
#        loss = K.mean(tf.stack([back_dice,fore_dice],axis=-1))
#        return loss
#
#    return loss_function
#
#
#def asymmetric_focal_loss(delta=0.7, gamma=2.):
#    """
#    Asymmetric Focal loss
#    For Imbalanced datasets
#    Parameters
#    ----------
#    delta : float, optional
#        controls weight given to false positive and false negatives, by default 0.7
#    gamma : float, optional
#        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
#    """
#    def loss_function(y_true, y_pred):
#        axis = identify_axis(y_true.get_shape())  
#
#        epsilon = K.epsilon()
#        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#        cross_entropy = -y_true * K.log(y_pred)
#
#        #calculate losses separately for each class, only suppressing background class
#        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
#        back_ce =  (1 - delta) * back_ce
#
#        fore_ce = cross_entropy[:,:,:,1]
#        fore_ce = delta * fore_ce
#
#        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))
#
#        return loss
#
#    return loss_function
#
#
#def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
#    """
#    Asymmetric Focal Tversky loss
#    This is the implementation for binary segmentation.
#    Parameters
#    ----------
#    delta : float, optional
#        controls weight given to false positive and false negatives, by default 0.7
#    gamma : float, optional
#        focal parameter controls degree of down-weighting of easy examples, by default 0.75
#    """
#    def loss_function(y_true, y_pred):
#        # Clip values to prevent division by zero error
#        epsilon = K.epsilon()
#        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#
#        axis = identify_axis(y_true.get_shape())
#        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
#        tp = K.sum(y_true * y_pred, axis=axis)
#        fn = K.sum(y_true * (1-y_pred), axis=axis)
#        fp = K.sum((1-y_true) * y_pred, axis=axis)
#        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)
#
#        #calculate losses separately for each class, only enhancing foreground class
#        back_dice = (1-dice_class[:,0]) 
#        fore_dice = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], -gamma) 
#
#        # Average class scores
#        loss = K.mean(tf.stack([back_dice,fore_dice],axis=-1))
#        return loss
#
#    return loss_function
#
#
#def sym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
#    """
#    Symmetric Unified Focal loss
#    The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
#    Parameters
#    ----------
#    weight : float, optional
#        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
#    delta : float, optional
#        controls weight given to each class, by default 0.6
#    gamma : float, optional
#        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
#    """
#    def loss_function(y_true, y_pred):
#      symmetric_ftl = symmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true, y_pred)
#      symmetric_fl = symmetric_focal_loss(delta=delta, gamma=gamma)(y_true, y_pred)
#      if weight is not None:
#        return (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)  
#      else:
#        return symmetric_ftl + symmetric_fl
#
#    return loss_function
#
#
#def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
#    """
#    Asymmetric Unified Focal loss
#    The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
#    Parameters
#    ----------
#    weight : float, optional
#        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
#    delta : float, optional
#        controls weight given to each class, by default 0.6
#    gamma : float, optional
#        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
#    """
#    def loss_function(y_true, y_pred):
#      asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true, y_pred)
#      asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true, y_pred)
#      if weight is not None:
#        return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)  
#      else:
#        return asymmetric_ftl + asymmetric_fl
#
#    return loss_function
#
#
#
