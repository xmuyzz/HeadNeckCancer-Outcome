import os
import sys
import glob
import numpy as np
import math
from time import time
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
K.set_image_data_format('channels_first')
#sys.path.append('/home/bhkann/git-repositories/hn-petct-net/3d-unet-petct/files/')
#from model_3d.model import isensee2017_model
from generator import Generator
#from generator_channels import Generator_channels
from model_callbacks import model_callbacks
from losses import precision_loss, dice_loss, tversky_loss, focal_tversky_loss, bce_loss, bce_dice_loss, wce_dice_loss
from utils import get_lr_metric
import time


def train():

    # get data 
    train_data = get_train_val_data(train_img_dir, train_seg_dir)
    data_tune = data['tune']['images']
    #data_train = data['train']['images']
    val_data = (data_tune, data["tune"]["labels"].astype('float32'))
    train_generator = Generator(
        images=train_data['images'],
        labels=train_data['labels'],
        batch_size=opt.batch_size,
        image_shape=opt.image_shape,
        blur_label=False,
        augment=True,
        elastic=True,
        shuffle=True)
    
    # get model
    optimizer = Adam(lr=opt.initial_lr)
    lr_metric = get_lr_metric(optimizer)
    original_model = isensee2017_model(
        input_shape=input_shape, 
        n_labels=1, 
        n_base_filters=16)
    UNet_model.compile(optimizer=optimizer, loss=wce_dice_loss, metrics=[lr_metric])  
    
    # model callbacks
    csv_logger = CSVLogger(log_dir + '/logger.csv', append=True, separator=',')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, min_lr=.00001)
    cbk = model_callbacks(UNet_model, log_dir, val_data)
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [cbk, csv_logger, reduce_lr]
    
    UNet_model.fit(
        train_generator,
        batch_size=opt.batch_size,
        epochs=opt.epochs,
        validation_data=val_data,
        callbacks=callbacks,   
        shuffle=True,       
        #max_queue_size=STEPS_PER_EPOCH*3, 
        #steps_per_epoch=math.floor(len(data["train"]["images"]) / BATCH_SIZE) 
        #workers=STEPS_PER_EPOCH*N_GPUS,
        use_multiprocessing=False)  


if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    input_shape = tuple([1] + list(IMAGE_SHAPE))
    ### AUGMENTATION - SET THESE ALL TO FALSE FOR INITIAL TRAINING RUN ###
    log_dir = 






