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
from generator import Generator
#from generator_channels import Generator_channels
from callbacks import model_callbacks
#from losses import (precision_loss, dice_loss, tversky_loss, focal_tversky_loss, bce_loss, 
#    wce, bce_dice_loss, wce_dice_loss, precision_loss, recall_loss, focal, balanced_cross_entropy)
                    #asym_unified_focal_loss, sym_unified_focal_loss,
                    #focal, wce, combo_loss)
from utils import get_lr_metric
from get_data import train_data, test_data
from opts import parse_opts
from model import isensee2017_model
from losses import precision_loss, dice_loss, tversky_loss, focal_tversky_loss, bce_loss, bce_dice_loss, wce_dice_loss



def train(opt):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    log_dir = opt.proj_dir + '/keras_seg_model/log'    
    # get data 
    tr_data, val_data = train_data(proj_dir=opt.proj_dir, crop_shape=opt.image_shape)
    print('tr data:', tr_data['img'].shape, tr_data['seg'].shape)
    print('val data:', val_data['img'].shape, val_data['seg'].shape)
    train_generator = Generator(
        images=tr_data['img'],
        labels=tr_data['seg'],
        batch_size=opt.batch_size,
        final_shape=opt.image_shape,
        blur_label=False,
        augment=True,
        elastic=True,
        shuffle=True)
    val_data = (val_data['img'], val_data['seg'].astype('float32'))
    
    # get model
    optimizer = Adam(learning_rate=opt.initial_lr)
    lr_metric = get_lr_metric(optimizer)
    UNet_model = isensee2017_model(
        input_shape=tuple([1] + list(opt.image_shape)), 
        n_labels=1, 
        n_base_filters=16)
    
    # loss function
    #loss = asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5)
    #loss = sym_unified_focal_loss()
    #loss = wce_dice_loss(beta=0.5)
    loss = wce_dice_loss
    #loss = balanced_cross_entropy()
    UNet_model.compile(optimizer=optimizer, loss=loss, metrics=[lr_metric])  
    
    # model callbacks
    csv_logger = CSVLogger(log_dir + '/logger.csv', append=True, separator=',')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, min_lr=.00001)
    #cbk = model_callbacks(UNet_model, log_dir=log_dir, val_data=val_data, n_labels=opt.n_labels)
    cbk = model_callbacks(model=UNet_model, RUN=1, dir_name=log_dir, val_data=val_data)
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    UNet_model.fit(
        train_generator,
        batch_size=opt.batch_size,
        epochs=opt.epochs,
        validation_data=val_data,
        callbacks=[cbk, csv_logger, reduce_lr],
        shuffle=True,
        #verbose=1,
        #max_queue_size=STEPS_PER_EPOCH*3, 
        #steps_per_epoch=math.floor(len(data["train"]["images"]) / BATCH_SIZE) 
        #workers=STEPS_PER_EPOCH*N_GPUS,
        use_multiprocessing=False)  


if __name__ == '__main__':

    opt = parse_opts()
    train(opt)






