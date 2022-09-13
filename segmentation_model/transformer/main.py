import os
#import tensorflow as tf
#from tensorflow import keras
import numpy as np
import torch
import glob
from torch import nn
import torch.nn.functional as tnf
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.optim as optim
import sklearn
from sklearn.model_selection import KFold
from volumentations import *
import nibabel as nib
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import AsDiscrete, Activations
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR, UNETR
from monai import data
from monai.data import decollate_batch
from utils import compute_dice, compute_per_channel_dice, flatten
from get_data import get_data_loader
from losses import (AsymmetricUnifiedFocalLoss, AsymmetricFocalTverskyLoss, 
    AsymmetricFocalLoss, DiceLoss, _AbstractDiceLoss)


def main(data_dir, model_dir, batch_size, load_saved_model, img_shape):
    
    train_dl, val_dl = get_data_loader(data_dir, batch_size) 
    # load model
    model = SwinUNETR(
        img_size=(160, 160, 64),
        in_channels=1,
        out_channels=2,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0).to(device)
    if load_saved_model:
        model.load_state_dict(torch.load(model_dir + '/saved_model.pth'))
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #seg_loss = DiceLoss(normalization='sigmoid')
    seg_loss = AsymmetricUnifiedFocalLoss()

    nb_train_batches = len(train_dl)
    nb_val_batches = len(val_dl)
    nb_iter = 0
    iters = list(range(1, 10))
    val_losses = []
    train_losses = []
    train_accuracy = []
    val_accuracy = []
    curr_epoch = 0
    best_val_DC = 0.8
    while curr_epoch < total_epoch:
        train_loss, val_loss = 0., 0.
        train_dsc, val_dsc = 0., 0.
        intersection, union = 0, 0
        # TRAINING #
        model.train()
        train_data = iter(train_dl)
        for k in range(nb_train_batches):
            imgs, seg_gts = train_data.next()
            #print(imgs.size(), seg_gts.size(), device)
            imgs, seg_gts = imgs.to(device), seg_gts.to(device)
            # Forward pass
            logits = model(imgs)
            #print(logits.size, imgs.size)
            loss = seg_loss(logits, seg_gts)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / nb_train_batches
            train_losses.append(train_loss)
            with torch.no_grad():
                preds = torch.argmax(logits, axis=1)
                pred = (preds==1.).type(torch.float32)
                gt = seg_gts.type(torch.int8)
                inter, u = compute_dice(pred, gt)
                intersection += inter
                union += u
                print(inter)
                print(u)
            train_dsc = intersection / union
            # Increase iterations
            nb_iter += 1
            print('\rEpoch {}, iter {}/{}, loss {:.6f}'.format(curr_epoch+1, k+1, nb_train_batches, loss.item()),
                  end='')
            print('\rEpoch {}, iter {}/{}, loss {:.6f}'.format(curr_epoch+1, k+1, nb_train_batches, train_loss),
                  end='')
        print()
        intersection, union = 0, 0

        # VALIDATION #
        model.eval()
        with torch.no_grad():
            val_data = iter(val_dl)
            for k in range(nb_val_batches):
                print(k)
                # Loads data
                imgs, seg_gts = val_data.next()
                imgs, seg_gts = imgs.to(device), seg_gts.to(device)
                # Forward pass
                logits = model(imgs)
                val_loss += seg_loss(logits, seg_gts).item() / nb_val_batches
                val_losses.append(val_loss)
                # Std out
                print('\rValidation iter {}/{}'.format(k+1, nb_val_batches), end='')
                # Compute segmentation metric
                preds = torch.argmax(logits, axis=1)
                pred = (preds==1.).type(torch.float32)
                gt = seg_gts.type(torch.int8)
                dsc = compute_dice(pred, gt)
                inter, u = compute_dice(pred, gt)
                intersection += inter
                union += u
                print(inter)
                print(u)
            val_dsc = intersection 
        print('\nEpoch {}, Train DC: {:.6f}, Val Loss:{:.6f}, Val DC:{:.6f} '.format(curr_epoch+1, train_dsc, 
              val_loss, val_dsc))
        if val_dsc > best_val_DC:
            torch.save(model.state_dict(), model_dir + '/SWIN_UNETR_final.pth')
            best_val_DC = val_dsc
            print('Best validation DC reached on epoch. Saved model weights.')
        print('_'*50)

        # End of epoch
        curr_epoch  += 1


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    data_dir = proj_dir + '/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task506_PN_crop'
    model_dir = proj_dir + '/seg_model/Task506/saved_model'
    output_dir = proj_dir + '/seg_model/Task506/output'
    load_saved_model = False
    batch_size = 1
    img_shape = (160, 160, 64)
    num_workers = 1
    total_epoch = 200
    lr = 0.0001
    device = torch.device("cuda:0")

    main(data_dir, model_dir, batch_size, load_saved_model, img_shape)



