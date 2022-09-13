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
from get_data import MyDataset



def main(data_dir, model_dir, batch_size, load_saved_model, img_shape):
    
    device = torch.device('cuda:0')

    # load data
    train_set = MyDataset(data_dir, step='train', img_shape=img_shape)
    test_set = MyDataset(data_dir, step='test', img_shape=img_shape)
    # train val split 80:20
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_train))
    train_idx, val_idx = indices[split:], indices[:split]
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True, 
        sampler=train_subsampler)
    val_loader = DataLoader(
        train_set,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True, 
        sampler=val_subsampler)
    
    # load model
    model = SwinUNETR(
        img_size=(160, 160, 64),
        in_channels=2,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0).to(device)
    if load_saved_model:
        saved_model = model_dir + '/saved_model.pth'
        model.load_state_dict(torch.load(saved_model))
    
    nb_train_batches = len(train_loader)
    nb_val_batches = len(val_loader)
    nb_iter = 0
    best_val_DC = 0.
    iters = list(range(1, 10))
    val_losses = []
    train_losses = []
    train_accuracy = []
    val_accuracy = []
    curr_epoch = 57
    best_val_DC = 0
    while curr_epoch < total_epoch:
        train_loss, val_loss = 0., 0.
        train_dsc1, val_dsc1 = 0., 0.
        train_dsc2, val_dsc2 = 0., 0.
        intersection1, union1 = 0, 0
        intersection2, union2 = 0, 0
        # TRAINING #
        model.train()
        train_data = iter(train_loader)
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
                pred1 = (preds==1.).type(torch.float32)
                pred2 = (preds==2.).type(torch.float32)
                gt1 = seg_gts[:, 1, :, :, :].type(torch.int8)
                gt2 = seg_gts[:, 2, :, :, :].type(torch.int8)
                inter1, u1 = compute_dice(pred1, gt1)
                inter2, u2 = compute_dice(pred2, gt2)
                intersection1 += inter1
                intersection2 += inter2
                union1 += u1
                union2 += u2
            train_dsc1 = intersection1 / union1
            train_dsc2 = intersection2 / union2
            # Increase iterations
            nb_iter += 1
            print('\rEpoch {}, iter {}/{}, loss {:.6f}'.format(curr_epoch+1, k+1, nb_train_batches, loss.item()),
                  end='')
            print('\rEpoch {}, iter {}/{}, loss {:.6f}'.format(curr_epoch+1, k+1, nb_train_batches, train_loss),
                  end='')
        print()
        intersection1, union1 = 0, 0
        intersection2, union2 = 0, 0

        # VALIDATION #
        model.eval()
        with torch.no_grad():
            val_data = iter(val_loader)
            for k in range(nb_val_batches):
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
                pred1 = (preds==1.).type(torch.float32)
                pred2 = (preds==2.).type(torch.float32)
                #pred_1 = (logits[:,0,:,:, :]>=0.5).type(torch.int8).cpu().to(device)
                #pred_2 = (logits[:,1,:,:, :]>=0.5).type(torch.int8).cpu().to(device)
                gt1 = seg_gts[:, 1, :, :, :].type(torch.int8)
                gt2 = seg_gts[:, 2, :, :, :].type(torch.int8)
                dsc1 = compute_dice_coef(pred1, gt1)
                dsc2 = compute_dice_coef(pred2, gt2)
                inter1, u1 = compute_dice_coef(pred1, gt1)
                inter2, u2 = compute_dice_coef(pred2, gt2)
                intersection1 += inter1
                intersection2 += inter2
                union1 += u1
                union2 += u2
            val_dsc1 = intersection1 / union1
            val_dsc2 = intersection2 / union2
        print('\nEpoch {}, Class 1 Train DC: {:.6f}, Class 2 Train DC: {:.6f}, Val Loss:{:.6f}, Class 1 Val DC:{:.6f}, Class 2 Val DC:{:.6f}'.format(curr_epoch + 1, train_dsc1, 
              train_dsc2, val_loss, val_dsc1, val_dsc2))
        if val_dsc1 + val_dsc2 > best_val_DC:
            torch.save(model.state_dict(), model_dir + '/SWIN_UNETR_final.pth')
            best_val_DC = val_dsc1 + val_dsc2
            print('Best validation DC reached on epoch. Saved model weights.')
        print('_'*50)

        # End of epoch
        curr_epoch  += 1


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    data_dir = proj_dir + '/nnUNetnn/UNet_raw_data_base/nnUNet_raw_data/Task506_PN_crop'
    model_dir = proj_dir + '/seg_model/Task506/saved_model'
    output_dir = proj_dir + '/seg_model/Task506/output'
    load_saved_model = False
    batch_size = 32
    img_shape = (160, 160, 64)
    num_workers = 1
    total_epoch = 200
    lr = 0.0001
    device = torch.device("cuda:0")

    main(data_dir, model_dir, batch_size, load_saved_model, img_shape)



