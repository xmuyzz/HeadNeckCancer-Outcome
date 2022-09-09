import os
import tensorflow as tf
from tensorflow import keras
import torch
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
from functools import partial
from IPython.display import clear_output
import matplotlib.pyplot as plt



def get_augmentation_train(patch_size):
    return Compose([
        Rotate((-8, 8), (-8, 8), (-8, 8), p=0.5),
        RandomCrop(shape = (160, 160, 64), p = 1.0),
        ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        #GaussianNoise(var_limit=(0, 5), p=0.2),
        #RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
        ], p=1.0)
 

def get_augmentation_test(patch_size):
    return Compose([RandomCrop(shape = (160, 160, 64), p = 1.0)], p=1.0)




def load_data(root_dir, filename, split, aug):
  filename = filename.upper()
  dir_img = os.path.join(root_dir, "imagesTr")
  dir_seg = os.path.join(root_dir, "labelsTr")
  if split == 'test':
    dir_img = os.path.join(root_dir, "imagesTs")
    dir_seg = os.path.join(root_dir, "labelsTs")
  filename_ct = os.path.join(dir_img,  filename[:-7] + '_0000.nii.gz')
  filename_pt = os.path.join(dir_img, filename[:-7] + '_0001.nii.gz')
  filename_mask = os.path.join(dir_seg, filename[:-7] + '.nii.gz')
  ct_img = np.expand_dims(nib.load(filename_ct).get_fdata(), axis = 3)
  pet_img = np.expand_dims(nib.load(filename_pt).get_fdata(), axis = 3)
  mask = nib.load(filename_mask).get_fdata()
  # mask_1 = np.expand_dims((mask==1.).astype(np.float32), axis = 3)
  # mask_2 = np.expand_dims((mask==2.).astype(np.float32), axis = 3)
  # mask = np.concatenate([mask_0, mask_1, mask_2], axis=3)
  mask = keras.utils.to_categorical(mask, num_classes=3, dtype='float32')
  img = np.concatenate((ct_img, pet_img), axis = 3)
  if aug: 
    if(split == 'train'):
      aug = get_augmentation_train((160, 160, 64))
    else:
      aug = get_augmentation_test((160, 160, 64))
    data = {'image': img, 'mask': mask}
    aug_data = aug(**data)
    img, mask = aug_data['image'], aug_data['mask']
    img, mask = img.transpose(3 , 0, 1, 2).astype(np.float32), mask.transpose(3, 0, 1, 2).astype(np.float32)
    return img, mask
  else:
    img, mask = img.transpose(3 , 0, 1, 2), mask.transpose(3, 0, 1, 2)
    return img, mask


class HEKDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.image_paths = []
        # Load data index
        if split == 'train':
          for path in os.listdir(os.path.join(root_dir, "labelsTr")):
                  self.image_paths.append(path)
        else:
          for path in os.listdir(os.path.join(root_dir, "labelsTs")):
              self.image_paths.append(path)
        print('Succesfully loaded {} dataset.'.format(split) + ' '*50)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        filename = self.image_paths[idx]
        return load_data(self.root_dir, filename, self.split, True)


def compute_dice_coef(inputs, targets, smooth = 1e-7):
  #inputs = tnf.softmax(inputs)
  inputs = inputs.view(-1)
  targets = targets.view(-1)
  intersection = (inputs * targets).sum()
  dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
  return (2.*intersection), (inputs.sum() + targets.sum())


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    tensor = tensor[:, 1:, :, :, :]
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    input = flatten(input)
    target = flatten(target)
    target = target.float()
    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect
    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))



def trainer():
    root_dir = "/content/gdrive/MyDrive/Summer Programs/HECKTOR2022/DATA/nnUNet_raw_data_base/nnUNet_raw_data/Task501_Hecktor"
    batch_size = 1
    num_workers = 2
    total_epoch = 200
    lr = 0.0001
    device = torch.device("cuda:0")

    model = SwinUNETR(
        img_size=(160, 160, 64),
        in_channels=2,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0).to(device)
    # model = UNETR(
    #     img_size=(160, 160, 64),
    #     in_channels=2,
    #     out_channels=2,
    #     feature_size=48
    # ).to(device)
    model.load_state_dict(torch.load('/content/gdrive/MyDrive/Summer Programs/HECKTOR2022/Saved Models/SWIN_UNETR_new.pth'))
    train_set = HEKDataset(root_dir, split='train')
    test_set = HEKDataset(root_dir, split='test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)
    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,)
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True, 
        sampler=train_subsampler)
    val_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True, 
        sampler=val_subsampler)

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

    while curr_epoch  < total_epoch:
        train_loss, val_loss = 0., 0.
        train_dsc_1, val_dsc_1 = 0., 0.
        train_dsc_2, val_dsc_2 = 0., 0.
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
                preds = torch.argmax(logits, axis = 1)
                pred_1 = (preds==1.).type(torch.float32)
                pred_2 = (preds==2.).type(torch.float32)
                gt_1 = seg_gts[:,1,:,:, :].type(torch.int8)
                gt_2 = seg_gts[:,2,:,:, :].type(torch.int8)
                inter1, u1 = compute_dice_coef(pred_1, gt_1)
                inter2, u2 = compute_dice_coef(pred_2, gt_2)
                intersection1 += inter1
                intersection2 += inter2
                union1 += u1
                union2 += u2
            train_dsc_1 = intersection1 / union1
            train_dsc_2 = intersection2 / union2

            # Increase iterations
            nb_iter += 1
            print('\rEpoch {}, iter {}/{}, loss {:.6f}'.format(curr_epoch+1, k+1, nb_train_batches, loss.item()),
                  end='')
        print('\rEpoch {}, iter {}/{}, loss {:.6f}'.format(curr_epoch+1, k+1, nb_train_batches, train_loss),
                  end='')
        print()
        intersection1, union1 = 0,0
        intersection2, union2 = 0,0
        ##############
        # VALIDATION #
        ##############
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
                preds = torch.argmax(logits, axis = 1)
                pred_1 = (preds==1.).type(torch.float32)
                pred_2 = (preds==2.).type(torch.float32)
                #pred_1 = (logits[:,0,:,:, :]>=0.5).type(torch.int8).cpu().to(device)
                #pred_2 = (logits[:,1,:,:, :]>=0.5).type(torch.int8).cpu().to(device)
                gt_1 = seg_gts[:,1,:,:, :].type(torch.int8)
                gt_2 = seg_gts[:,2,:,:, :].type(torch.int8)
                dsc_1 = compute_dice_coef(pred_1, gt_1)
                dsc_2 = compute_dice_coef(pred_2, gt_2)
                inter1, u1 = compute_dice_coef(pred_1, gt_1)
                inter2, u2 = compute_dice_coef(pred_2, gt_2)
                intersection1 += inter1
                intersection2 += inter2
                union1 += u1
                union2 += u2
            val_dsc_1 = intersection1 / union1
            val_dsc_2 = intersection2 / union2
        print('\nEpoch {}, Class 1 Train DC: {:.6f}, Class 2 Train DC: {:.6f}, Val Loss:{:.6f}, Class 1 Val DC:{:.6f}, Class 2 Val DC:{:.6f}'.format(curr_epoch + 1, train_dsc_1, train_dsc_2, val_loss, val_dsc_1, val_dsc_2))
        if val_dsc_1 + val_dsc_2 > best_val_DC:
            torch.save(model.state_dict(), '/content/gdrive/MyDrive/Summer Programs/HECKTOR2022/Saved Models/SWIN_UNETR_final.pth')
            best_val_DC = val_dsc_1 + val_dsc_2
            print('Best validation DC reached on epoch. Saved model weights.')
        print('_'*50)

        # End of epoch
        curr_epoch  += 1

