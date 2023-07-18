import torch
from torch import nn
import torch.nn.functional as tnf
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import sklearn
from sklearn.model_selection import KFold
from volumentations import *
import SimpleITK as sitk
import nibabel as nib
from monai import transforms
from monai.transforms import AsDiscrete, Activations
import glob
import numpy as np
from monai.transforms import (AddChannel, AsChannelFirst, EnsureChannelFirst, RepeatChannel,
    ToTensor, RemoveRepeatedChannel, EnsureType, Compose, CropForeground, LoadImage,
    Orientation, RandSpatialCrop, Spacing, Resize, ScaleIntensity, RandRotate, RandZoom,
    RandGaussianNoise, RandGaussianSharpen, RandGaussianSmooth, RandFlip, Rotate90, RandRotate90, 
    EnsureType, RandAffine)


class MyDataset(Dataset):
    """
    create a Dataset for dataloader
    args:
        data_dir {path} -- data dir
        step {str} -- train, test
        transform {class} -- train transform, val transform, test transform
    returns:
        img, seg
    Raise issues:
        none
    """
    def __init__(self, proj_dir, step, transform):
        self.proj_dir = proj_dir
        self.step = step
        self.transform = transform
        self.img_paths = []
        self.seg_paths = []
        # Load data index
        tr_img_path = proj_dir + '/HKTR_TCIA_DFCI/TOT/crop_img_160/*nii.gz'
        tr_seg_path = proj_dir + '/HKTR_TCIA_DFCI/TOT/crop_seg_160/*nii.gz'
        if step == 'train':
            self.img_paths = [i for i in sorted(glob.glob(tr_img_path))]
            self.seg_paths = [i for i in sorted(glob.glob(tr_seg_path))]
        elif step == 'test':
            self.img_paths = [i for i in glob.glob(proj_dir + '/TCIA/img_crop_160/*nrrd')]
            self.seg_paths = [i for i in glob.glob(proj_dir + '/TCIA/seg_pn_crop_160/*nrrd')]
        print('Succesfully loaded {} dataset.'.format(step) + ' '*50)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_dir = self.img_paths[idx]
        seg_dir = self.seg_paths[idx]
        #print(img_dir)
        #print(seg_dir)
        img = sitk.ReadImage(img_dir)
        seg = sitk.ReadImage(seg_dir)
        img_arr = sitk.GetArrayFromImage(img)
        seg_arr = sitk.GetArrayFromImage(seg)
        img = np.expand_dims(img_arr, axis=3)
        seg = np.expand_dims(seg_arr, axis=3)
        #print(img.shape)
        #print(seg.shape)
        img = img.transpose(3, 0, 1, 2).astype(np.float32)
        seg = seg.transpose(3, 0, 1, 2).astype(np.float32)
        # augmentation
        img = self.transform(img)
        seg = self.transform(seg)

        return img, seg


def get_data_loader(proj_dir, batch_size):

    # augmentation
    train_transform = Compose([
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        #Resized(spatial_size=(96, 96, 96)),
        RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
        RandGaussianSharpen(),
        RandGaussianSmooth(),
        RandAffine(prob=0.5, translate_range=10),
        RandFlip(prob=0.5, spatial_axis=None),
        RandRotate(prob=0.5, range_x=5, range_y=5, range_z=5),
        EnsureType(data_type='tensor')])
    val_transform = Compose([
        #AddChannel,
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        EnsureType(data_type='tensor')])

    # load data
    train_set = MyDataset(proj_dir=proj_dir, step='train', transform=train_transform)
    #test_set = MyDataset(data_dir, step='test', transform=test_transform)
    # train val split 80:20
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_train))
    train_idx, val_idx = indices[split:], indices[:split]
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)
    train_dl = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        sampler=train_subsampler)
    val_dl = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        sampler=val_subsampler)

    return train_dl, val_dl






