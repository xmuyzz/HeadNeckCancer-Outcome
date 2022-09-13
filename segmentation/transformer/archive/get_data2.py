import torch
from torch import nn
import torch.nn.functional as tnf
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import sklearn
from sklearn.model_selection import KFold
from volumentations import *
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


def get_augmentation_train(img_shape):
#    return Compose([
#        Rotate((-8, 8), (-8, 8), (-8, 8), p=0.5),
#        RandomCrop(shape=img_shape, p=1.0),
#        ElasticTransform((0, 0.25), interpolation=2, p=0.1),
#        GaussianNoise(var_limit=(0, 5), p=0.2),
#        #RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
#        ], p=1.0)
    return Compose([
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

def get_augmentation_test(img_shape):
#    return Compose([RandomCrop(shape=img_shape, p=1.0)], p=1.0)
    return Compose([
        #AddChannel,
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        #RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
        #RandAffine(prob=0.5, translate_range=10),
        EnsureType(data_type='tensor')])

def load_data(img_path, seg_path, step, img_shape, do_aug):
    """
    load data, transform data, data augmentation
    """
    img = np.expand_dims(nib.load(img_path).get_fdata(), axis=3)
    seg = nib.load(seg_path).get_fdata()
    #seg = keras.utils.to_categorical(seg, num_classes=3, dtype='float32')
    # data augmentation
    if do_aug:
        if step == 'train':
          aug = get_augmentation_train(img_shape)
        elif step == 'test':
          aug = get_augmentation_test(img_shape)
        data = {'img': img, 'seg': seg}
        aug_data = aug(**data)
        img, seg = aug_data['img'], aug_data['seg']
        img, seg = img.transpose(3, 0, 1, 2).astype(np.float32), mask.transpose(3, 0, 1, 2).astype(np.float32)
        return img, seg
    else:
        img, mask = img.transpose(3, 0, 1, 2), mask.transpose(3, 0, 1, 2)
        return img, seg


class MyDataset(Dataset):
    """
    create a Dataset for dataloader
    """
    def __init__(self, data_dir, step, img_shape):
        self.data_dir = data_dir
        self.step = step
        self.img_shape = img_shape
        self.img_paths = []
        self.seg_paths = []
        # Load data index
        if step == 'train':
            self.img_paths = [path for path in glob.glob(data_dir + '/imagesTr/' + '*nii.gz')]
            self.seg_paths = [path for path in glob.glob(data_dir + '/labelsTr/' + '*nii.gz')]
        elif step == 'test':
            self.img_paths = [path for path in glob.glob(data_dir + '/imagesTs/' + '*nii.gz')]
            self.seg_paths = [path for path in glob.glob(data_dir + '/labelsTs/' + '*nii.gz')]
        print('Succesfully loaded {} dataset.'.format(step) + ' '*50)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        seg_path = self.seg_paths[idx]
        return load_data(img_path, seg_path, self.step, self.img_shape, True)





