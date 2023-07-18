import sys
import os
import pydicom
import glob
import SimpleITK as sitk
from dcm_to_nrrd import dcm_to_nrrd
from rtstruct_to_nrrd import rtstruct_to_nrrd
from combine_structures import combine_structures
from interpolate import interpolate
from crop import crop_roi, crop_top, crop_top_image_only
from registration import nrrd_reg_rigid
import SimpleITK as sitk
import shutil
import nibabel as nib
import numpy as np
from scipy import ndimage


def crop(proj_dir):
    
    """
    With TOP-CROP HPC ### NEED TO RUN FOR image_crop, image_crop_p, and image_crop_n  
    WILL ONLY WORK WITH SPACING = 1,1,3
    """

    crop_shape = (76, 160, 160) #x,y,z
    img_in_dir = proj_dir + '/imagesTr_'
    seg_in_dir = proj_dir + '/labelsTr_'
    img_crop_dir = proj_dir + '/imagesTr'
    seg_crop_dir = proj_dir + '/labelsTr'
    if not os.path.exists(img_crop_dir):
        os.makedirs(img_crop_dir)
    if not os.path.exists(seg_crop_dir):
        os.makedirs(seg_crop_dir)
    img_dirs = [i for i in sorted(glob.glob(img_in_dir + '/*nii.gz'))]
    seg_dirs = [i for i in sorted(glob.glob(seg_in_dir + '/*nii.gz'))]
    # registration for image and seg
    img_ids = []
    count = 0
    for img_dir, seg_dir in zip(img_dirs, seg_dirs):
        img_id = img_dir.split('/')[-1]
        seg_id = seg_dir.split('/')[-1]
        count += 1
        print(count, img_id, seg_id)
        img_arr = nib.load(img_dir).get_data()
        seg_arr = nib.load(seg_dir).get_data() 
        ## Return top 25 rows of 3D volume, centered in x-y space / start at anterior (y=0)?
        #print('image_arr shape:', image_arr.shape)
        c, y, x = img_arr.shape
        ## Get center of mass to center the crop in Y plane
        mask_arr = np.copy(img_arr) 
        mask_arr[mask_arr > -500] = 1
        mask_arr[mask_arr <= -500] = 0
        mask_arr[mask_arr >= -500] = 1 
        #print('mask_arr min and max:', np.amin(mask_arr), np.amax(mask_arr))
        centermass = ndimage.measurements.center_of_mass(mask_arr) # z,x,y   
        cpoint = c - crop_shape[2]//2
        #print('cpoint, ', cpoint)
        centermass = ndimage.measurements.center_of_mass(mask_arr[cpoint, :, :])   
        #print('center of mass: ', centermass)
        startx = int(centermass[0] - crop_shape[0]//2)
        starty = int(centermass[1] - crop_shape[1]//2)      
        #startx = x//2 - crop_shape[0]//2       
        #starty = y//2 - crop_shape[1]//2
        startz = int(c - crop_shape[2])
        #print('start X, Y, Z: ', startx, starty, startz)
        # cut bottom slices
        #image_arr = image_arr[30:, :, :]
        #label_arr = label_arr[30:, :, :]
        if startz < 0:
            img_arr = np.pad(
                img_arr,
                ((abs(startz)//2, abs(startz)//2), (0, 0), (0, 0)), 
                'constant', 
                constant_values=-1024)
            seg_arr = np.pad(
                seg_arr,
                ((abs(startz)//2, abs(startz)//2), (0, 0), (0, 0)), 
                'constant', 
                constant_values=0)
            img_arr_crop = img_arr[0:crop_shape[2], starty:starty+crop_shape[1], startx:startx+crop_shape[0]]
            seg_arr_crop = seg_arr[0:crop_shape[2], starty:starty+crop_shape[1], startx:startx+crop_shape[0]]
        else:
            img_arr_crop = img_arr[0:crop_shape[2], starty:starty+crop_shape[1], startx:startx+crop_shape[0]]
            seg_arr_crop = seg_arr[0:crop_shape[2], starty:starty+crop_shape[1], startx:startx+crop_shape[0]]
        # save nii
        print(img_arr_crop.shape)
        print(seg_arr_crop.shape)
        img_fn = img_crop_dir + '/' + img_id
        img = nib.Nifti1Image(img_arr_crop, np.eye(4))
        nib.save(img, img_fn)
        seg_fn = seg_crop_dir + '/' + seg_id
        seg = nib.Nifti1Image(seg_arr_crop, np.eye(4))
        nib.save(seg, seg_fn)


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/hecktor2022/DATA2/nnUNet_raw_data_base/nnUNet_raw_data/Task500_ToySet'
    crop(proj_dir)


    
