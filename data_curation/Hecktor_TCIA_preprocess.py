import sys
import os
import pydicom
import glob
import SimpleITK as sitk
import pandas as pd
import numpy as np
from dcm_to_nrrd import dcm_to_nrrd
from rtstruct_to_nrrd import rtstruct_to_nrrd
from combine_structures import combine_structures
from interpolate import interpolate
from crop_image import crop_top, crop_top_image_only, crop_full_body
from registration import nrrd_reg_rigid
import SimpleITK as sitk
import shutil
import nibabel as nib


def registration(proj_dir, root_dir, image_format):
    """
    Rigid Registration - followed by top crop
    """
    print('\n------start registration--------')
    img_raw_dir = proj_dir + '/raw_img'
    seg_raw_dir = proj_dir + '/raw_seg'
    img_reg_dir = proj_dir + '/reg_img'
    seg_reg_dir = proj_dir + '/reg_seg'
    if not os.path.exists(img_reg_dir):
        os.makedirs(img_reg_dir)
    if not os.path.exists(seg_reg_dir):
        os.makedirs(seg_reg_dir)
    img_dirs = [i for i in sorted(glob.glob(img_raw_dir + '/*' + image_format))]
    seg_dirs = [i for i in sorted(glob.glob(seg_raw_dir + '/*' + image_format))]
    img_ids = []
    count = 0
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        #print(img_id)
        for seg_dir in seg_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            #print(seg_id)
            if img_id == seg_id:
                img_ids.append(img_id)
                count += 1
                print(count, img_id)
                # --- interpolation for image and seg to 1x1x3 ---
                # interpolate images
                img_interp = interpolate(
                    patient_id=img_id, 
                    path_to_nrrd=img_dir, 
                    interpolation_type='linear', #"linear" for image
                    new_spacing=(1, 1, 3), 
                    return_type='sitk_obj', 
                    output_dir=img_reg_dir,
                    image_format=image_format)
                # interpolate segs
                seg_interp = interpolate(
                    patient_id=img_id, 
                    path_to_nrrd=seg_dir, 
                    interpolation_type='nearest_neighbor', # nearest neighbor for label
                    new_spacing=(1, 1, 3), 
                    return_type='sitk_obj', 
                    output_dir=seg_reg_dir,
                    image_format=image_format)        
                # --- crop full body scan ---
                z_img = img_interp.GetSize()[2]
                z_seg = seg_interp.GetSize()[2]
                if z_img > 200:
                    img = crop_full_body(img_interp, int(z_img * 0.65))
                    seg = crop_full_body(seg_interp, int(z_seg * 0.65))
                # --- registration for image and seg ---    
                fixed_img_dir = os.path.join(root_dir, 'DFCI/img_interp/10020741814.nrrd')
                fixed_img = sitk.ReadImage(fixed_img_dir, sitk.sitkFloat32)
                # register images
                fixed_image, moving_image, final_transform = nrrd_reg_rigid( 
                    patient_id=img_id, 
                    moving_image=img_interp, 
                    output_dir=img_reg_dir, 
                    fixed_image=fixed_img,
                    image_format=image_format)
                # register segmentations
                moving_label = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
                moving_label_resampled = sitk.Resample(
                    moving_label, 
                    fixed_image, 
                    final_transform, 
                    sitk.sitkNearestNeighbor, 
                    0.0, 
                    moving_image.GetPixelID())
                output_fn = seg_reg_dir + '/' + img_id + '.' + image_format
                sitk.WriteImage(moving_label_resampled, output_fn)
                #transform = sitk.ReadTransform('.tfm')


def crop(proj_dir, crop_shape, image_format):
    
    """
    With TOP-CROP HPC ### NEED TO RUN FOR image_crop, image_crop_p, and image_crop_n  
    WILL ONLY WORK WITH SPACING = 1,1,3
    """
    print('\n------start cropping--------')
    img_reg_dir = proj_dir + '/reg_img'
    seg_reg_dir = proj_dir + '/reg_seg'
    img_crop_dir = proj_dir + '/crop_img'
    seg_crop_dir = proj_dir + '/crop_seg'
    if not os.path.exists(img_crop_dir):
        os.makedirs(img_crop_dir)
    if not os.path.exists(seg_crop_dir):
        os.makedirs(seg_crop_dir)
    img_reg_dirs = [i for i in sorted(glob.glob(img_reg_dir + '/*' + image_format))]
    seg_reg_dirs = [i for i in sorted(glob.glob(seg_reg_dir + '/*' + image_format))]
    # registration for image and seg
    img_ids = []
    count = 0
    for img_dir in img_reg_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        for seg_dir in seg_reg_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                count += 1
                print(count, img_id)
                img_ids.append(img_id)
                try:
                    crop_top(
                        patient_id=img_id,
                        img_dir=img_dir,
                        seg_dir=seg_dir,
                        crop_shape=crop_shape,
                        return_type='sitk_object',
                        output_img_dir=img_crop_dir,
                        output_seg_dir=seg_crop_dir,
                        image_format=image_format)
                except Exception as e:
                    print(e, 'crop failed!')


if __name__ == '__main__':

    root_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    proj_dir = root_dir + '/Hecktor_TCIA_data'
    image_format = 'nii.gz'
    #crop_shape = (160, 160, 64)
    crop_shape = (172, 172, 76)
    
    do_register = True
    do_crop = False

    if do_register:
        registration(proj_dir, root_dir, image_format)
    if do_crop:
        crop(proj_dir, crop_shape, image_format)



    
