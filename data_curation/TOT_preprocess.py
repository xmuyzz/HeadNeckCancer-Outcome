import sys
import os
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


def copy_files(proj_dir):
    HKTR_img_dir = proj_dir + '/HKTR/interp_img'
    HKTR_seg_dir = proj_dir + '/HKTR/interp_seg'
    TCIA_img_dir = proj_dir + '/TCIA/interp_img'
    TCIA_seg_dir = proj_dir + '/TCIA/interp_seg'
    DFCI_img_dir = proj_dir + '/DFCI/interp_img'
    DFCI_seg_dir = proj_dir + '/DFCI/interp_seg'
    TOT_img_dir = proj_dir + '/TOT/interp_img'
    TOT_seg_dir = proj_dir + '/TOT/interp_seg'
    HKTR_img_dirs = [i for i in sorted(glob.glob(HKTR_img_dir + '/*nii.gz'))]
    HKTR_seg_dirs = [i for i in sorted(glob.glob(HKTR_seg_dir + '/*nii.gz'))]
    DFCI_img_dirs = [i for i in sorted(glob.glob(DFCI_img_dir + '/*nii.gz'))]
    DFCI_seg_dirs = [i for i in sorted(glob.glob(DFCI_seg_dir + '/*nii.gz'))]
    img_dirs = HKTR_img_dirs + DFCI_img_dirs
    seg_dirs = HKTR_img_dirs + DFCI_seg_dirs
    data_dirss = [img_dirs, seg_dirs]
    save_dirs = [TOT_img_dir, TOT_seg_dir]
    for data_dirs, save_dir in zip(data_dirss, save_dirs):
        for i, data_dir in enumerate(data_dirs):
            fn = data_dir.split('/')[-1]
            print(i, fn)
            save_fn = save_dir + '/' + fn
            shutil.copyfile(data_dir, save_fn)


def combine_TCIA_HKTR(proj_dir):

    """
    save nrrd to nii 
    """
    TCIA_img_dir = proj_dir + '/TCIA/interp_img'
    TCIA_seg_dir = proj_dir + '/TCIA/interp_seg_p_n'
    HKTR_img_dir = proj_dir + '/HKTR/interp_img'
    HKTR_seg_dir = proj_dir + '/HKTR/interp_seg'
    TOT_img_dir = proj_dir + '/TOT/interp_img'
    TOT_seg_dir = proj_dir + '/TOT/interp_seg'

    # get hecktor data id
    img_ids = []
    for img_dir in sorted(glob.glob(HKTR_img_dir + '/*nii.gz')):
        img_id = img_dir.split('/')[-1].split('.')[0]
        img_ids.append(img_id)
    print('\ntotal hecktor data:', len(img_ids))
    seg_ids = []
    for seg_dir in sorted(glob.glob(HKTR_seg_dir + '/*nii.gz')):
        seg_id = seg_dir.split('/')[-1].split('.')[0]
        seg_ids.append(seg_id)
    print('\ntotal hecktor data:', len(seg_ids))

    # move TCIA data to hecktor
    count = 0
    print('\n--- combine img data ----')
    for data_dir in sorted(glob.glob(TCIA_img_dir + '/*nii.gz')):
        data_id = data_dir.split('/')[-1].split('.')[0]
        if data_id not in img_ids:
            count += 1
            print(count, data_id)
            save_fn = TOT_img_dir + '/' + data_id + '.nii.gz'
            shutil.copyfile(data_dir, save_fn)
    
    print('\n---combine seg data----')
    for data_dir in sorted(glob.glob(TCIA_seg_dir + '/*nii.gz')):
        data_id = data_dir.split('/')[-1].split('.')[0]
        if data_id not in seg_ids:
            count += 1
            print(count, data_id)
            save_fn = TOT_seg_dir + '/' + data_id + '.nii.gz'
            shutil.copyfile(data_dir, save_fn)


def interpolation(proj_dir, image_format, dataset):
    """
    interpolation
    """
    print('\n --- start interpolation ---')
    img_raw_dir = proj_dir + '/' + dataset + '/raw_img'
    seg_raw_dir = proj_dir + '/' + dataset + '/raw_seg'
    img_itp_dir = proj_dir + '/' + dataset + '/itp_img'
    seg_itp_dir = proj_dir + '/' + dataset + '/itp_seg'
    if not os.path.exists(img_itp_dir):
        os.makedirs(img_itp_dir)
    if not os.path.exists(seg_itp_dir):
        os.makedirs(seg_itp_dir)
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
                    output_dir=img_itp_dir,
                    image_format=image_format)
                # interpolate segs
                seg_interp = interpolate(
                    patient_id=img_id, 
                    path_to_nrrd=seg_dir, 
                    interpolation_type='nearest_neighbor', # nearest neighbor for label
                    new_spacing=(1, 1, 3), 
                    return_type='sitk_obj', 
                    output_dir=seg_itp_dir,
                    image_format=image_format)        
        

def reg_crop(proj_dir, root_dir, image_format, crop_shape):

    interp_img_dir = proj_dir + '/interp_img'
    interp_seg_dir = proj_dir + '/interp_seg_p_n'
    crop_img_dir = proj_dir + '/crop_img_160'
    crop_seg_dir = proj_dir + '/crop_seg_p_n_160'
    if not os.path.exists(crop_img_dir):
        os.makedirs(crop_img_dir)
    if not os.path.exists(crop_seg_dir):
        os.makedirs(crop_seg_dir)
    img_dirs = [i for i in sorted(glob.glob(interp_img_dir + '/*' + image_format))]
    seg_dirs = [i for i in sorted(glob.glob(interp_seg_dir + '/*' + image_format))]
    img_ids = []
    bad_ids = []
    count = 0
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        for seg_dir in seg_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                if 'MDA_6' in img_id or 'PMH_17' in img_id or 'PMH_2' in img_id or 'PMH_3' in img_id \
                    or 'PMH_4' in img_id or 'PMH_5' in img_id or 'PMH_6' in img_id:
                    count += 1
                    print(count, img_id)
                    img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
                    seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
                    # --- crop full body scan ---
                    z_img = img.GetSize()[2]
                    z_seg = seg.GetSize()[2]
                    if z_img > 200:
                        img = crop_full_body(img, int(z_img * 0.65))
                        seg = crop_full_body(seg, int(z_seg * 0.65))
                    # --- registration for image and seg ---    
                    fixed_img_dir = os.path.join(root_dir, 'DFCI/img_interp/10020741814.nrrd')
                    fixed_img = sitk.ReadImage(fixed_img_dir, sitk.sitkFloat32)
                    try:
                        # register images
                        reg_img, fixed_img, moving_img, final_transform = nrrd_reg_rigid( 
                            patient_id=img_id, 
                            moving_img=img, 
                            output_dir='', 
                            fixed_img=fixed_img,
                            image_format=image_format)
                        # register segmentations
                        reg_seg = sitk.Resample(
                            seg, 
                            fixed_img, 
                            final_transform, 
                            sitk.sitkNearestNeighbor, 
                            0.0, 
                            moving_img.GetPixelID())
                        # crop
                        crop_top(
                            patient_id=img_id,
                            img=reg_img,
                            seg=reg_seg,
                            crop_shape=crop_shape,
                            return_type='sitk_object',
                            output_img_dir=crop_img_dir,
                            output_seg_dir=crop_seg_dir,
                            image_format=image_format)
                    except Exception as e:
                        bad_ids.append(img_id)
                        print(img_id, e)
    print('bad ids:', bad_ids)
                    

if __name__ == '__main__':

    root_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/TOT'
    #proj_dir = root_dir + '/HKTR_TCIA_DFCI'
    image_format = 'nii.gz'
    crop_shape = (160, 160, 64)
    #crop_shape = (172, 172, 76)
    dataset = 'DFCI'
    step = 'reg_crop'

    if step == 'interpolate':
        interpolation(proj_dir, image_format, dataset)
    elif step == 'reg_crop':
        reg_crop(proj_dir, root_dir, image_format, crop_shape)
    elif step == 'copy files':
        copy_files(proj_dir)
    elif step == 'combine HKTR TCIA':
        combine_TCIA_HKTR(proj_dir)





    
