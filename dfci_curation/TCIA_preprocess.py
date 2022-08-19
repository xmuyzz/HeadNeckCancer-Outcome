### Initiate this file after nrrds have been generated for labels and images in run_bk file 
## Benjamin Kann
### Order of operations: combine label nrrds, interpolate image, interpolate combined label, crop roi

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



def transfer_file(raw_data_dir, proj_dir):
    
    """
    1) Transfer image, pn seg, n seg and p seg file to organzied folder;
    2) rename files;
    """

    CHUS_dir = raw_data_dir + '/CHUS_files/interpolated'
    CHUM_dir = raw_data_dir + '/CHUM_files/interpolated'
    MDACC_dir = raw_data_dir + '/MDACC_files/interpolated'
    PMH_dir = raw_data_dir + '/PMH_files/interpolated'
    dst_img_dir = proj_dir + '/TCIA/imgs'
    pn_seg_dir = proj_dir + '/TCIA/pn_segs'
    p_seg_dir = proj_dir + '/TCIA/p_segs'
    n_seg_dir = proj_dir + '/TCIA/n_segs'
    if not os.path.exists(dst_img_dir):
        os.makedirs(dst_img_dir)
    if not os.path.exists(pn_seg_dir):
        os.makedirs(pn_seg_dir)
    if not os.path.exists(p_seg_dir):
        os.makedirs(p_seg_dir)
    if not os.path.exists(n_seg_dir):
        os.makedirs(n_seg_dir)
    
    # dataset
    count = 0
    #data_dirs = [CHUM_dir, CHUS_dir, MDACC_dir, PMH_dir]
    #cohorts = ['CHUM', 'CHUS', 'MDACC', 'PMH']
    data_dirs = [MDACC_dir, PMH_dir]
    cohorts = ['MDACC', 'PMH']
    for data_dir, cohort in zip(data_dirs, cohorts):
        for path in sorted(glob.glob(data_dir + '/*nrrd')):
            if cohort in ['CHUM', 'CHUS']:
                ID = path.split('/')[-1].split('_')[1].split('-')[2]
            elif cohort == 'MDACC':
                ID = path.split('/')[-1].split('_')[1].split('-')[2][1:]
            elif cohort == 'PMH':
                ID = path.split('/')[-1].split('_')[1].split('-')[1][2:]
            fn = cohort + '_' + ID + '.nrrd'
            data_type = path.split('/')[-1].split('_')[2]
            count += 1
            print(count, fn)
            if data_type == 'ct':
                dst_dir = dst_img_dir + '/' + fn
                print(path)
                print(dst_dir)
                shutil.copyfile(path, dst_dir)
            elif data_type == 'label':
                label = path.split('/')[-1].split('_')[3]
                if label == 'interpolated':
                    dst_dir = pn_seg_dir + '/' + fn
                elif label == 'n':
                    dst_dir = n_seg_dir + '/' + fn
                elif label == 'p':
                    dst_dir = p_seg_dir + '/' + fn
                print(path)
                print(dst_dir) 
                shutil.copyfile(path, dst_dir)


def registration(proj_dir, tumor_type):
    """
    Rigid Registration - followed by top crop
    """
    img_interp_dir = os.path.join(proj_dir, 'TCIA/imgs')
    seg_pn_interp_dir = os.path.join(proj_dir, 'TCIA/pn_segs')
    seg_p_interp_dir = os.path.join(proj_dir, 'TCIA/p_segs')
    seg_n_interp_dir = os.path.join(proj_dir, 'TCIA/n_segs')
    img_reg_dir = os.path.join(proj_dir, 'TCIA/img_reg')
    seg_pn_reg_dir = os.path.join(proj_dir, 'TCIA/pn_seg_reg')
    seg_p_reg_dir = os.path.join(proj_dir, 'TCIA/p_seg_reg')
    seg_n_reg_dir = os.path.join(proj_dir, 'TCIA/n_seg_reg')
    if not os.path.exists(img_reg_dir):
        os.makedirs(img_reg_dir)
    if not os.path.exists(seg_pn_reg_dir):
        os.makedirs(seg_pn_reg_dir)
    if not os.path.exists(seg_p_reg_dir):
        os.makedirs(seg_p_reg_dir)
    if not os.path.exists(seg_n_reg_dir):
        os.makedirs(seg_n_reg_dir)
    fixed_img_dir = os.path.join(proj_dir, 'DFCI/img_interp/10020741814.nrrd')
    # register images
    count = 0
    img_ids = []
    img_dirs = [i for i in sorted(glob.glob(img_interp_dir + '/*nrrd'))]
    seg_pn_dirs = [i for i in sorted(glob.glob(seg_pn_interp_dir + '/*nrrd'))]
    seg_p_dirs = [i for i in sorted(glob.glob(seg_p_interp_dir + '/*nrrd'))]
    seg_n_dirs = [i for i in sorted(glob.glob(seg_n_interp_dir + '/*nrrd'))]
    
    if tumor_type == 'pn':
        seg_dirs = seg_pn_dirs
        seg_reg_dir = seg_pn_reg_dir
    elif tumor_type == 'p':
        seg_dirs = seg_p_dirs
        seg_reg_dir = seg_p_reg_dir
    elif tumor_type == 'n':
        seg_dirs = seg_n_dirs
        seg_reg_dir = seg_n_reg_dir
    # registration for image and seg
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        for seg_dir in seg_pn_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                img_ids.append(img_id)
                count += 1
                print(count, img_id)
                try:
                    # register images
                    fixed_image, moving_image, final_transform = nrrd_reg_rigid( 
                        patient_id=img_id, 
                        input_dir=img_dir, 
                        output_dir=img_reg_dir, 
                        fixed_img_dir=fixed_img_dir)
                    # register segmentations
                    moving_label = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
                    moving_label_resampled = sitk.Resample(
                        moving_label, 
                        fixed_image, 
                        final_transform, 
                        sitk.sitkNearestNeighbor, 
                        0.0, 
                        moving_image.GetPixelID())
                    sitk.WriteImage(
                        moving_label_resampled, 
                        os.path.join(seg_pn_reg_dir, img_id + '.nrrd'))
                    #transform = sitk.ReadTransform('.tfm')
                except Exception as e:
                    print(e, 'segmentation registration failed')
    # register images that have no segs
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        if img_id not in img_ids:
            count += 1
            print(count, img_id)
            try:
                fixed_image, moving_image, final_transform = nrrd_reg_rigid(
                    patient_id=img_id,
                    input_dir=img_dir,
                    output_dir=img_reg_dir,
                    fixed_img_dir=fixed_img_dir)
            except Exception as e:
                print(e)


def crop(proj_dir, tumor_type='pn'):
    
    """
    With TOP-CROP HPC ### NEED TO RUN FOR image_crop, image_crop_p, and image_crop_n  
    WILL ONLY WORK WITH SPACING = 1,1,3
    """

    roi_size = (172, 172, 76) #x,y,z
    size_str = '172 x 172 x 76'
    img_reg_dir = os.path.join(proj_dir, 'TCIA/img_reg')
    seg_pn_reg_dir = os.path.join(proj_dir, 'TCIA/pn_seg_reg')
    seg_p_reg_dir = os.path.join(proj_dir, 'TCIA/p_seg_reg')
    seg_n_reg_dir = os.path.join(proj_dir, 'TCIA/n_seg_reg')
    img_crop_dir = proj_dir + '/TCIA/img_crop'
    seg_pn_crop_dir = proj_dir + '/TCIA/seg_pn_crop'
    seg_p_crop_dir = proj_dir + '/TCIA/seg_p_crop'
    seg_n_crop_dir = proj_dir + '/TCIA/seg_n_crop'
    if not os.path.exists(img_crop_dir):
        os.makedirs(img_crop_dir)
    if not os.path.exists(seg_pn_crop_dir):
        os.makedirs(seg_pn_crop_dir)
    if not os.path.exists(seg_p_crop_dir):
        os.makedirs(seg_p_crop_dir)
    if not os.path.exists(seg_n_crop_dir):
        os.makedirs(seg_n_crop_dir)
    img_reg_dirs = [i for i in sorted(glob.glob(img_reg_dir + '/*nrrd'))]
    seg_pn_reg_dirs = [i for i in sorted(glob.glob(seg_pn_reg_dir + '/*nrrd'))]
    seg_p_reg_dirs = [i for i in sorted(glob.glob(seg_p_reg_dir + '/*nrrd'))]
    seg_n_reg_dirs = [i for i in sorted(glob.glob(seg_n_reg_dir + '/*nrrd'))]
    if tumor_type == 'pn':
        seg_dirs = seg_pn_reg_dirs
        seg_crop_dir = seg_pn_crop_dir
    elif tumor_type == 'p':
        seg_dirs = seg_p_reg_dirs
        seg_crop_dir = seg_p_crop_dir
    elif tumor_type == 'n':
        seg_dirs = seg_n_reg_dirs
        seg_crop_dir = seg_n_crop_dir
    # registration for image and seg
    img_ids = []
    count = 0
    for img_dir in img_reg_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        for seg_dir in seg_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                count += 1
                print(count, img_id)
                img_ids.append(img_id)
#                try:
#                    image_obj, label_obj = crop_top(
#                        patient_id=img_id,
#                        img_dir=img_dir,
#                        seg_dir=seg_dir,
#                        crop_shape=roi_size,
#                        return_type='sitk_object',
#                        output_img_dir=img_crop_dir,
#                        output_seg_dir=seg_crop_dir)
#                except Exception as e:
#                    print(e, 'crop failed!')
    for img_dir in img_reg_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        if img_id not in img_ids:
            count += 1
            print(count, img_id)
            try:
                image_obj = crop_top_image_only(
                    patient_id=img_id,
                    img_dir=img_dir,
                    crop_shape=roi_size,
                    return_type='sitk_object',
                    output_img_dir=img_crop_dir)
            except Exception as e:
                print(e)                        



if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    raw_data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated' 

    do_transfer_file = False
    do_register = False
    do_crop = True

    if do_transfer_file:
        transfer_file(raw_data_dir, proj_dir)
    if do_crop:
        crop(proj_dir, tumor_type='pn')    
    if do_register:
        registration(proj_dir=proj_dir, tumor_type='pn')


    
