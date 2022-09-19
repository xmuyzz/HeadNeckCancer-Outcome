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
import nibabel as ni


def transfer_img_file(proj_dir, raw_data_dir):
    
    """
    1) Transfer image, pn seg, n seg and p seg file to organzied folder;
    2) rename files;
    """

    HGJ_img_dir = raw_data_dir + '/HGJ_files/0_image_raw_HGJ'
    HGJ_seg_dir = raw_data_dir + '/HGJ_files/1_label_raw_HGJ_named'
    HMR_img_dir = raw_data_dir + '/HMR_files/0_image_raw_HMR'
    HMR_seg_dir = raw_data_dir + '/HGJ_files/1_label_raw_HMR'
    HGJ_img_save_dir = proj_dir + '/TCIA/HGJ/raw_imgs'
    HGJ_seg_save_dir = proj_dir + '/TCIA/HGJ/raw_segs'
    HMR_img_save_dir = proj_dir + '/TCIA/HMR/raw_imgs'
    HMR_seg_save_dir = proj_dir + '/TCIA/HMR/raw_segs'

    if not os.path.exists(HGJ_img_save_dir):
        os.makedirs(HGJ_img_save_dir)
    if not os.path.exists(HGJ_seg_save_dir):
        os.makedirs(HGJ_seg_save_dir)
    if not os.path.exists(HMR_img_save_dir):
        os.makedirs(HMR_img_save_dir)
    if not os.path.exists(HMR_seg_save_dir):
        os.makedirs(HMR_seg_save_dir)
    
    # rename img file and transfer to lab drive
    data_dirs = [HGJ_img_dir, HMR_img_dir]
    save_dirs = [HGJ_img_save_dir, HMR_img_save_dir]
    cohorts = ['HGJ', 'HMR']
    count = 0
    for data_dir, save_dir, cohort in zip(data_dirs, save_dirs, cohorts):
        for img_dir in glob.glob(data_dir + '/*nrrd'):
            if cohort == 'HMR':
                if 'CT-SIM' in img_dir:
                    count += 1
                    ID = img_dir.split('/')[-1].split('_')[1].split('-')[2]
                    fn = cohort + '_' + ID + '.nrrd'
                    print(count, fn)
                    dst_dir = save_dir + '/' + fn
                    shutil.copyfile(img_dir, dst_dir)
            elif cohort == 'HGJ':
                if 'CT-PET' in img_dir:
                    count += 1
                    ID = img_dir.split('/')[-1].split('_')[1].split('-')[2]
                    fn = cohort + '_' + ID + '.nrrd'
                    print(count, fn)
                    dst_dir = save_dir + '/' + fn
                    shutil.copyfile(img_dir, dst_dir)
    

def combine_segmentation(proj_dir):
    """
    1) combine p_seg and n_seg to a 4d nii image;
    2) p_seg and n_seg in different channels;
    Args:
        proj_dir {path} -- project path
    Returns:
        save nii files
    Raise issues:
        none
    """
    p_seg_path = proj_dir + '/TCIA/p_segs'
    n_seg_path = proj_dir + '/TCIA/n_segs'
    pn_seg_path = proj_dir + '/TCIA/pn_segs'
    p_n_seg_path = proj_dir + '/TCIA/p_n_segs'
    fns = [i for i in os.listdir(pn_seg_path)]
    for i, fn in enumerate(fns):
        seg_id = fn.split('.')[0]
        print(i, seg_id)
        # primary tumor
        p_seg_dir = p_seg_path + '/' + fn
        if os.path.exists(p_seg_dir):
            p_seg = sitk.ReadImage(p_seg_dir)
            p_seg = sitk.GetArrayFromImage(p_seg)
        else:
            print('no primary segmentation...')
            p_seg = np.zeros(shape=(176, 176, 72))
        # node
        n_seg_dir = p_seg_path + '/' + fn
        if os.path.exists(n_seg_dir):
            n_seg = sitk.ReadImage(n_seg_dir)
            n_seg = sitk.GetArrayFromImage(n_seg)
        else:
            print('no node segmentation...')
            n_seg = np.zeros(shape=(176, 176, 72))
        pn_seg = np.stack((p_seg, n_seg), axis=3)
        pn_seg = nib.Nifti1Image(pn_seg, affine=np.eye(4))
        nib.save(pn_seg, os.path.join(pn_save_path, seg_id, '.nii.gz'))


def combine_mask(proj_dir, raw_data_dir):
    """
    COMBINING MASKS 
    """
    HGJ_img_dir = proj_dir + '/TCIA/HGJ/raw_imgs'
    HMR_img_dir = proj_dir + '/TCIA/HMR/raw_imgs'
    HGJ_seg_dir = raw_data_dir + '/HGJ_files/1_label_raw_HGJ_named'
    HMR_seg_dir = raw_data_dir + '/HMR_files/1_label_raw_HMR'
    HGJ_seg_save_dir = proj_dir + '/TCIA/HGJ/raw_segs'
    HMR_seg_save_dir = proj_dir + '/TCIA/HMR/raw_segs'
 
    # rename img file and transfer to lab drive
    data_dirs = [HGJ_img_dir, HMR_img_dir]
    mask_dirs = [HGJ_seg_dir, HMR_seg_dir]
    save_dirs = [HGJ_seg_save_dir, HMR_seg_save_dir]
    cohorts = ['HGJ', 'HMR']
    count = 0
    for data_dir, mask_dir, save_dir, cohort in zip(data_dirs, mask_dirs, save_dirs, cohorts):
        for img_dir in sorted(glob.glob(data_dir + '/*nrrd')):
            seg_dirs = []
            seg_IDs = []
            img_id = img_dir.split('/')[-1].split('.')[0]
            #print(img_id)
            for seg_folder in sorted(glob.glob(mask_dir + '/*')):
                #print(folder)
                seg_id = cohort + '_' + seg_folder.split('-')[-1]
                if seg_id == img_id:
                    print('ID:', seg_id)
                    for seg_dir in sorted(glob.glob(seg_folder + '/*nrrd')):
                        seg_ID = seg_dir.split('/')[-1].split('.')[0]
                        #print(seg_dir)
                        #print(seg_ID)
                        if 'GTV' in seg_ID and 'cm' not in seg_ID and 'mm' not in seg_ID \
                            and '+' not in seg_ID and '**' not in seg_ID and 'z' not in seg_ID:
                            seg_dirs.append(seg_dir)
                            seg_IDs.append(seg_ID)
                    #print(seg_dirs)
                    print('seg IDs:', seg_IDs)
                    combined_mask = combine_structures(
                        patient_id=seg_id, 
                        mask_arr=seg_dirs, 
                        path_to_reference_image_nrrd=img_dir, 
                        binary=2, 
                        return_type='sitk_object', 
                        output_dir=save_dir)
                    print('combine successfully')
            

def registration(proj_dir, tumor_type, label_only):
    """
    Rigid Registration - followed by top crop
    """
    print('------start registration--------')
    img_raw_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/raw_img')
    seg_pn_raw_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/raw_pn_seg')
    seg_p_raw_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/raw_p_seg')
    seg_n_raw_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/raw_n_seg')

    img_reg_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/reg_img')
    seg_pn_reg_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/reg_seg_pn')
    seg_p_reg_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/reg_seg_p')
    seg_n_reg_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/reg_seg_n')
    if not os.path.exists(img_reg_dir):
        os.makedirs(img_reg_dir)
    if not os.path.exists(seg_pn_reg_dir):
        os.makedirs(seg_pn_reg_dir)
    if not os.path.exists(seg_p_reg_dir):
        os.makedirs(seg_p_reg_dir)
    if not os.path.exists(seg_n_reg_dir):
        os.makedirs(seg_n_reg_dir)

    img_dirs = [i for i in sorted(glob.glob(img_raw_dir + '/*nrrd'))]
    seg_pn_dirs = [i for i in sorted(glob.glob(seg_pn_raw_dir + '/*nrrd'))]
    seg_p_dirs = [i for i in sorted(glob.glob(seg_p_raw_dir + '/*nrrd'))]
    seg_n_dirs = [i for i in sorted(glob.glob(seg_n_raw_dir + '/*nrrd'))]
    if tumor_type == 'pn':
        seg_dirs = seg_pn_dirs
        seg_reg_dir = seg_pn_reg_dir
    elif tumor_type == 'p':
        seg_dirs = seg_p_dirs
        seg_reg_dir = seg_p_reg_dir
    elif tumor_type == 'n':
        seg_dirs = seg_n_dirs
        seg_reg_dir = seg_n_reg_dir
    img_ids = []
    count = 0
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        for seg_dir in seg_pn_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                img_ids.append(img_id)
                count += 1
                print(count, img_id)
                # --- interpolation for image and seg to 1x1x3 ---
                # interpolate images
                img_interp = interpolate(
                    dataset='DFCI', 
                    patient_id=img_id, 
                    data_type='ct', 
                    path_to_nrrd=img_dir, 
                    interpolation_type='linear', #"linear" for image
                    new_spacing=(1, 1, 3), 
                    return_type='sitk_obj', 
                    output_dir='')
                # interpolate segs
                seg_interp = interpolate(
                    dataset='DFCI', 
                    patient_id=img_id, 
                    data_type='seg', 
                    path_to_nrrd=seg_dir, 
                    interpolation_type='nearest_neighbor', # nearest neighbor for label
                    new_spacing=(1, 1, 3), 
                    return_type='sitk_obj', 
                    output_dir='')
                
                # --- registration for image and seg to 1x1x3 ---    
                fixed_img_dir = os.path.join(proj_dir, 'DFCI/img_interp/10020741814.nrrd')
                fixed_img = sitk.ReadImage(fixed_img_dir, sitk.sitkFloat32)
                # register images
                fixed_image, moving_image, final_transform = nrrd_reg_rigid( 
                    patient_id=img_id, 
                    moving_image=img_interp, 
                    output_dir=img_reg_dir, 
                    fixed_image=fixed_img,
                    label_only=label_only)
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
                    os.path.join(seg_reg_dir, img_id + '.nrrd'))
                #transform = sitk.ReadTransform('.tfm')


def crop(proj_dir, tumor_type, label_only):
    
    """
    With TOP-CROP HPC ### NEED TO RUN FOR image_crop, image_crop_p, and image_crop_n  
    WILL ONLY WORK WITH SPACING = 1,1,3
    """
    print('------start cropping--------')
    #crop_shape = (160, 160, 64) #x,y,z
    crop_shape = (172, 172, 76)
    img_reg_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/reg_img')
    seg_pn_reg_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/reg_seg_pn')
    seg_p_reg_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/reg_seg_p')
    seg_n_reg_dir = os.path.join(proj_dir, 'TCIA/HGJ_HMR_data/reg_seg_n')
    img_crop_dir = proj_dir + '/TCIA/HGJ_HMR_data/crop_img'
    seg_pn_crop_dir = proj_dir + '/TCIA/HGJ_HMR_data/crop_seg_pn'
    seg_p_crop_dir = proj_dir + '/TCIA/HGJ_HMR_data/crop_seg_p'
    seg_n_crop_dir = proj_dir + '/TCIA/HGJ_HMR_data/crop_seg_n'
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
                try:
                    image_obj, label_obj = crop_top(
                        patient_id=img_id,
                        img_dir=img_dir,
                        seg_dir=seg_dir,
                        crop_shape=crop_shape,
                        return_type='sitk_object',
                        output_img_dir=img_crop_dir,
                        output_seg_dir=seg_crop_dir)
                except Exception as e:
                    print(e, 'crop failed!')


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    raw_data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated' 
    label_only = False
    do_transfer_file = False
    do_combine_mask = False
    do_register = False
    do_crop = True

    if do_transfer_file:
        transfer_img_file(proj_dir, raw_data_dir)
    if do_combine_mask:
        combine_mask(proj_dir, raw_data_dir)
    if do_register:
        for tumor_type in ['pn']:
            registration(
                proj_dir=proj_dir, 
                tumor_type=tumor_type,
                label_only=label_only)
    if do_crop:
        for tumor_type in ['pn']:
            crop(
                proj_dir, 
                tumor_type=tumor_type,
                label_only=label_only)



    
