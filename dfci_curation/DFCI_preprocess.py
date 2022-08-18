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




def combine_mask(proj_dir):
    """
    COMBINING MASKS 
    """
    dfci_img_dir = os.path.join(proj_dir, 'DFCI/dfci_data')
    dfci_seg_dir = os.path.join(proj_dir, 'DFCI/dfci_seg')
    output_dir = os.path.join(proj_dir, 'DFCI/combined_seg')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img_dir in sorted(glob.glob(dfci_img_dir + '/*nrrd')):
        seg_dirs = []
        seg_IDs = []
        pat_id = img_dir.split('/')[-1].split('_')[1]
        #print(pat_id)
        try:
            for folder in os.listdir(dfci_seg_dir):
                ID = folder.split('_')[1]
                if ID == pat_id:
                    #print(ID)
                    for seg_dir in glob.glob(os.path.join(dfci_seg_dir, folder) + '/*nrrd'):
                        seg_ID = seg_dir.split('/')[-1].split('.')[0]
                        if 'GTV' in seg_ID and 'cm' not in seg_ID and 'mm' not in seg_ID \
                        and '+' not in seg_ID and '**' not in seg_ID:
                            seg_dirs.append(seg_dir)
                            seg_IDs.append(seg_ID)
                    #print(seg_dirs)
                    print(seg_IDs)
                    combined_mask = combine_structures(
                        dataset='DFCI', 
                        patient_id=pat_id, 
                        data_type='ct', 
                        mask_arr=seg_dirs, 
                        path_to_reference_image_nrrd=img_dir, 
                        binary=2, 
                        return_type='sitk_object', 
                        output_dir=output_dir
                        )
                    print('combine successfully')
        except Exception as e: 
            print(e)    



def interpolate_img(proj_dir):
    """
    image interpolation
    """
    dfci_img_dir = os.path.join(proj_dir, 'DFCI/dfci_data')
    dfci_seg_dir = os.path.join(proj_dir, 'DFCI/combined_seg')
    img_interp_dir = os.path.join(proj_dir, 'DFCI/img_interp')
    seg_interp_dir = os.path.join(proj_dir, 'DFCI/seg_interp')
    if not os.path.exists(img_interp_dir):
        os.makedirs(img_interp_dir)
    if not os.path.exists(seg_interp_dir):
        os.makedirs(seg_interp_dir)
    try:
        # interpolate images
        print('interpolate images...')
        for count, img_dir in enumerate(sorted(glob.glob(dfci_img_dir + '/*nrrd'))):
            print(count)
            pat_id = img_dir.split('/')[-1].split('_')[1]
            interpolated_nrrd = interpolate(
                dataset='DFCI', 
                patient_id=pat_id, 
                data_type='ct', 
                path_to_nrrd=img_dir, 
                interpolation_type='linear', #"linear" for image, nearest neighbor for label
                new_spacing=(1, 1, 3), 
                return_type='numpy_array', 
                output_dir=img_interp_dir)
        # interpolate labels
        print('interpolate segmetations...')
        for count, seg_dir in enumerate(sorted(glob.glob(dfci_seg_dir + '/*nrrd'))):
            print(count)
            pat_id = seg_dir.split('/')[-1].split('_')[1]
            print(pat_id)        
            interpolated_nrrd = interpolate(
                dataset='DFCI', 
                patient_id=pat_id, 
                data_type='seg', 
                path_to_nrrd=seg_dir, 
                interpolation_type='nearest_neighbor', 
                new_spacing=(1, 1, 3), 
                return_type='numpy_array', 
                output_dir=seg_interp_dir)
    except Exception as e:
        print(e)



def registration(proj_dir):
    """
    Rigid Registration - followed by top crop
    """
    img_interp_dir = os.path.join(proj_dir, 'DFCI/img_interp')
    seg_interp_dir = os.path.join(proj_dir, 'DFCI/seg_interp')
    img_reg_dir = os.path.join(proj_dir, 'DFCI/img_reg')
    seg_reg_dir = os.path.join(proj_dir, 'DFCI/seg_reg')
    fixed_img_dir = os.path.join(proj_dir, 'DFCI/img_interp/10020741814.nrrd')
    if not os.path.exists(img_reg_dir):
        os.makedirs(img_reg_dir)
    if not os.path.exists(seg_reg_dir):
        os.makedirs(seg_reg_dir)
    # register images
    count = 0
    img_ids = []
    img_dirs = [i for i in sorted(glob.glob(img_interp_dir + '/*nrrd'))]
    seg_dirs = [i for i in sorted(glob.glob(seg_interp_dir + '/*nrrd'))]
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        for seg_dir in seg_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                img_ids.append(img_id)
                count += 1
                print(count)
                print(img_id)
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
                        os.path.join(seg_reg_dir, img_id + '.nrrd'))
                    #transform = sitk.ReadTransform('.tfm')
                except Exception as e:
                    print(e, 'segmentation registration failed')
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        if img_id not in img_ids:
            count += 1
            print(count)
            print(img_id)
            try:
                fixed_image, moving_image, final_transform = nrrd_reg_rigid(
                    patient_id=img_id,
                    input_dir=img_dir,
                    output_dir=img_reg_dir,
                    fixed_img_dir=fixed_img_dir)
            except Exception as e:
                print(e)


def crop(proj_dir):
    
    """
    With TOP-CROP HPC ### NEED TO RUN FOR image_crop, image_crop_p, and image_crop_n  
    WILL ONLY WORK WITH SPACING = 1,1,3
    """

    roi_size = (172, 172, 76) #x,y,z
    size_str = '172 x 172 x 76'
    img_dirs = [i for i in sorted(glob.glob(dfci_img_dir + '/*nrrd'))]
    seg_dirs = [i for i in sorted(glob.glob(dfci_seg_dir + '/*nrrd'))]
    img_crop_dir = proj_dir + '/DFCI/img_crop'
    seg_crop_dir = proj_dir + '/DFCI/seg_crop'
    if not os.path.exists(img_crop_dir):
        os.makedirs(img_crop_dir)
    if not os.path.exists(seg_crop_dir):
        os.makedirs(seg_crop_dir)
    img_dirs = [i for i in sorted(glob.glob(img_reg_dir + '/*nrrd'))]
    seg_dirs = [i for i in sorted(glob.glob(seg_reg_dir + '/*nrrd'))]
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        for seg_dir in seg_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                img_ids.append(img_id)
                try:
                    image_obj, label_obj = crop_top(
                        patient_id=img_id,
                        img_dir=img_dir,
                        seg_dir=seg_dir,
                        crop_shape=roi_size,
                        return_type='sitk_object',
                        output_img_folder=output_img_folder,
                        output_seg_folder=output_seg_folder)
                except:
                    print('crop failed!')
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        if img_id not in img_ids:
            count += 1
            print(count)
            print(img_id)
            try:
                image_obj, label_obj = crop_top_image_only(
                    patient_id,
                    img_dir,
                    crop_shape=roi_size,
                    return_type='sitk_object',
                    output_img_folder=output_img_folder)
            except Exception as e:
                print(e)                        



if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    
    do_combine_mask = True
    do_interpolate = True
    do_register = True
    do_crop = True

    if do_combine_mask:
        combine_mask(proj_dir)
    if do_interpolate:
        interpolate_img(proj_dir)    
    if do_register:
        registration(proj_dir)


    
