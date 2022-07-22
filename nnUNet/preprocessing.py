import numpy as np
import os
import glob
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from get_data.bbox import get_bbox_3D
from get_data.resize_3d import resize_3d
from get_data.respacing import respacing



def prepro_img(img_dir, seg_dir, new_spacing, norm_type, input_img_type, input_channel, 
               d_max=70, h_max=70, w_max=70, padding=True, do_respacing=True):

    """
    get cnosistent 3D tumor&node data using masking, bbox and padding
    
    Args:
      img_dir {path} -- dir for image in nrrd format
      label_dir {path} -- dir for label in nrrd format
      r_max {int} -- row of largest bbox
      c_max {int} -- column of largest bbox
      z_max {int} -- z of largest bbox
    
    Returns:
        Preprocessed images in nii.gz or np.array formats;

    """

    # respacing images and segmentations from (1, 1, 3) to (2, 2, 2)      
    if do_respacing:
        # respacing for image
        img_arr = respacing(
            nrrd_dir=img_dir,
            interp_type='linear',
            new_spacing=new_spacing,
            patient_id=None,
            return_type='npy',
            save_dir=None
            )
        # respacing for segmentation
        seg_arr = respacing(
            nrrd_dir=seg_dir,
            interp_type='nearest_neighbor',
            new_spacing=new_spacing,
            patient_id=None,
            return_type='npy',
            save_dir=None
            )
    else:
        img_nrrd = sitk.ReadImage(img_dir)
        img_arr = sitk.GetArrayFromImage(img_nrrd)
        seg_nrrd = sitk.ReadImage(seg_dir)
        seg_arr = sitk.GetArrayFromImage(seg_nrrd)

    # normalize CT image
    data = img_arr
    data[data <= -1024] = -1024
    # strip skull, skull UHI = ~700
    data[data > 700] = 0
    # normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
    if norm_type == 'np_interp':
        norm_img = np.interp(data, [-200, 200], [0, 1])
    elif norm_type == 'np_clip':
        data = np.clip(data, a_min=-200, a_max=200)
        MAX, MIN = data.max(), data.min()
        norm_img = (data - MIN) / (MAX - MIN)

   

    return image
