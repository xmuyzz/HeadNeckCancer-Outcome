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
    #---------------------------------------------------------------
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
    #-------------------
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

    # apply mask to image
    #--------------------
    masked_arr = np.where(seg_arr==1, norm_img, seg_arr)
    #masked_arr = norm_img * seg_arr
    
    # get 3d bounding box
    #--------------------
    dmin, dmax, hmin, hmax, wmin, wmax = get_bbox_3D(masked_arr)
    # choose masked image or whole image in bbox
    if input_img_type == 'masked_img':
        img_bbox = masked_arr[dmin:dmax+1, hmin:hmax+1, wmin:wmax+1]
    elif input_img_type == 'raw_img':
        img_bbox = img_arr[dmin:dmax+1, hmin:hmax+1, wmin:wmax+1]
    #print('masked_arr:', masked_arr.shape)
    #print('img_bbox:', img_bbox.shape)
    
    # padding to match max bbox
    #--------------------------
    if padding:
        d_pad = d_max - img_bbox.shape[0]
        h_pad = h_max - img_bbox.shape[1]
        w_pad = w_max - img_bbox.shape[2]
        # keep consistent bbox size
        pad_1s = []
        pad_2s = []
        for pad in [d_pad, h_pad, w_pad]:
            assert pad >= 0, print('pad:', pad, img_bbox.shape)
            if pad % 2 == 0:
                pad_1 = pad_2 = pad // 2
            else:
                pad_1 = pad // 2
                pad_2 = pad // 2 + 1
            pad_1s.append(pad_1)
            pad_2s.append(pad_2)
        # numpy padding
        img_pad = np.pad(
            array=img_bbox, 
            pad_width=((pad_1s[0], pad_2s[0]), 
                       (pad_1s[1], pad_2s[1]), 
                       (pad_1s[2], pad_2s[2])), 
            mode='constant',
            constant_values=[(0, 0), (0, 0), (0, 0)]
            )
        #print('img_pad:', img_pad.shape)
        image = img_pad
        #img = img_pad.transpose(1, 2, 0)
        #img = np.transpose(img_pad, axes=[1, 2, 0])
    else:
        image = img_bbox
    #print('img:', image.shape)
   

    return image
