import numpy as np
import os
import glob
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from get_data.bbox import bbox_3D
from get_data.resize_3d import resize_3d


def prepro_img(img_arr, seg_arr, z_max, y_max, x_max, norm_type, 
                   input_type, input_channel, save_img_type):

    """
    get cnosistent 3D tumor&node data using masking, bbox and padding
    
    @ params:
      img_dir   - required: dir for image in nrrd format
      label_dir - required: dir for label in nrrd format
      r_max     - required: row of largest bbox
      c_max     - required: column of largest bbox
      z_max     - required: z of largest bbox
    """

    # apply mask to image
    #--------------------
    masked_arr = np.where(seg_arr==1, img_arr, seg_arr)
    # alternative: masked_arr = img_arr*label_arr
    #print('img arr:', img_arr[:, :, 68])

    # get 3d bbox
    #--------------
    zmin, zmax, ymin, ymax, xmin, xmax = bbox_3D(masked_arr)
    #print(zmin, zmax, ymin, ymax, xmin, xmax)

    ## choose masked image or whole image in bbox
    if input_type == 'masked_img':
        img_bbox = masked_arr[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    elif input_type == 'raw_img':
        img_bbox = img_arr[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]

    # padding to match max bbox
    #--------------------------
    z_pad = z_max - img_bbox.shape[0]
    y_pad = y_max - img_bbox.shape[1]
    x_pad = x_max - img_bbox.shape[2]
    # keep consistent bbox size
    pad_ls = []
    pad_rs = []
    for pad in [z_pad, y_pad, x_pad]:
        assert pad >= 0, print('pad:', pad, img_bbox.shape)
        if pad % 2 == 0:
            pad_l = pad_r = pad // 2
        else:
            pad_l = pad // 2
            pad_r = pad // 2 + 1
        pad_ls.append(pad_l)
        pad_rs.append(pad_r)
    # numpy padding
    img_pad = np.pad(
        array=img_bbox, 
        pad_width=((pad_ls[0], pad_rs[0]), 
                   (pad_ls[1], pad_rs[1]), 
                   (pad_ls[2], pad_rs[2])), 
        mode='constant',
        constant_values=[(0, 0), (0, 0), (0, 0)]
        )
    #print(img_pad.shape)

    # normalize CT image
    #---------------------
    data = img_pad
    data[data <= -1024] = -1024
    # strip skull, skull UHI = ~700
    data[data > 700] = 0
    # normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
    if norm_type == 'np_interp':
        norm_data = np.interp(data, [-200, 200], [0, 1])
    elif norm_type == 'np_clip':
        data = np.clip(data, a_min=-200, a_max=200)
        MAX, MIN = data.max(), data.min()
        norm_data = (data - MIN) / (MAX - MIN)
    
    # reshape arr for inputs for CNN
    #-----------------------------------------------------
    arr = norm_data
    if input_channel == 1:
        #img_arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        img_arr = arr
        #print('img_arr shape:', img_arr.shape)
    elif input_channel == 3:
        img_arr = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        #print(img_arr.shape)
        #img_arr = np.transpose(img_arr, (1, 2, 3, 0))
        #print('img_arr shape:', img_arr.shape)
    
    # choose image saving type: npy or nii
    if save_img_type == 'npy':
        img = img_arr
    elif save_img_type == 'nii':
        img = nib.Nifti1Image(img_arr, np.eye(4))
    
    return img
