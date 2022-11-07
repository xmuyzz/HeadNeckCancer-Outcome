import numpy as np
import os
import glob
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from get_data.bbox import get_bbox_3D
from get_data.resize_3d import resize_3d
from get_data.respacing import respacing


def bbox_img(img_dir, seg_dir, patient_id, max_bbox=(70, 70, 70), output_dir):

    """
    get cnosistent 3D tumor&node data using masking, bbox and padding
    Args:
      img_dir {path} -- dir for image in nrrd format
      label_dir {path} -- dir for label in nrrd format
    Returns:
        Preprocessed images in nii.gz or np.array formats;
    """

    img = sitk.ReadImage(img_dir)
    img_arr = sitk.GetArrayFromImage(img)
    seg = sitk.ReadImage(seg_dir)
    seg_arr = sitk.GetArrayFromImage(seg)
 
    # get 3d bounding box
    #--------------------
    dmin, dmax, hmin, hmax, wmin, wmax = get_bbox_3D(seg_arr)

    # choose mask_img or bbox_img
    if input_img_type == 'mask_img':
        #apply mask to image
        masked_arr = np.where(seg_arr==1, img, seg_arr)
        #masked_arr = norm_img * seg_arr
        img_bbox = masked_arr[dmin:dmax+1, hmin:hmax+1, wmin:wmax+1]
    elif input_img_type == 'bbox_img':
        img_bbox = img_arr[dmin:dmax+1, hmin:hmax+1, wmin:wmax+1]
    #print('masked_arr:', masked_arr.shape)
    #print('img_bbox:', img_bbox.shape)
    
    # padding to match max bbox
    #--------------------------
    d_pad = max_bbox[0] - img_bbox.shape[0]
    h_pad = max_bbox[1] - img_bbox.shape[1]
    w_pad = max_bbox[2] - img_bbox.shape[2]
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
        constant_values=[(0, 0), (0, 0), (0, 0)])
    #print('img_pad:', img_pad.shape)
    output_img = img_pad
    #img = img_pad.transpose(1, 2, 0)
    #img = np.transpose(img_pad, axes=[1, 2, 0])
    
    fn = output_dir + '/' + patient_id + '.nii.gz'
    output_img.SetSpacing(img.GetSpacing())
    output_img.SetOrigin(img.GetOrigin())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(fn)
    writer.SetUseCompression(True)
    writer.Execute(output_img)
    
    return output_img


def save_bbox_img(proj_dir, tumor_type):
    
    bbox_img_pn_dir = proj_dir + '/data/TOT/bbox_img_pn'
    bbox_img_p_dir = proj_dir + '/data/TOT/bbox_img_p'
    bbox_img_n_dir = proj_dir + '/data/TOT/bbox_img_n'
    if not os.path.exists(bbox_img_pn_dir):
        os.makedirs(bbox_img_pn_dir)
    if not os.path.exists(bbox_img_p_dir):
        os.makedirs(bbox_img_p_dir)
    if not os.path.exists(bbox_img_n_dir):
        os.makedirs(bbox_img_n_dir)
    if tumor_type == 'pn':
        save_dir = bbox_img_pn_dir
    elif tumor_type == 'p':
        save_dir = bbox_img_p_dir
    elif tumor_type == 'n':
        save_dir = bbox_img_n_dir

    img_dirs = [i for i in sorted(glob.glob(tot_img_dir + '/*nii.gz'))]
    seg_dirs = [i for i in sorted(glob.glob(tot_seg_dir + '/*nii.gz'))]
    img_ids = []
    bad_ids = []
    count = 0
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        for seg_dir in seg_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                print(count, img_id)
                bbox_img(
                    img_dir, 
                    seg_dir, 
                    patient_id=img_id, 
                    max_bbox=(70, 70, 70), 
                    output_dir=save_dir)



