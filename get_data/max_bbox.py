import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
from get_data.bbox import bbox_3D


def max_bbox(data_dir, tumor_type):


    """
    get the max lenths of r, c, z of bbox
    
    @ params:
      tumor_type - required: tumor + node or tumor
      Cdata_dir  - required: tumor+node label dir CHUM cohort
    """
    
    CHUM_seg_pn_dir = os.path.join(data_dir, 'CHUM_files/label_reg')
    CHUS_seg_pn_dir = os.path.join(data_dir, 'CHUS_files/label_reg')
    PMH_seg_pn_dir = os.path.join(data_dir, 'PMH_files/label_reg')
    MDACC_seg_pn_dir = os.path.join(data_dir, 'MDACC_files/label_reg')
    CHUM_seg_p_dir = os.path.join(data_dir, 'CHUM_files/label_p_reg')
    CHUS_seg_p_dir = os.path.join(data_dir, 'CHUS_files/label_p_reg')
    PMH_seg_p_dir = os.path.join(data_dir, 'PMH_files/label_p_reg')
    MDACC_seg_p_dir = os.path.join(data_dir, 'MDACC_files/label_p_reg')
    
    ## tumor type: tumor + node or tumor
    if tumor_type == 'primary_node':
        dirs = [
            CHUM_seg_pn_dir, 
            CHUS_seg_pn_dir, 
            PMH_seg_pn_dir, 
            MDACC_seg_pn_dir
            ]
    elif tumor_type == 'primary':
        dirs = [
            CHUM_seg_p_dir, 
            CHUS_seg_p_dir, 
            PMH_seg_p_dir, 
            MDACC_seg_p_dir
            ]
    
    ## append all label lists to a large list
    seg_dirss = []
    for dir in dirs:
        seg_dirs = [path for path in sorted(glob.glob(dir + '/*nrrd'))]
        seg_dirss.extend(seg_dirs)
    
    ## get the max lengths of r, c, z
    count = 0
    z_lens = []
    y_lens = []
    x_lens = []
    empty_segs = []
    for seg_dir in seg_dirss:
        count += 1
        #print(count)
        seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
        seg_arr = sitk.GetArrayFromImage(seg)
        #print(label_dir.split('/')[-1])
        #print(label_arr.shape)
        if np.any(seg_arr):
            zmin, zmax, ymin, ymax, xmin, xmax = bbox_3D(seg_arr)
            z_len = zmax - zmin
            y_len = ymax - ymin
            x_len = xmax - xmin
            z_lens.append(z_len)
            y_lens.append(y_len)
            x_lens.append(x_len)
        elif not np.any(seg_arr):
            print('empty seg file:', seg_dir.split('/')[-1])
            empty_segs.append(seg_dir.split('/')[-1])
            continue
    
    ## get the max lengths of r, c, z
    #print('r:', r_lens)
    #print('c:', c_lens)
    #print('z:', z_lens)
    z_max = max(z_lens)
    y_max = max(y_lens)
    x_max = max(x_lens)
    print('z_max:', z_max)
    print('y_max:', y_max)
    print('x_max:', x_max)
    
    print(empty_segs)

    return z_max, y_max, x_max


