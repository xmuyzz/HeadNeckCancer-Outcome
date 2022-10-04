import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
from bbox import get_bbox_3D


def max_bbox(data_dir, tumor_type):


    """
    get the max lenths of w, h, d of bbox
    
    Args:
      tumor_type - required: tumor + node or tumor;
      Cdata_dir  - required: tumor+node label dir CHUM cohort;
    
    Returns:
        lenths of width, height and depth of max bbox.
    """
    
    # primary and node
    CHUM_seg_pn_dir = os.path.join(data_dir, 'CHUM_files/label_reg')
    CHUS_seg_pn_dir = os.path.join(data_dir, 'CHUS_files/label_reg')
    PMH_seg_pn_dir = os.path.join(data_dir, 'PMH_files/label_reg')
    MDACC_seg_pn_dir = os.path.join(data_dir, 'MDACC_files/label_reg')
    # primary
    CHUM_seg_p_dir = os.path.join(data_dir, 'CHUM_files/label_p_reg')
    CHUS_seg_p_dir = os.path.join(data_dir, 'CHUS_files/label_p_reg')
    PMH_seg_p_dir = os.path.join(data_dir, 'PMH_files/label_p_reg')
    MDACC_seg_p_dir = os.path.join(data_dir, 'MDACC_files/label_p_reg')
    # node
    CHUM_seg_n_dir = os.path.join(data_dir, 'CHUM_files/label_n_reg')
    CHUS_seg_n_dir = os.path.join(data_dir, 'CHUS_files/label_n_reg')
    PMH_seg_n_dir = os.path.join(data_dir, 'PMH_files/label_n_reg')
    MDACC_seg_n_dir = os.path.join(data_dir, 'MDACC_files/label_n_reg')

    ## tumor type: tumor + node or tumor
    if tumor_type == 'primary_node':
        dirs = [
            CHUM_seg_pn_dir, 
            CHUS_seg_pn_dir, 
            PMH_seg_pn_dir, 
            MDACC_seg_pn_dir]
    elif tumor_type == 'primary':
        dirs = [
            CHUM_seg_p_dir, 
            CHUS_seg_p_dir, 
            PMH_seg_p_dir, 
            MDACC_seg_p_dir]
    elif tumor_type == 'node':
        dirs = [
            CHUM_seg_n_dir, 
            CHUS_seg_n_dir, 
            PMH_seg_n_dir, 
            MDACC_seg_n_dir]

    ## append all label lists to a large list
    seg_dirss = []
    for dir in dirs:
        seg_dirs = [path for path in sorted(glob.glob(dir + '/*nrrd'))]
        seg_dirss.extend(seg_dirs)
    
    ## get the max lengths of r, c, z
    count = 0
    d_lens = []
    h_lens = []
    w_lens = []
    empty_segs = []
    for seg_dir in seg_dirss:
        count += 1
        #print(count)
        seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
        seg_arr = sitk.GetArrayFromImage(seg)
        #print(label_dir.split('/')[-1])
        #print(label_arr.shape)
        if np.any(seg_arr):
            dmin, dmax, hmin, hmax, wmin, wmax = get_bbox_3D(seg_arr)
            d_len = dmax - dmin
            h_len = hmax - hmin
            w_len = wmax - wmin
            d_lens.append(d_len)
            h_lens.append(h_len)
            w_lens.append(w_len)
        elif not np.any(seg_arr):
            print('empty seg file:', seg_dir.split('/')[-1])
            empty_segs.append(seg_dir.split('/')[-1])
            continue
    
    d_max = max(d_lens)
    h_max = max(h_lens)
    w_max = max(w_lens)
    print('d_max:', d_max)
    print('h_max:', h_max)
    print('w_max:', w_max)
    
    print(empty_segs)

    return d_max, h_max, w_max


if __name__ == '__main__':

    data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated'
    tumor_type = 'primary_node'
    
    print(tumor_type)
    max_bbox(
        data_dir=data_dir,
        tumor_type=tumor_type
        )






