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
    ## get the max lengths of r, c, z
    count = 0
    d_lens = []
    h_lens = []
    w_lens = []
    empty_segs = []
    for seg_path in sorted(glob.glob(seg_dir + '/*nrrd')):
        count += 1
        #print(count)
        seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)
        seg_arr = sitk.GetArrayFromImage(seg)
        if tumor_type == 'pn':
            seg_arr = 
        elif tumor_type == 'p':
            seg_arr =
        elif tumor_type == 'n':
            seg_arr =
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
        else:
            print('empty seg file:', seg_path.split('/')[-1])
            empty_segs.append(seg_path.split('/')[-1])
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
        tumor_type=tumor_type)






