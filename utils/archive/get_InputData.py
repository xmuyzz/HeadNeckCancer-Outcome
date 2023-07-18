import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk



def bbox2(img):
    r = np.any(img, axis=1)
    c = np.any(img, axis=0)
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

def get_bounding_box(x):
    """ Calculates the bounding box of a ndarray"""
    mask = x == 0
    bbox = []
    all_axis = np.arange(x.ndim)
    for kdim in all_axis:
        nk_dim = np.delete(all_axis, kdim)
        mask_i = mask.all(axis=tuple(nk_dim))
        dmask_i = np.diff(mask_i)
        idx_i = np.nonzero(dmask_i)[0]
        if len(idx_i) != 2:
            raise ValueError('Algorithm failed, {} does not have 2 elements!'.format(idx_i))
        bbox.append(slice(idx_i[0]+1, idx_i[1]+1))
    return bbox

test_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/test'
img_dir = '//mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated/CHUM_files/image_reg'
label_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated/CHUM_files/label_reg'
img_file = 'HN-CHUM-001_registered.nrrd'
label_file = 'HN-CHUM-001_label_registered.nrrd'

## apply mask to image
img = sitk.ReadImage(os.path.join(img_dir, img_file), sitk.sitkFloat32)
label = sitk.ReadImage(os.path.join(label_dir, label_file), sitk.sitkFloat32)
img_arr = sitk.GetArrayFromImage(img)
label_arr = sitk.GetArrayFromImage(label)
masked_arr = img_arr*label_arr
masked_arr2 = np.where(label_arr==1, img_arr, label_arr)
print('img arr:', img_arr[:, :, 68])
#print('label arr:', label_arr[:, :, 68])
#print('masked_arr:', masked_arr[:, :, 68])
#print('masked_arr2:', masked_arr2[:, :, 68])

## bounding box
rmin, rmax, cmin, cmax, zmin, zmax = bbox2(masked_arr2)
print(rmin, rmax, cmin, cmax, zmin, zmax)
#bbox = bbox2(masked_arr2)
img_bbox = masked_arr2[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]
print('img_bbox:', img_bbox)

#bbox = get_bounding_box(masked_arr2)
#print(bbox)
#print(len(bbox))
#print((img_arr[bbox]!=0).astype(int))

## padding 
r_width = (r_max - img_bbox.shape[0]) // 2
c_width = (c_max - img_bbox.shape[1]) // 2
z_width = (z_max - img_bbox.shape[2]) // 2
img_pad = np.pad(img_bbox, (r_width, c_width, z_width), 'constant')

x = np.array([[1,2], [3,4]])
y = np.pad(x, (2, 2), 'constant')
print(x)
print(y)
