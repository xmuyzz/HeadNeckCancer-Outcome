import numpy as np
import os
import glob             
import pandas as pd                                                      
import SimpleITK as sitk

def get_bbox_3D(img):

    """
    Returns bounding box fit to the boundaries of non-zeros
    
    r: row
    c: column
    z: z direction
    """
    
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    rmin = int(rmin)
    rmax = int(rmax)
    cmin = int(cmin)
    cmax = int(cmax)
    zmin = int(zmin)
    zmax = int(zmax)

    return rmin, rmax, cmin, cmax, zmin, zmax


def get_bbox(mask_data):
    
    """
    Returns min, max, length, and centers across Z, Y, and X. (12 values)
    """
    
    # crop maskData to only the 1's
    # http://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    # maskData order is z,y,x because we already rolled it
    Z = np.any(mask_data, axis=(1, 2))
    Y = np.any(mask_data, axis=(0, 2))
    X = np.any(mask_data, axis=(0, 1))
    #
    Z_min, Z_max = np.where(Z)[0][[0, -1]]
    Y_min, Y_max = np.where(Y)[0][[0, -1]]
    X_min, X_max = np.where(X)[0][[0, -1]]
    # 1 is added to account for the final slice also including the mask
    
    return Z_min, Z_max, Y_min, Y_max, X_min, X_max, Z_max-Z_min+1, Y_max-Y_min+1, \
           X_max-X_min+1, (Z_max-Z_min)/2 + Z_min, (Y_max-Y_min)/2 + Y_min, (X_max-X_min)/2 + X_min

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
