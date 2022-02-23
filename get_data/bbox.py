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

    #z = np.any(img, axis=(0, 1))
    #r = np.any(img, axis=(1, 2))
    #c = np.any(img, axis=(0, 2))
    d = np.any(img, axis=(1, 2))
    h = np.any(img, axis=(0, 2))
    w = np.any(img, axis=(0, 1))
    #print('z:', z)
    #print('y:', y)
    #print('x:', x)
    dmin, dmax = np.where(d)[0][[0, -1]]
    hmin, hmax = np.where(h)[0][[0, -1]]
    wmin, wmax = np.where(w)[0][[0, -1]]

    dmin = int(dmin)
    dmax = int(dmax)
    hmin = int(hmin)
    hmax = int(hmax)
    wmin = int(wmin)
    wmax = int(wmax)
    
    #print('dmin:', dmin)
    #print('dmax:', dmax)
    #print('hmin:', hmin)
    #print('hmax:', hmax)
    #print('wmin:', wmin)
    #print('wmax:', wmax)

    return dmin, dmax, hmin, hmax, wmin, wmax

def bbox_3D(img):

    """
    Returns bounding box fit to the boundaries of non-zeros

    z: z direction
    y: y direction
    x: x direction
    """

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    xmin, xmax = np.where(x)[0][[0, -1]]
    zmin = int(zmin)
    zmax = int(zmax)
    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)
    xmax = int(xmax)

    return zmin, zmax, ymin, ymax, xmin, xmax


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
