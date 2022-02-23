import SimpleITK as sitk
import sys
import os
import numpy as np


def respacing(nrrd_dir, interp_type, new_spacing, patient_id, return_type, save_dir): 
    
    ### calculate new spacing
    img = sitk.ReadImage(nrrd_dir)
    old_size = img.GetSize()
    old_spacing = img.GetSpacing()
    #print('{} {}'.format('old size: ', old_size))
    #print('{} {}'.format('old spacing: ', old_spacing))

    new_size = [
        int(round((old_size[0] * old_spacing[0]) / float(new_spacing[0]))),
        int(round((old_size[1] * old_spacing[1]) / float(new_spacing[1]))),
        int(round((old_size[2] * old_spacing[2]) / float(new_spacing[2])))
        ]

    #print('{} {}'.format('new size: ', new_size))

    ### choose interpolation algorithm
    if interp_type == 'linear':
        interp_type = sitk.sitkLinear
    elif interp_type == 'bspline':
        interp_type = sitk.sitkBSpline
    elif interp_type == 'nearest_neighbor':
        interp_type = sitk.sitkNearestNeighbor
    
    ### interpolate
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputDirection(img.GetDirection())
    resample.SetInterpolator(interp_type)
    resample.SetDefaultPixelValue(img.GetPixelIDValue())
    resample.SetOutputPixelType(sitk.sitkFloat32)
    img_nrrd = resample.Execute(img) 
    
    ## save nrrd images
    if save_dir != None:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(save_dir, '{}.nrrd'.format(patient_id)))
        writer.SetUseCompression(True)
        writer.Execute(img_nrrd)

    ## save as numpy array
    img_arr = sitk.GetArrayFromImage(img_nrrd)

    if return_type == 'nrrd':
        return img_nrrd
    
    elif return_type == 'npy':
        return img_arr

