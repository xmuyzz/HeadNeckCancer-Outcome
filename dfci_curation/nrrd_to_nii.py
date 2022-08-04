import sys
import os
import pydicom
import glob
import SimpleITK as sitk
import nibabel as nib
import numpy as np


def nrrd_to_nii(proj_dir):

    """
    save nrrd to nii 
    """

    img_nrrd_dir = proj_dir + '/DFCI/img_crop2'
    seg_nrrd_dir = proj_dir + '/DFCI/seg_crop2'
    img_nii_dir = proj_dir + '/DFCI/img_nii'
    seg_nii_dir = proj_dir + '/DFCI/seg_nii'
    if not os.path.exists(img_nii_dir):
        os.makedirs(img_nii_dir)
    if not os.path.exists(seg_nii_dir):
        os.makedirs(seg_nii_dir)
    ## save img nii
    for count, img_dir in enumerate(sorted(glob.glob(img_nrrd_dir + '/*nrrd'))):
        img_id = img_dir.split('/')[-1].split('.')[0]
        print(count, img_id)
        fn = str(img_id) + '.nii.gz'
        img = sitk.ReadImage(img_dir)
        img = sitk.GetArrayFromImage(img)
        img = nib.Nifti1Image(img, affine=np.eye(4))
        nib.save(img, os.path.join(img_nii_dir, fn))
    ## save seg nii
    for count, seg_dir in enumerate(sorted(glob.glob(seg_nrrd_dir + '/*nrrd'))):
        seg_id = seg_dir.split('/')[-1].split('.')[0]
        print(count, seg_id)
        fn = str(seg_id) + '.nii.gz'
        seg = sitk.ReadImage(seg_dir)
        seg = sitk.GetArrayFromImage(seg)
        seg = nib.Nifti1Image(seg, affine=np.eye(4))
        nib.save(seg, os.path.join(seg_nii_dir, fn))


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    nrrd_to_nii(proj_dir)




