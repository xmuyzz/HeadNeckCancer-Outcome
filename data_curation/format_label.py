import os
import pandas as pd
import numpy as np
import glob
import SimpleITK as sitk
import nibabel as nib


def format_label(proj_dir, image_format):
    
    seg_dirs = [i for i in sorted(glob.glob(proj_dir + '/raw_seg/*nii.gz'))]
    out_dir = proj_dir + '/raw_seg'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, seg_dir in enumerate(seg_dirs):
        seg_id = seg_dir.split('/')[-1].split('.')[0]
        print(i, seg_id)
        seg_img = sitk.ReadImage(seg_dir)
        seg_arr = sitk.GetArrayFromImage(seg_img)
        if 3 in seg_arr:
            print('problematic mask:', seg_id)
            seg_arr[seg_arr == 3] = 1
            seg_arr = seg_arr.astype(int)
            sitk_obj = sitk.GetImageFromArray(seg_arr)
            sitk_obj.SetSpacing(seg_img.GetSpacing())
            sitk_obj.SetOrigin(seg_img.GetOrigin())
            writer = sitk.ImageFileWriter()
            writer.SetFileName(out_dir + '/' + seg_id + '.' + image_format)
            writer.SetUseCompression(True)
            writer.Execute(sitk_obj)


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/Hecktor_TCIA_data'
    image_format = 'nii.gz'
    #proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task508_P_N'
    format_label(proj_dir, image_format)


