import os
import pandas as pd
import numpy as np
import SimpleITK as sitk


def main(root_dir):
    
    proj_dir = root_dir + '/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task511_TOT_pn'
    seg_path1 = proj_dir + '/labelsTr'
    seg_path2 = proj_dir + '/labelsTs'
    seg_path3 = proj_dir + '/labelsTs2'
    img_path1 = proj_dir + '/imagesTr'
    img_path2 = proj_dir + '/imagesTs'
    img_path3 = proj_dir + '/imagesTs2'
    seg_paths = [seg_path1, seg_path2, seg_path3]
    img_paths = [img_path1, img_path2, img_path3]
    for seg_path, img_path in zip(seg_paths, img_paths):
        fns = [i for i in sorted(os.listdir(seg_path))]
        for i, fn in enumerate(fns):
            img_id = fn.split('.')[0]
            print(i, img_id)
            # image
            img_dir = img_path + '/' + img_id
            img = sitk.ReadImage(img_dir)
            img_arr = sitk.GetArrayFromImage(img)
            # primary tumor
            seg_dir = seg_path + '/' + fn
            seg = sitk.ReadImage(seg_dir)
            seg_arr = sitk.GetArrayFromImage(seg)
            seg_arr[seg_arr != 0] = 1
            sitk_obj = sitk.GetImageFromArray(seg_arr)
            sitk_obj.SetSpacing(img.GetSpacing())
            sitk_obj.SetOrigin(img.GetOrigin())
            # write new nrrd
            writer = sitk.ImageFileWriter()
            writer.SetFileName(seg_path + '/' + pat_id + '.nii.gz')
            writer.SetUseCompression(True)
            writer.Execute(sitk_obj)


def main2(root_dir):

    proj_dir = root_dir + '/HKTR_TCIA_DFCI/TOT'
    seg_path = proj_dir + '/crop_seg_160'
    img_path = proj_dir + '/crop_img_160'
    seg_pn_path = proj_dir + '/crop_seg_pn_160'
    if not os.path.exists(seg_pn_path):
        os.makedirs(seg_pn_path)
    for i, fn in enumerate(sorted(os.listdir(seg_path))):
        print(i, fn)
        # image
        img_dir = img_path + '/' + fn
        img = sitk.ReadImage(img_dir)
        img_arr = sitk.GetArrayFromImage(img)
        # primary tumor
        seg_dir = seg_path + '/' + fn
        seg = sitk.ReadImage(seg_dir)
        seg_arr = sitk.GetArrayFromImage(seg)
        seg_arr[seg_arr != 0] = 1
        sitk_obj = sitk.GetImageFromArray(seg_arr)
        sitk_obj.SetSpacing(img.GetSpacing())
        sitk_obj.SetOrigin(img.GetOrigin())
        # write new nrrd
        writer = sitk.ImageFileWriter()
        writer.SetFileName(seg_pn_path + '/' + fn + '.nii.gz')
        writer.SetUseCompression(True)
        writer.Execute(sitk_obj)

if __name__ == '__main__':

    root_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    
    #main(root_dir)
    main2(root_dir)








