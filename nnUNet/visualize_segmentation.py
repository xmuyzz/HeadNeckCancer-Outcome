import os
import glob
import sys
from scipy import ndimage
import numpy as np
import pandas as pd
import SimpleITK as sitk
from get_data import get_data
from utils import threshold, get_spacing, calculate_metrics
from plot_images import plot_images



def visualize_seg(img_path, gt_path, pr_path, output_dir):
    """
    Test segmentation model and save prediction results
    Args:
        proj_dir {path} -- project dir;
        model_name {string} -- UNet model name;
        image_type {string} -- ct or others;
        hausdorff_percent {number} -- hausdorff_percent;
        overlap_tolerance {number} -- overlap_tolerance;
        surface_dice_tolerance {number} -- surface_dice_tolerance;
    Returns:
        Dice score, prediction maps;
    Raise errors:
        None;
    """
    
    img_dirs = [i for i in sorted(glob.glob(img_path + '/*0000.nii.gz'))]
    gt_dirs = [i for i in sorted(glob.glob(gt_path + '/*nii.gz'))]
    pr_dirs = [i for i in sorted(glob.glob(pr_path + '/*nii.gz'))]
    for img_dir, gt_dir, pr_dir in zip(img_dirs, gt_dirs, pr_dirs): 
        patient_id = gt_dir.split('/')[-1].split('.')[0]
        img_sitk_obj = sitk.ReadImage(img_dir)
        gt_sitk_obj = sitk.ReadImage(gt_dir)
        pr_sitk_obj = sitk.ReadImage(pr_dir)
        spacing = (1, 1, 3)
        #print("img shape:", img.shape)
        #img = img.reshape(1, 1, *img.shape)
        #img = img.reshape(1, 1, 64, 160, 160)
        #print("img shape:", img.shape)
        # get arrays from data
        img_arr = sitk.GetArrayFromImage(img_sitk_obj)
        gt_arr = sitk.GetArrayFromImage(gt_sitk_obj)
        pr_arr = sitk.GetArrayFromImage(pr_sitk_obj)
        if pr_arr[pr_arr==1].sum() > 0:
            result, dice, bbox_metrics = calculate_metrics(
                patient_id=patient_id, 
                spacing=spacing, 
                label_arr_org=gt_arr, 
                pred_arr_org=pr_arr, 
                hausdorff_percent=95, 
                overlap_tolerance=5,
                surface_dice_tolerance=5)    
            dice = round(dice, 3)
            # plot 5x3 views
            try:
                plot_images(
                    dataset='Hecktor',
                    patient_id=patient_id,
                    data_arr=img_arr,
                    gt_arr=gt_arr,
                    pred_arr=pr_arr,
                    output_dir=output_dir, 
                    bbox_flag=True,
                    bbox_metrics=bbox_metrics,
                    dice=dice)
                print ("{}, dice: {}".format(patient_id, dice))
            except Exception as e:
                print(e)


if __name__ == '__main__':
    
    proj_dir = '/mnt/aertslab/USERS/Zezhong/hecktor2022/DATA2'
    img_path = proj_dir + '/nnUNet_raw_data_base/nnUNet_raw_data/Task501_Hecktor/imagesTs'
    gt_path = proj_dir + '/nnUNet_raw_data_base/nnUNet_raw_data/Task501_Hecktor/labelsTs'
    pr_path = proj_dir + '/results_test'
    output_dir = proj_dir + '/visualization/Task501'

    visualize_seg(img_path, gt_path, pr_path, output_dir)








