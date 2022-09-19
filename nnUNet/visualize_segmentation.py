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


def visualize_seg2(img_path, gt_path, pr_path, output_dir):
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

    for j, (i, tumor_type) in enumerate(zip([1, 2], ['Primary', 'Node'])):
        for img_dir, gt_dir, pr_dir in zip(img_dirs, gt_dirs, pr_dirs): 
            patient_id = gt_dir.split('/')[-1].split('.')[0]
            img_sitk_obj = sitk.ReadImage(img_dir)
            gt_sitk_obj = sitk.ReadImage(gt_dir)
            pr_sitk_obj = sitk.ReadImage(pr_dir)
            img_arr = sitk.GetArrayFromImage(img_sitk_obj)
            gt_arr = sitk.GetArrayFromImage(gt_sitk_obj)
            pr_arr = sitk.GetArrayFromImage(pr_sitk_obj)
            gt_arr = np.where(gt_arr == i, 1, 0)
            pr_arr = np.where(pr_arr == i, 1, 0)
            if pr_arr[pr_arr==1].sum() > 0:
                try:
                    result, dice, bbox_metrics = calculate_metrics(
                        patient_id=patient_id, 
                        spacing=(1, 1, 3), 
                        label_arr_org=gt_arr, 
                        pred_arr_org=pr_arr, 
                        hausdorff_percent=95, 
                        overlap_tolerance=5,
                        surface_dice_tolerance=5)    
                    dice = round(dice, 3)
                    # plot 5x3 views
                    plot_images(
                        dataset='TCIA',
                        patient_id=patient_id,
                        tumor_type=tumor_type,
                        data_arr=img_arr,
                        gt_arr=gt_arr,
                        pred_arr=pr_arr,
                        output_dir=output_dir, 
                        bbox_flag=True,
                        bbox_metrics=bbox_metrics,
                        dice=dice)
                    print ("{}, dice: {}".format(patient_id, dice))
                except Exception as e:
                    print(patient_id, e)


def visualize_seg(img_path, gt_path, pr_path, output_dir):

    img_dirs = [i for i in sorted(glob.glob(img_path + '/*0000.nii.gz'))]
    gt_dirs = [i for i in sorted(glob.glob(gt_path + '/*nii.gz'))]
    pr_dirs = [i for i in sorted(glob.glob(pr_path + '/*nii.gz'))]
    for img_dir, gt_dir, pr_dir in zip(img_dirs, gt_dirs, pr_dirs):
        patient_id = gt_dir.split('/')[-1].split('.')[0]
        img_sitk_obj = sitk.ReadImage(img_dir)
        gt_sitk_obj = sitk.ReadImage(gt_dir)
        pr_sitk_obj = sitk.ReadImage(pr_dir)
        img_arr = sitk.GetArrayFromImage(img_sitk_obj)
        gt_arr = sitk.GetArrayFromImage(gt_sitk_obj)
        pr_arr = sitk.GetArrayFromImage(pr_sitk_obj)
        #gt_arr = np.where(gt_arr == 0, 1, 0)
        #pr_arr = np.where(pr_arr == 0, 1, 0)
        if pr_arr[pr_arr==1].sum() > 0:
            result, dice, bbox_metrics = calculate_metrics(
                patient_id=patient_id,
                spacing=(1, 1, 3),
                label_arr_org=gt_arr,
                pred_arr_org=pr_arr,
                hausdorff_percent=95,
                overlap_tolerance=5,
                surface_dice_tolerance=5)
            dice = round(dice, 3)
            # plot 5x3 views
            plot_images(
                dataset='TCIA',
                patient_id=patient_id,
                tumor_type='PN',
                data_arr=img_arr,
                gt_arr=gt_arr,
                pred_arr=pr_arr,
                output_dir=output_dir,
                bbox_flag=True,
                bbox_metrics=bbox_metrics,
                dice=dice)
            print ("{}, dice: {}".format(patient_id, dice))

if __name__ == '__main__':
    
    task = 'Task507_PN'
    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data'
    img_path = proj_dir + '/' + task + '/imagesTs2'
    gt_path = proj_dir + '/' + task + '/labelsTs2'
    pr_path = proj_dir + '/' + task + '/predsTs2'
    output_dir = proj_dir + '/' + task + '/output/visual'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print('folder exists!')

    visualize_seg(img_path, gt_path, pr_path, output_dir)








