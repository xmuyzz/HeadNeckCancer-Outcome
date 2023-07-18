import os
import glob
import sys
from scipy import ndimage
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
from get_data import get_data
from utils import (generate_sitk_obj_from_npy_array, threshold, get_spacing, calculate_metrics, 
                   save_candidate_roi, multi_prediction, get_lr_metric)
from plot_images import plot_images
from losses import (precision_loss, dice_loss, tversky_loss, focal_tversky_loss, bce_loss, 
                    bce_dice_loss, wce_dice_loss)
from opts import parse_opts



def main(opts):
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

    # check tf version and GPU
    print('\ntf version:', tf. __version__)
    tf.test.gpu_device_name()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.config.list_physical_devices('GPU')
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # get test data
    test_set = 'DFCI'
    if test_set == 'HGJ':
        test_img_dir = opt.proj_dir + '/TCIA/HGJ_HMR_data/crop_img_160x160x64'
        test_seg_dir = opt.proj_dir + '/TCIA/HGJ_HMR_data/crop_seg_n_160x160x64'
    elif test_set == 'DFCI':
        test_img_dir = opt.proj_dir + '/DFCI/new_curation/crop_img_160'
        test_seg_dir = opt.proj_dir + '/DFCI/new_curation/crop_seg_n_160'
    data = get_data(
        test_img_dir=test_img_dir, 
        test_seg_dir=test_seg_dir)
    print('\nsuccessfully load test data!')
    # get model
    model_dir = opt.proj_dir + '/keras_seg_model/' + opt.model_name
    original_model = load_model(
        model_dir, 
        custom_objects={
            'InstanceNormalization': InstanceNormalization, 
            'wce_dice_loss': wce_dice_loss, 
            'lr': get_lr_metric})
    # prediction
    if test_set == 'HGJ':
        pred_dir = opt.proj_dir + '/keras_seg_model/HGJ/pred'
        output_dir = opt.proj_dir + '/keras_seg_model/HGJ/output'
    elif test_set == 'DFCI':
        pred_dir = opt.proj_dir + '/keras_seg_model/DFCI/pred'
        output_dir = opt.proj_dir + '/keras_seg_model/DFCI/output'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = []
    no_results = []
    #print(data)
    for patient in data:
        pat_id = patient['patient_id']
        #print(pat_id)
        img = patient['image']
        img_sitk_obj = patient['image_sitk_obj']
        #image_sitk_obj = sitk.ConstantPadImageFilter(image_sitk_obj,(172,172,76)
        seg_sitk_obj = patient['seg_sitk_obj']
        spacing = get_spacing(img_sitk_obj)
        print('img shape:', img.shape)
        img = img.reshape(1, 1, *img.shape)
        #img = img.reshape(1, 1, 64, 160, 160)
        print('img shape:', img.shape)
        #seg_pred = original_model.predict(img)
        seg_pred = original_model.predict(
            img,
            batch_size=None,
            verbose=1,
            workers=1,
            use_multiprocessing=False)
        seg_pred = threshold(np.squeeze(seg_pred)) # 0.5
        # if there are voxels predicted:
        if seg_pred[seg_pred==1].sum() > 0: 
            # save model output as nrrd
            # this will pad the prediction to match the size of the originals
            # for localization, 80, 96, 96 => 84, 108, 108
            # for segmentation, 64, 160, 160 => 76, 196, 196
            pred_sitk_obj = generate_sitk_obj_from_npy_array(
                img_sitk_obj,
                seg_pred,
                True,
                pred_dir + '/' + pat_id + '.nrrd')
            print('prediction nrrd saved:', pat_id)
            # get arrays from data
            img_arr_org = sitk.GetArrayFromImage(img_sitk_obj)
            seg_arr_org = sitk.GetArrayFromImage(seg_sitk_obj)
            pred_arr_org = sitk.GetArrayFromImage(pred_sitk_obj)
            # metrics
            result, dice, bbox_metrics = calculate_metrics(
                patient_id=pat_id, 
                spacing=spacing, 
                label_arr_org=seg_arr_org, 
                pred_arr_org=pred_arr_org, 
                hausdorff_percent=opt.hausdorff_percent, 
                overlap_tolerance=opt.overlap_tolerance,
                surface_dice_tolerance=opt.surface_dice_tolerance)    
            results.append(result)
            # plot 5x3 views
            do_plot_imgs = False
            if do_plot_imgs:
                plot_images(
                    dataset='CT',
                    patient_id=pat_id,
                    data_arr=img_arr_org,
                    gt_arr=seg_arr_org,
                    pred_arr=pred_arr_org,
                    output_dir=output_dir, 
                    bbox_flag=True,
                    bbox_metrics=bbox_metrics,
                    dice=dice)
            print ('{}, dice: {}'.format(pat_id, result['dice']))
            
            no_results.append(pat_id)
          #  # temporary for segmentation task
          #  result = {}
          #  result["patient_id"] = patient_id
          #  result["precision"] = 0
          #  result["recall"] = 0
          #  result["jaccard"] = 0
          #  result["dice"] = 0
          #  result["segmentation_score"] = 0
          #  result["x_distance"] = 0
          #  result["y_distance"] = 0
          #  result["z_distance"] = 0
          #  result["distance"] = 0
          #  result["average_surface_distance_gt_to_pr"] = 0
          #  result["average_surface_distance_pr_to_gt"] = 0
          #  result["robust_hausdorff"] = 0
          #  result["overlap_fraction_gt_with_pr"] = 0
          #  result["overlap_fraction_pr_with_gt"] = 0
          #  result["surface_dice"] = 0
          #  for axes in ["X", "Y", "Z"]:
          #      for location in ["min", "center", "max", "length"]:
          #          result["prediction_{}_{}".format(axes, location)] = 0
          #  results.append(result)
    print('no results :: ', no_results)
    df = pd.DataFrame.from_dict(results)
    df.to_csv(output_dir + '/seg_results.csv')
    print('median Dice overall:', np.median(df['dice']))
    print('median surface Dice:', np.median(df['surface_dice']))



if __name__ == '__main__':
    
    opt = parse_opts()

    main(opt)







