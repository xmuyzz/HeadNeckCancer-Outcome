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
from get_data import test_data
from utils import (generate_sitk_obj_from_npy_array, threshold, get_spacing, calculate_metrics, 
                   save_candidate_roi, multi_prediction, get_lr_metric)
from plot_images import plot_images
from losses import (precision_loss, dice_loss, tversky_loss, focal_tversky_loss, bce_loss, 
                    bce_dice_loss, wce_dice_loss)
from opts import parse_opts



def test(opts):
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.config.list_physical_devices('GPU')
    print('\nNum GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # get test data
    ts1_data, ts2_data = test_data(proj_dir=opt.proj_dir, crop_shape=opt.image_shape)
    print('\nsuccessfully load test data!')
    
    # get model
    model_dir = opt.proj_dir + '/keras_seg_model/log/' + opt.model_name
    unet_model = load_model(
        model_dir, 
        custom_objects={
            'InstanceNormalization': InstanceNormalization, 
            'wce_dice_loss': wce_dice_loss, 
            'lr': get_lr_metric})
    # prediction
    if opt.test_set == 'test1':
        pred_dir = opt.proj_dir + '/keras_seg_model/output/pred1'
        visual_dir = opt.proj_dir + '/keras_seg_model/output/visual1'
        ts_data = ts1_data
    elif opt.test_set == 'test2':
        pred_dir = opt.proj_dir + '/keras_seg_model/output/pred2'
        visual_dir = opt.proj_dir + '/keras_seg_model/output/visual2'
        ts_data = ts2_data
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    results = []
    no_results = []
    #print(data)
    for data in ts_data:
        pat_id = data['patient_id']
        #print(pat_id)
        img = data['img_arr']
        img_sitk_obj = data['img_sitk_obj']
        #image_sitk_obj = sitk.ConstantPadImageFilter(image_sitk_obj,(172,172,76)
        seg_sitk_obj = data['seg_sitk_obj']
        spacing = get_spacing(img_sitk_obj)
        print('img shape:', img.shape)
        img = img.reshape(1, 1, *img.shape)
        #img = img.reshape(1, 1, 64, 160, 160)
        print('img shape:', img.shape)
        #seg_pred = original_model.predict(img)
        seg_pred = unet_model.predict(
            img,
            batch_size=None,
            verbose=1,
            workers=1,
            use_multiprocessing=False)
        seg_pred = threshold(np.squeeze(seg_pred)) # 0.5
        # if there are voxels predicted:
        if seg_pred[seg_pred==1].sum() > 0: 
            # save model output as nrrd
            pred_sitk_obj = generate_sitk_obj_from_npy_array(
                img_sitk_obj,
                seg_pred,
                True,
                pred_dir + '/' + pat_id + '.nrrd')
            print('prediction nrrd saved:', pat_id)
            # get arrays from data
            img_arr = sitk.GetArrayFromImage(img_sitk_obj)
            seg_arr = sitk.GetArrayFromImage(seg_sitk_obj)
            pred_arr = sitk.GetArrayFromImage(pred_sitk_obj)
            #i = 2
            #seg_arr = np.where(seg_arr == i, 1, 0)
            #pred_arr = np.where(pred_arr == i, 1, 0)
            # metrics
            try:
                result, dice, bbox_metrics = calculate_metrics(
                    patient_id=pat_id, 
                    spacing=spacing, 
                    label_arr=seg_arr, 
                    pred_arr=pred_arr, 
                    hausdorff_percent=opt.hausdorff_percent, 
                    overlap_tolerance=opt.overlap_tolerance,
                    surface_dice_tolerance=opt.surface_dice_tolerance)    
                results.append(result)
                # plot 5x3 views
                if opt.plot_img:
                    plot_images(
                        dataset='CT',
                        patient_id=pat_id,
                        data_arr=img_arr_org,
                        gt_arr=seg_arr_org,
                        pred_arr=pred_arr_org,
                        output_dir=visual_dir, 
                        bbox_flag=True,
                        bbox_metrics=bbox_metrics,
                        dice=dice)
                print ('{}, dice: {}'.format(pat_id, result['dice'])) 
                no_results.append(pat_id)
            except Exception as e:
                print(pat_id, e)
    print('no results :: ', no_results)
    df = pd.DataFrame.from_dict(results)
    df.to_csv(opt.proj_dir + '/keras_seg_model/output/seg_results.csv')
    print('median Dice overall:', np.median(df['dice']))
    print('median surface Dice:', np.median(df['surface_dice']))



if __name__ == '__main__':
    
    opt = parse_opts()

    test(opt)







