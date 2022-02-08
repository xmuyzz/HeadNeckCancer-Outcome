"""
  ----------------------------------------------
  DeepContrast - run DeepContrast pipeline step1
  ----------------------------------------------
  ----------------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.6.8
  ----------------------------------------------
  
  Deep-learning-based IV contrast detection
  in CT scans - all param.s are read
  from a config file stored under "/config"
  
"""


import numpy as np
import os
import glob
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from get_data.prepro_img import prepro_img
from get_data.max_bbox import max_bbox
from get_data.get_dir import get_dir
from get_data.bbox import bbox_3D


def save_img(save_img_type, save_dir, img, seg_fn):
    
    """
    save img as numpy or nii file
    """
    
    if save_img_type == 'npy':
        fn = str(seg_fn) + '.npy'
        np.save(os.path.join(save_dir, fn), img, allow_pickle=True)
    elif save_img_type == 'nii':
        fn = str(seg_fn) + '.nii.gz'
        nib.save(img, os.path.join(save_dir, fn))


def input_arr(data_dir, proj_dir, aimlab_dir, norm_type, tumor_type, input_type, input_channel, 
              run_max_bbox, img_dirss, seg_pn_dirss, toyset, save_img_type):

    """
    save np arr for masked img for CT scans
    
    args:
        tumor_type {'string'} - tumor + node or tumor
        data_dir {'path'} - tumor+node label dir CHUM cohort
        arr_dir {path} - tumor+node label dir CHUS cohort

    return:
        
    """
     
    pn_masked_img_dir = os.path.join(aimlab_dir, 'data/pn_masked_img')
    pn_raw_img_dir = os.path.join(aimlab_dir, 'data/pn_raw_img')
    p_masked_img_dir = os.path.join(aimlab_dir, 'data/PMH_files/p_masked_img')
    p_raw_img_dir = os.path.join(aimlab_dir, 'data/MDACC_files/p_raw_img')
    toyset_dir = os.path.join(aimlab_dir, 'data/toyset')

    if not os.path.exists(toyset_dir): os.mkdir(toyset_dir)
    if not os.path.exists(pn_raw_img_dir): os.mkdir(pn_raw_img_dir)

    ## get the max lenths of r, c, z of bbox
    #---------------------------------------
    if run_max_bbox==False:
        z_max = 64
        #y_max = 132
        y_max = 178
        x_max = 178
    else:
        z_max, y_max, x_max = max_bbox(
            data_dir=data_dir,
            tumor_type=tumor_type
            )
    
    ## choose tumor type and input img type
    #--------------------------------------
    if tumor_type == 'primary_node':
        print(tumor_type)
        seg_dirss = seg_pn_dirss
        if input_type == 'masked_img':
            print(input_type)
            if toyset:
                save_dir = toyset_dir
            else:
                save_dir = pn_masked_img_dir
    elif input_type == 'raw_img':
            print(input_type)
            save_dir = pn_raw_img_dir
    elif tumor_type == 'primary':
        print(tumor_type)
        seg_dirss = seg_p_dirss
        if input_type == 'masked_img':
            print(input_type)
            save_dir = p_masked_img_dir
        elif input_type == 'raw_img':
            print(input_type)
            save_dir = p_raw_img_dir
    
    # load image and label to get numpy arrays
    #------------------------------------------
    cohorts = ['CHUM', 'CHUS', 'PMH', 'MDACC'] 
    for cohort, img_dirs, seg_dirs in zip(cohorts, img_dirss, seg_dirss):
        ## CHUM and CHUS cohort
        if cohort in ['CHUM', 'CHUS']:
            print('CHUM and CHUS dataset:')
            for img_dir, seg_dir in zip(img_dirs, seg_dirs):
                ## img and seg numbers are not equal
                img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
                seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
                img_arr = sitk.GetArrayFromImage(img)
                seg_arr = sitk.GetArrayFromImage(seg)
                img_fn = img_dir.split('/')[-1].split('-')[1] + \
                         img_dir.split('/')[-1].split('-')[2].split('_')[0]
                seg_fn = seg_dir.split('/')[-1].split('-')[1] + \
                         seg_dir.split('/')[-1].split('-')[2].split('_')[0]
                if np.any(seg_arr) and img_fn == seg_fn:
                    print(seg_fn)
                    print(img_fn)
                    ## img preprocessing
                    arr = prepro_img(
                        img_arr=img_arr,
                        seg_arr=seg_arr,
                        z_max=z_max,
                        y_max=y_max,
                        x_max=x_max,
                        norm_type=norm_type,
                        input_type=input_type,
                        input_channel=input_channel,
                        save_img_type=save_img_type
                        )
                    # save img as numpy array or nii
                    save_img(
                        save_img_type=save_img_type, 
                        save_dir=save_dir, 
                        img=arr,
                        seg_fn=seg_fn
                        )
                else:
                    #print(seg_arr.shape)
                    print('problematic data:', seg_fn, img_fn)
                    continue
            continue
    
        ##PMH cohort
        elif cohort == 'PMH':
            print('PMH dataset:')
            for img_dir, seg_dir in zip(img_dirs, seg_dirs):
                img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
                seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
                img_arr = sitk.GetArrayFromImage(img)
                seg_arr = sitk.GetArrayFromImage(seg)
                img_fn = 'PMH' + img_dir.split('/')[-1].split('-')[1].split('_')[0][2:]
                seg_fn = 'PMH' + seg_dir.split('/')[-1].split('-')[1].split('_')[0][2:]
                ## load img and label data
                if np.any(seg_arr) and img_fn == seg_fn:
                    print(seg_fn)
                    print(img_fn)
                    ## image preprocessing
                    arr = prepro_img(
                        img_arr=img_arr,
                        seg_arr=seg_arr,
                        z_max=z_max,
                        y_max=y_max,
                        x_max=x_max,
                        norm_type=norm_type,
                        input_type=input_type,
                        input_channel=input_channel,
                        save_img_type=save_img_type
                        )
                    # save img as numpy array or nii
                    save_img(
                        save_img_type=save_img_type, 
                        save_dir=save_dir, 
                        img=arr,
                        seg_fn=seg_fn
                        )
                else:
                    #print(seg_arr.shape)
                    print('problematic data:', seg_fn, img_fn)
                    continue
            continue

        ## MDACC cohort
        elif cohort == 'MDACC':
            print('MDACC dataset:')
            for img_dir, seg_dir in zip(img_dirs, seg_dirs):
                img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
                seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
                img_arr = sitk.GetArrayFromImage(img)
                seg_arr = sitk.GetArrayFromImage(seg)                
                img_fn = 'MDACC' + img_dir.split('/')[-1].split('-')[2].split('_')[0][1:]
                seg_fn = 'MDACC' + seg_dir.split('/')[-1].split('-')[2].split('_')[0][1:]
                if np.any(seg_arr) and seg_fn == img_fn:
                    print(seg_fn)
                    print(img_fn)
                    arr = prepro_img(
                        img_arr=img_arr,
                        seg_arr=seg_arr,
                        z_max=z_max,
                        y_max=y_max,
                        x_max=x_max,
                        norm_type=norm_type,
                        input_type=input_type,
                        input_channel=input_channel,
                        save_img_type=save_img_type
                        )
                    # save img as numpy array or nii
                    save_img(
                        save_img_type=save_img_type, 
                        save_dir=save_dir, 
                        img=arr,
                        seg_fn=seg_fn
                        )
                else:
                    #print(seg_arr.shape)
                    print('problematic data:', seg_fn, img_fn)
                    continue
            pass
        
        print('successfully save numpy files!!')


