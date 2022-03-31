import numpy as np
import os
import glob
import pickle
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from get_data.prepro_img import prepro_img
from get_data.max_bbox import max_bbox
from get_data.get_dir import get_dir
from get_data.bbox import bbox_3D



def input_arr(data_dir, proj_dir, new_spacing, norm_type, tumor_type, input_type, 
              input_channel, run_max_bbox, img_dirss, seg_pn_dirss, save_img_type):

    """
    save np arr for masked img for CT scans
    
    args:
        tumor_type {'string'} - tumor + node or tumor
        data_dir {'path'} - tumor+node label dir CHUM cohort
        arr_dir {path} - tumor+node label dir CHUS cohort

    return:
        images with preprocessing;        
    """
     
    pn_masked_img_dir = os.path.join(proj_dir, 'data/pn_masked_img')
    pn_raw_img_dir = os.path.join(proj_dir, 'data/pn_raw_img')
    p_masked_img_dir = os.path.join(proj_dir, 'data/PMH_files/p_masked_img')
    p_raw_img_dir = os.path.join(proj_dir, 'data/MDACC_files/p_raw_img')
    toyset_dir = os.path.join(proj_dir, 'data/toyset')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')

    if not os.path.exists(pn_raw_img_dir): os.mkdir(pn_raw_img_dir)

    """ load dirss list from pickle
    """
    fn = os.path.join(pro_data_dir, 'img_dirss.pkl')
    with open(fn, 'rb') as f:
        img_dirss = pickle.load(f)
    fn = os.path.join(pro_data_dir, 'seg_pn_dirss.pkl')
    with open(fn, 'rb') as f:
        seg_pn_dirss = pickle.load(f)

    ## get the max lenths of r, c, z of bbox
    #---------------------------------------
    if run_max_bbox:
        d_max, h_max, w_max = max_bbox(
            data_dir=data_dir,
            tumor_type=tumor_type
            )
    else:
        #z_max = 64
        #y_max = 132
        #x_max = 178
        d_max = 96
        h_max = 96
        w_max = 96


    """choose tumor type and input img type
    """
    if tumor_type == 'primary_node':
        print(tumor_type)
        seg_dirss = seg_pn_dirss
        if input_type == 'masked_img':
            print(input_type)
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
    
    """load image and label to get numpy arrays
    """
    cohorts = ['CHUM', 'CHUS', 'PMH', 'MDACC'] 
    for cohort, img_dirs, seg_dirs in zip(cohorts, img_dirss, seg_dirss):
        ## CHUM and CHUS cohort
        if cohort in ['CHUM', 'CHUS']:
            print('CHUM and CHUS dataset:')
            for img_dir, seg_dir in zip(img_dirs, seg_dirs):
                ## img and seg numbers are not equal
                img_fn = img_dir.split('/')[-1].split('-')[1] + \
                         img_dir.split('/')[-1].split('-')[2].split('_')[0]
                seg_fn = seg_dir.split('/')[-1].split('-')[1] + \
                         seg_dir.split('/')[-1].split('-')[2].split('_')[0]
                if img_fn == seg_fn:
                    print(seg_fn)
                    ## img preprocessing
                    img = prepro_img(
                        img_dir=img_dir,
                        seg_dir=seg_dir,
                        new_spacing=new_spacing,
                        norm_type=norm_type,
                        input_type=input_type,
                        input_channel=input_channel,
                        d_max=d_max,
                        h_max=h_max,
                        w_max=w_max,
                        )
                    if save_img_type == 'npy':
                        fn = str(seg_fn) + '.npy'
                        np.save(os.path.join(save_dir, fn), img, allow_pickle=True)
                    elif save_img_type == 'nii':
                        fn = str(seg_fn) + '.nii.gz'
                        img = nib.Nifti1Image(img, affine=np.eye(4))
                        nib.save(img, os.path.join(save_dir, fn))
                else:
                    #print(seg_arr.shape)
                    print('problematic data:', seg_fn, img_fn)
                    continue
            continue
    
        ##PMH cohort
        elif cohort == 'PMH':
            print('PMH dataset:')
            for img_dir, seg_dir in zip(img_dirs, seg_dirs):
                img_fn = 'PMH' + img_dir.split('/')[-1].split('-')[1].split('_')[0][2:]
                seg_fn = 'PMH' + seg_dir.split('/')[-1].split('-')[1].split('_')[0][2:]
                ## load img and label data
                if img_fn == seg_fn:
                    print(seg_fn)
                    ## image preprocessing
                    img = prepro_img(
                        img_dir=img_dir,
                        seg_dir=seg_dir,
                        new_spacing=new_spacing,
                        norm_type=norm_type,
                        input_type=input_type,
                        input_channel=input_channel,
                        d_max=d_max,
                        h_max=h_max,
                        w_max=w_max,
                        )
                    if save_img_type == 'npy':
                        fn = str(seg_fn) + '.npy'
                        np.save(os.path.join(save_dir, fn), img, allow_pickle=True)
                    elif save_img_type == 'nii':
                        fn = str(seg_fn) + '.nii.gz'
                        img = nib.Nifti1Image(img, affine=np.eye(4))
                        nib.save(img, os.path.join(save_dir, fn))
                else:
                    #print(seg_arr.shape)
                    print('problematic data:', seg_fn, img_fn)
                    continue
            continue

        ## MDACC cohort
        elif cohort == 'MDACC':
            print('MDACC dataset:')
            for img_dir, seg_dir in zip(img_dirs, seg_dirs):
                img_fn = 'MDACC' + img_dir.split('/')[-1].split('-')[2].split('_')[0][1:]
                seg_fn = 'MDACC' + seg_dir.split('/')[-1].split('-')[2].split('_')[0][1:]
                if seg_fn == img_fn:
                    print(seg_fn)
                    img = prepro_img(
                        img_dir=img_dir,
                        seg_dir=seg_dir,
                        new_spacing=new_spacing,
                        norm_type=norm_type,
                        input_type=input_type,
                        input_channel=input_channel,
                        d_max=d_max,
                        h_max=h_max,
                        w_max=w_max,
                        )
                    if save_img_type == 'npy':
                        fn = str(seg_fn) + '.npy'
                        np.save(os.path.join(save_dir, fn), img, allow_pickle=True)
                    elif save_img_type == 'nii':
                        fn = str(seg_fn) + '.nii.gz'
                        img = nib.Nifti1Image(img, affine=np.eye(4))
                        nib.save(img, os.path.join(save_dir, fn))                
                else:
                    #print(seg_arr.shape)
                    print('problematic data:', seg_fn, img_fn)
                    continue
            pass
        
        print('successfully save numpy files!!')


