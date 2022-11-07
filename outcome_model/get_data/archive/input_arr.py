import numpy as np
import os
import glob
import pickle
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from get_data.preprocess_img import preprocess_img
from get_data.get_dir import get_dir



def input_arr(data_dir, proj_dir, new_spacing, norm_type, tumor_type, input_img_type, 
              input_channel, run_max_bbox, save_img_type):

    """
    save np arr for masked img for CT scans
    args:
        tumor_type {'string'} - tumor + node or tumor
        data_dir {'path'} - tumor+node label dir CHUM cohort
        arr_dir {path} - tumor+node label dir CHUS cohort
    return:
        images with preprocessing;        
    """
    
    # data path
    pn_mask_img_dir = proj_dir + '/data/pn_mask_img'
    pn_bbox_img_dir = proj_dir + '/data/pn_bbox_img'
    p_mask_img_dir = proj_dir + '/data/p_mask_img'
    p_bbox_img_dir = proj_dir + '/data/p_bbox_img'
    n_mask_img_dir = proj_dir + '/data/n_mask_img'
    n_bbox_img_dir = proj_dir + '/data/n_bbox_img'
    pro_data_dir = proj_dir + '/pro_data'
    if not os.path.exists(pn_mask_img_dir): 
        os.makedirs(pn_mask_img_dir)
    if not os.path.exists(pn_bbox_img_dir): 
        os.makedirs(pn_bbox_img_dir)
    if not os.path.exists(p_mask_img_dir): 
        os.makedirs(p_mask_img_dir)
    if not os.path.exists(p_bbox_img_dir): 
        os.makedirs(p_bbox_img_dir)
    if not os.path.exists(n_mask_img_dir): 
        os.makedirs(n_mask_img_dir)
    if not os.path.exists(n_bbox_img_dir): 
        os.makedirs(n_bbox_img_dir)

    # load dirss list from pickle
    if tumor_type == 'pn':
        assert tumor_type in ['pn', 'p', 'n']
        print(tumor_type)
        # load dirss from pickle
        dirsss = []
        for fn in ['img_pn_dirss.pkl', 'seg_pn_dirss.pkl']:
            fn = pro_data_dir + '/' + fn
            with open(fn, 'rb') as f:
                dirss = pickle.load(f)
            dirsss.append(dirss)
        img_dirss = dirsss[0]
        seg_dirss = dirsss[1]
        # choose save dir for masked or raw img
        if input_img_type == 'mask_img':
            save_dir = pn_mask_img_dir
        elif input_img_type == 'bbox_img':
            save_dir = pn_bbox_img_dir
        # max bounding box
        max_bbox = (96, 96, 96)
    elif tumor_type == 'p':
        assert tumor_type in ['pn', 'p', 'n']
        print(tumor_type)
        dirsss = []
        for fn in ['img_p_dirss.pkl', 'seg_p_dirss.pkl']:
            fn = pro_data_dir + '/' + fn
            with open(fn, 'rb') as f:
                dirss = pickle.load(f)
            dirsss.append(dirss)
        img_dirss = dirsss[0]
        seg_dirss = dirsss[1]
        #print('dirss:', dirss)
        #print('img_dirss:', img_dirss)
        #print('seg_dirss:', seg_dirss)
        if input_img_type == 'mask_img':
            save_dir = p_mask_img_dir
        elif input_img_type == 'bbox_img':
            save_dir = p_bbox_img_dir
        # max bounding max
        max_bbox = (88, 88, 88)
    elif tumor_type == 'n':
        assert tumor_type in ['pn', 'p', 'n']
        print(tumor_type)
        dirsss = []
        for fn in ['img_n_dirss.pkl', 'seg_n_dirss.pkl']:
            fn = pro_data_dir +'/' + fn
            with open(fn, 'rb') as f:
                dirss = pickle.load(f)
            dirsss.append(dirss)
        img_dirss = dirsss[0]
        seg_dirss = dirsss[1]
        if input_img_type == 'mask_img':
            save_dir = n_mask_img_dir
        elif input_img_type == 'bbox_img':
            save_dir = n_bbox_img_dir
        # max bounding max
        max_bbox = (90, 90, 90)

    # load image and label to get numpy arrays
    cohorts = ['CHUM', 'CHUS', 'PMH', 'MDACC']
    for cohort, img_dirs, seg_dirs in zip(cohorts, img_dirss, seg_dirss):
        ## CHUM and CHUS cohort
        #print('img_dirs:', img_dirs)
        #print('seg_dirs:', seg_dirs)
        if cohort in ['CHUM', 'CHUS']:
            print('CHUM and CHUS dataset:')
            for img_dir, seg_dir in zip(img_dirs, seg_dirs):
                ## img andc   seg numbers are not equal
                #print(img_dir)
                #print(seg_dir)
                img_fn = img_dir.split('/')[-1].split('-')[1] + \
                         img_dir.split('/')[-1].split('-')[2].split('_')[0]
                seg_fn = seg_dir.split('/')[-1].split('-')[1] + \
                         seg_dir.split('/')[-1].split('-')[2].split('_')[0]
                if img_fn == seg_fn:
                    print(seg_fn)
                    ## img preprocessing
                    img = preprocess_img(
                        img_dir=img_dir,
                        seg_dir=seg_dir,
                        new_spacing=new_spacing,
                        norm_type=norm_type,
                        input_img_type=input_img_type,
                        input_channel=input_channel,
                        max_bbox = max_bbox)
                    img = nib.Nifti1Image(img, affine=np.eye(4))
                    nib.save(img, save_dir + '/' + seg_fn + '.nii.gz')
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
                    img = preprocess_img(
                        img_dir=img_dir,
                        seg_dir=seg_dir,
                        new_spacing=new_spacing,
                        norm_type=norm_type,
                        input_img_type=input_img_type,
                        input_channel=input_channel,
                        max_bbox = max_bbox)
                    img = nib.Nifti1Image(img, affine=np.eye(4))
                    nib.save(img, save_dir + '/' + seg_fn + '.nii.gz')
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
                    img = preprocess_img(
                        img_dir=img_dir,
                        seg_dir=seg_dir,
                        new_spacing=new_spacing,
                        norm_type=norm_type,
                        input_img_type=input_img_type,
                        input_channel=input_channel,
                        max_bbox = max_bbox)
                    img = nib.Nifti1Image(img, affine=np.eye(4))
                    nib.save(img, save_dir + '/' + seg_fn + '.nii.gz')
                else:
                    #print(seg_arr.shape)
                    print('problematic data:', seg_fn, img_fn)
                    continue
            pass
        
        print('successfully save numpy files!!')


