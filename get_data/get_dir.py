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
import SimpleITK as sitk
import pickle



def get_dir(data_dir, proj_dir, tumor_type, run_empty_seg=True):


    """
    save np arr for masked img for CT scans
    
    Args:
      tumor_type {str} -- tumor + node or tumor;
      data_dir {path} -- data directory;
      proj_dir {path} -- project directory;

    Returns:
        data paths for images and segmentations;

    Raise errors:
        None;

    """

    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    # image dirs
    CHUM_img_dir = os.path.join(data_dir, 'CHUM_files/image_reg')
    CHUS_img_dir = os.path.join(data_dir, 'CHUS_files/image_reg')
    PMH_img_dir = os.path.join(data_dir, 'PMH_files/image_reg')
    MDACC_img_dir = os.path.join(data_dir, 'MDACC_files/image_reg')
    # primary and node segmentation dirs
    CHUM_seg_pn_dir = os.path.join(data_dir, 'CHUM_files/label_reg')
    CHUS_seg_pn_dir = os.path.join(data_dir, 'CHUS_files/label_reg')
    PMH_seg_pn_dir = os.path.join(data_dir, 'PMH_files/label_reg')
    MDACC_seg_pn_dir = os.path.join(data_dir, 'MDACC_files/label_reg')
    # primary segmentation dirs
    CHUM_seg_p_dir = os.path.join(data_dir, 'CHUM_files/label_p_reg')
    CHUS_seg_p_dir = os.path.join(data_dir, 'CHUS_files/label_p_reg')
    PMH_seg_p_dir = os.path.join(data_dir, 'PMH_files/label_p_reg')
    MDACC_seg_p_dir = os.path.join(data_dir, 'MDACC_files/label_p_reg')
    # node segmentation dirs
    CHUM_seg_p_dir = os.path.join(data_dir, 'CHUM_files/label_n_reg')
    CHUS_seg_p_dir = os.path.join(data_dir, 'CHUS_files/label_n_reg')
    PMH_seg_p_dir = os.path.join(data_dir, 'PMH_files/label_n_reg')
    MDACC_seg_p_dir = os.path.join(data_dir, 'MDACC_files/label_n_reg')


    # get dirs for all img and seg
    #------------------------------
    dirs = [
        CHUM_img_dir, CHUS_img_dir, PMH_img_dir, MDACC_img_dir,
        CHUM_seg_pn_dir, CHUS_seg_pn_dir, PMH_seg_pn_dir, MDACC_seg_pn_dir,
        CHUM_seg_p_dir, CHUS_seg_p_dir, PMH_seg_p_dir, MDACC_seg_p_dir,
        CHUM_seg_p_dir, CHUS_seg_p_dir, PMH_seg_p_dir, MDACC_seg_p_dir
        ]
    img_dirss = []
    seg_pn_dirss = []
    seg_p_dirss = []
    seg_n_dirss = []
    list_img_dir = []
    list_pn_dir = []
    list_p_dir = []
    list_n_dir = []
    for dir in dirs:
        if dir.split('/')[-1] == 'image_reg':
            img_dirs = [path for path in sorted(glob.glob(dir + '/*nrrd'))]
            img_dirss.append(img_dirs)
            list_img_dir.extend(img_dirs)
        elif dir.split('/')[-1] == 'label_reg':
            seg_pn_dirs = [path for path in sorted(glob.glob(dir + '/*nrrd'))]
            seg_pn_dirss.append(seg_pn_dirs)
            list_pn_dir.extend(seg_pn_dirs)
        elif dir.split('/')[-1] == 'label_p_reg':
            seg_p_dirs = [path for path in sorted(glob.glob(dir + '/*nrrd'))]
            seg_p_dirss.append(seg_p_dirs)
            list_p_dir.extend(seg_p_dirs)
        elif dir.split('/')[-1] == 'label_n_reg':
            seg_n_dirs = [path for path in sorted(glob.glob(dir + '/*nrrd'))]
            seg_n_dirss.append(seg_n_dirs)
            list_n_dir.extend(seg_n_dirs)
    print('img:', len(list_img_dir))
    print('pn seg:', len(list_pn_dir))
    print('p seg:', len(list_p_dir))
    print('n seg:', len(list_n_dir))

    # get missing segs for pn, p and n
    #----------------------------------
    fnss = []
    for list_dir in [
        list_img_dir, 
        list_pn_dir, 
        list_p_dir, 
        list_n_dir]:
        fns = []
        for dir in list_dir:
            fn = dir.split('/')[-1].split('_')[0]
            fns.append(fn)
        fnss.append(fns)
    ## get missing names of pn, p and n segs
    missing_pn = list(set(fnss[0]) - set(fnss[1]))
    missing_p = list(set(fnss[0]) - set(fnss[2]))
    missing_p = list(set(fnss[0]) - set(fnss[3]))
    print('missing pn:', missing_pn)
    print('missing p:', missing_p)
    print('missing n:', missing_n)
    print('missing pn:', len(missing_pn))
    print('missing p:', len(missing_p))
    print('missing n:', len(missing_n))

    # get empty seg
    #---------------
    print('empty seg:')
    if run_empty_seg:
        count = 0
        fnss = []
        for list_seg_dir in [list_pn_dir, list_p_dir, list_n_dir]:
            fns = []
            for seg_dir in list_seg_dir:
                #count += 1
                print(count)
                seg_img = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
                seg_arr = sitk.GetArrayFromImage(seg_img)
                if np.any(seg_arr):
                    continue
                elif not np.any(seg_arr):
                    fn = seg_dir.split('/')[-1].split('_')[0]
                    print('empty seg:', fn)
                    fns.append(fn)
            fnss.append(fns)
        empty_pn = fnss[0]
        empty_p = fnss[1]
        empty_n = fnss[2]
    else:
        empty_pn = ['HN-CHUS-101', 'HN-CHUS-102', 'OPC-00320', 
                    'HNSCC-01-0005', 'HNSCC-01-0008', 'HNSCC-01-0010']
        empty_p = ['HN-CHUS-101', 'HN-CHUS-102', 'OPC-00320', 
                   'HNSCC-01-0005', 'HNSCC-01-0008', 'HNSCC-01-0010']
        empty_n = []
    print('empty pn:', empty_pn)
    print('empty p:', empty_p)
    print('empty n:', empty_n)
    print('empty pn:', len(empty_pn))
    print('empty p:', len(empty_p))
    print('empty n:', len(empty_n))

    # get img and seg lists with correct patients
    #-------------------------------------------
    ## exlcude patients with empty or missiong segs
    exclude_pn = missing_pn + empty_pn 
    exclude_p = missing_p + empty_p
    exlcude_n = missing_n + empty_n
    
    ## get img dirs with matching img and pn seg
    if tumor_type == 'primary_node':
        excludes = exclude_pn
        _dirss = []
    for dirss in [img_dirss, seg_pn_dirss, seg_p_dirss, seg_n_dirss]:
        for dirs in dirss:
            _dirs = []
            for dir in dirs:
                fn = dir.split('/')[-1].split('_')[0]
                if fn not in excludes:
                    #print(fn)
                    _dirs.append(dir)
            _dirss.append(_dirs)
        seg_pn_dirss_ = []
        for seg_pn_dirs in seg_pn_dirss:
            seg_pn_dirs_ = []
            for seg_pn_dir in seg_pn_dirs:
                fn = seg_pn_dir.split('/')[-1].split('_')[0]
                if fn not in excludes:
                    #print(fn)
                    seg_pn_dirs_.append(seg_pn_dir)
            seg_pn_dirss_.append(seg_pn_dirs_)

    img_dirss = img_dirss_
    seg_pn_dirss = seg_pn_dirss_
    print(len(img_dirss[1]))
    print(len(seg_pn_dirss[1]))

    ## get img dirs with matching img and pn seg
    if tumor_type == 'primary_node':
        excludes = exclude_pn
        img_dirss_ = []
        for img_dirs in img_dirss:
            img_dirs_ = []
            for img_dir in img_dirs:
                fn = img_dir.split('/')[-1].split('_')[0]
                if fn not in excludes:
                    #print(fn)
                    img_dirs_.append(img_dir)
            img_dirss_.append(img_dirs_)
        seg_pn_dirss_ = []
        for seg_pn_dirs in seg_pn_dirss:
            seg_pn_dirs_ = []
            for seg_pn_dir in seg_pn_dirs:
                fn = seg_pn_dir.split('/')[-1].split('_')[0]
                if fn not in excludes:
                    #print(fn)
                    seg_pn_dirs_.append(seg_pn_dir)
            seg_pn_dirss_.append(seg_pn_dirs_)

    img_dirss = img_dirss_
    seg_pn_dirss = seg_pn_dirss_
    print(len(img_dirss[1]))
    print(len(seg_pn_dirss[1]))

    # save lists of dirs to pickle
    for dirss, fn in zip([img_dirss, seg_pn_dirss], 
                         ['img_dirss.pkl', 'seg_pn_dirss.pkl']):
        fn = os.path.join(pro_data_dir, fn)
        with open(fn, 'wb') as f:
            pickle.dump(dirss, f)
    print('successfully save all data dir!')
    

    return img_dirss, seg_pn_dirss, exclude_pn, exclude_p
