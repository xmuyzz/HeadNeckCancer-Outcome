import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold


def df_img_label(proj_dir, tumor_type, input_img_type, save_img_type):

    """
    create df for data and pat_id to match labels 
    Args:
        proj_dir {path} -- project dir;
        out_dir {path} -- output dir;
        save_img_type {str} -- image type: nii or npy;
    Returns:
        Dataframe with image dirs and labels;
    Raise errors:
        None
    """
    img_pn_dir = proj_dir + '/data/TOT/bbox_img_pn'
    img_p_dir = proj_dir + '/data/TOT/bbox_img_p'
    img_n_dir = proj_dir + '/data/TOT/bbox_img_n'
    pro_data_dir = proj_dir + '/pro_data'

    # create df for data and pat_id to match labels
    if tumor_type == 'pn':
        save_fn = 'img_label_pn.csv'
        img_dirs = [i for i in sorted(glob.glob(img_pn_dir + '/*nii.gz'))]
    if tumor_type == 'p':
        save_fn = 'img_label_p.csv'
        img_dirs = [i for i in sorted(glob.glob(img_p_dir + '/*nii.gz'))]
    if tumor_type == 'n':
        save_fn = 'img_label_n.csv'
        img_dirs = [i for i in sorted(glob.glob(img_n_dir + '/*nii.gz'))]

    fns = []
    for img_dir in img_dirs:
        ID = img_dir.split('/')[-1].split('.')[0]
        fns.append(ID)
    print('pat_id:', len(fns))
    print('img_dir:', len(img_dirs))
    df_img = pd.DataFrame({'patid': fns, 'img_dir': img_dirs})
    print('total img number:', df_img.shape[0])
    print(df_img[0:10])

    # create matching label df
    df = pd.read_csv(pro_data_dir + '/label.csv')
    print('total label number:', df.shape)
    pat_ids = []
    for pat_id in df['pat_id']:
        if pat_id not in fns:
            pat_ids.append(pat_id)
    df_label = df[~df['pat_id'].isin(pat_ids)]
    print('total label number:', df_label.shape)
    print(df_label[0:10])

    # merge df_img and df_label using matching patient ID
    df = pd.merge(df_label, df_img, on='pat_id')
    print('total df size:', df.shape)
    print(df[0:20])
    df.to_csv(pro_data_dir + '/' + save_fn, index=False)
    print('complete img and lable df have been saved!!!')
    





