import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold, train_test_split


def df_img_label(proj_dir, surv_type, data_set, img_type, tumor_type):
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
    print('surv_type:', surv_type)
    print('img type:', img_type)
    print('tumor type:', tumor_type)
    print('data set:', data_set)
    
    csv_dir = proj_dir + '/csv_file'
    data_dir = proj_dir + '/' + img_type
    img_dir = data_dir + '/' + data_set + '_' + tumor_type 
    img_dirs = [i for i in sorted(glob.glob(img_dir + '/*nii.gz'))]

    if data_set == 'tr':
        label_file = 'tr_label.csv'
    elif data_set in ['ts_gt', 'ts_pr']:
        label_file = 'ts_label.csv'
    elif data_set in ['tx1_gt', 'tx1_pr']:
        label_file = 'tx1_label.csv'
    elif data_set in ['tx2_gt', 'tx2_pr']:
        label_file = 'tx2_label.csv'

    fns = []
    for img_dir in img_dirs:
        fn = img_dir.split('/')[-1]
        fns.append(fn)
    df_img = pd.DataFrame({'seg_nn_id': fns, 'img_dir': img_dirs})
    print('total img number:', df_img.shape[0])
    print(df_img[0:10])
    df_label = pd.read_csv(csv_dir + '/' + label_file)
    df = df_label.merge(df_img, how='left', on='seg_nn_id')
    # exclude img_dir = none
    df = df[df['img_dir'].notna()]
    print('total df size:', df.shape)
    print(df[0:20])
    
    # drop patients without clinical data: rfs, os, lr, dr
    df = df.dropna(subset=[surv_type + '_event', surv_type + '_time'])

    if data_set == 'tr':
        # stratify tr and val based on rfs
        df_tr, df_va = train_test_split(df, test_size=0.2, stratify=df[surv_type + '_event'])
        print('train data shape:', df_tr.shape)
        print('val data shape:', df_va.shape)
        tr_fn = 'tr_img_label_' + tumor_type + '.csv'
        va_fn = 'va_img_label_' + tumor_type + '.csv'
        df_tr.to_csv(data_dir + '/' + tr_fn, index=False)
        df_va.to_csv(data_dir + '/' + va_fn, index=False)
        print('train and val dfs have been saved!!!')
    else:
        print('test data shape:', df.shape)
        fn = data_set + '_img_label_' + tumor_type + '.csv'
        df.to_csv(data_dir + '/' + fn, index=False)
        print('test dfs have been saved!!!')
    

if __name__ == '__main__':
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data'
    task = 'rfs'
    #data_set = 'ts_gt'
    tumor_type = 'pn'
    img_type = '2channel'
    for data_set in ['tr', 'ts_gt', 'ts_pr']:
        df_img_label(proj_dir, task, data_set, img_type, tumor_type)
