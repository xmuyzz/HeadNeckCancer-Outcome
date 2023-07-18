import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold


def df_img_label(proj_dir, data_type, tumor_type, input_img_type, save_img_type):
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
    csv_dir = proj_dir + '/csv_file'
    tr_pn_dir = proj_dir + '/bbox_img/tr_pn'
    tr_p_dir = proj_dir + '/bbox_img/tr_p'
    tr_n_dir = proj_dir + '/bbox_img/tr_n'
    ts_pn_dir = proj_dir + '/bbox_img/ts_pn'
    ts_p_dir = proj_dir + '/bbox_img/ts_p'
    ts_n_dir = proj_dir + '/bbox_img/ts_n'

    if data_type == 'train':
        if tumor_type == 'pn':
            save_fn = 'tr_img_label_pn.csv'
            img_dirs = [i for i in sorted(glob.glob(img_pn_dir + '/*nii.gz'))]
        if tumor_type == 'p':
            save_fn = 'img_label_p.csv'
            img_dirs = [i for i in sorted(glob.glob(img_p_dir + '/*nii.gz'))]
        if tumor_type == 'n':
            save_fn = 'img_label_n.csv'
            img_dirs = [i for i in sorted(glob.glob(img_n_dir + '/*nii.gz'))]
    elif data_type == 'test':
        if tumor_type == 'pn':
            save_fn = 'img_label_pn.csv'
            img_dirs = [i for i in sorted(glob.glob(img_pn_dir + '/*nii.gz'))]
        if tumor_type == 'p':
            save_fn = 'img_label_p.csv'
            img_dirs = [i for i in sorted(glob.glob(img_p_dir + '/*nii.gz'))]
        if tumor_type == 'n':
            save_fn = 'img_label_n.csv'
            img_dirs = [i for i in sorted(glob.glob(img_n_dir + '/*nii.gz'))]

    # create df for data and pat_id to match labels
    #---------------------------------------------
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
    

if __name__ == '__main__':

    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data'
    df_img_label()


