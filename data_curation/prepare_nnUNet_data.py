import numpy as np
import os
import glob
import pickle
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import shutil
from sklearn.model_selection import train_test_split



def prepare_nnUNet_data(proj_dir, data_dir, data_type):

    # make paths to store data
    img_tr_dir = data_dir + '/imagesTr'
    seg_tr_dir = data_dir + '/labelsTr'
    img_ts_dir = data_dir + '/imagesTs'
    seg_ts_dir = data_dir + '/labelsTs'
    img_ts2_dir = data_dir + '/imagesTs2'
    seg_ts2_dir = data_dir + '/labelsTs2'
    img_ts5_dir = data_dir + '/imagesTs5'
    seg_ts5_dir = data_dir + '/labelsTs5'
    img_noseg_dir = data_dir + '/images_no_seg'
    img_noseg2_dir = data_dir + '/images_no_seg2'
    if not os.path.exists(img_tr_dir):
        os.makedirs(img_tr_dir)
    if not os.path.exists(seg_tr_dir):
        os.makedirs(seg_tr_dir)
    if not os.path.exists(img_ts_dir):
        os.makedirs(img_ts_dir)
    if not os.path.exists(seg_ts_dir):
        os.makedirs(seg_ts_dir)
    if not os.path.exists(img_ts2_dir):
        os.makedirs(img_ts2_dir)
    if not os.path.exists(seg_ts2_dir):
        os.makedirs(seg_ts2_dir)
    if not os.path.exists(img_ts5_dir):
        os.makedirs(img_ts5_dir)
    if not os.path.exists(seg_ts5_dir):
        os.makedirs(seg_ts5_dir)
    if not os.path.exists(img_noseg_dir):
        os.makedirs(img_noseg_dir)
    if not os.path.exists(img_noseg2_dir):
        os.makedirs(img_noseg2_dir)

    # get df
    if data_type == 'train':
        img_crop_dir = proj_dir + '/data/TOT/crop_img_160'
        seg_crop_dir = proj_dir + '/data/TOT/crop_seg_p_n_160'
    elif data_type == 'test2':
        img_crop_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/DFCI/new_curation/crop_img'
        seg_crop_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/DFCI/new_curation/crop_seg_p_n'
    elif data_type == 'test5':
        img_crop_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH/crop_img'
        seg_crop_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH/crop_seg_pn'
    img_crop_dirs = [i for i in sorted(glob.glob(img_crop_dir + '/*nii.gz'))]
    seg_crop_dirs = [i for i in sorted(glob.glob(seg_crop_dir + '/*nii.gz'))]
    img_dirs = []
    seg_dirs = []
    img_ids = []
    cohorts = []
    _img_dirs = []
    _img_ids = []
    _cohorts = []
    for img_dir in img_crop_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        cohort = img_id.split('_')[0]
        for seg_dir in seg_crop_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                img_dirs.append(img_dir)
                seg_dirs.append(seg_dir)
                img_ids.append(img_id)
                cohorts.append(cohort)
    for img_dir in img_crop_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        cohort = img_id.split('_')[0]
        if img_id not in img_ids:
            _img_dirs.append(img_dir)
            _img_ids.append(img_id)
            _cohorts.append(cohort)
    df = pd.DataFrame({'cohort': cohorts, 'img_id': img_ids, 'img_dir': img_dirs, 'seg_dir': seg_dirs})
    df_noseg = pd.DataFrame({'cohort': _cohorts, 'img_id': _img_ids, 'img_dir': _img_dirs})
    print(df_noseg)
    
    # prepare data
    if data_type == 'train':
        print(df)
        # set DFCI data as external test
        df_ts2 = df[df['cohort']=='DFCI']
        df_tr = df[df['cohort']!='DFCI']
        # set 10% data as internal test
        df_tr, df_ts = train_test_split(df_tr, test_size=0.15)
        
        ## train dataset
        img_nn_ids = []
        seg_nn_ids = []
        for i, (img_dir, seg_dir) in enumerate(zip(df_tr['img_dir'], df_tr['seg_dir'])):
            img_nn_id = 'TR_' + str(f'{i:04}') + '_0000.nii.gz'
            seg_nn_id = 'TR_' + str(f'{i:04}') + '.nii.gz'
            print(i, img_nn_id)
            img_save_dir = img_tr_dir + '/' + img_nn_id
            seg_save_dir = seg_tr_dir + '/' + seg_nn_id
            img = sitk.ReadImage(img_dir)
            sitk.WriteImage(img, img_save_dir)
            seg = sitk.ReadImage(seg_dir)
            sitk.WriteImage(seg, seg_save_dir)
            img_nn_ids.append(img_nn_id)
            seg_nn_ids.append(seg_nn_id)
        df_tr['img_nn_id'], df_tr['seg_nn_id'] = [img_nn_ids, seg_nn_ids]
        df_tr.to_csv(data_dir + '/df_tr_pn.csv', index=False)
        
        ### test dataset 1
        img_nn_ids = []
        seg_nn_ids = []
        for i, (img_dir, seg_dir) in enumerate(zip(df_ts['img_dir'], df_ts['seg_dir'])):
            img_nn_id = 'TS_' + str(f'{i:04}') + '_0000.nii.gz'
            seg_nn_id = 'TS_' + str(f'{i:04}') + '.nii.gz'
            print(i, img_nn_id)
            img_save_dir = img_ts_dir + '/' + img_nn_id
            seg_save_dir = seg_ts_dir + '/' + seg_nn_id
            img = sitk.ReadImage(img_dir)
            sitk.WriteImage(img, img_save_dir)
            seg = sitk.ReadImage(seg_dir)
            sitk.WriteImage(seg, seg_save_dir)
            img_nn_ids.append(img_nn_id)
            seg_nn_ids.append(seg_nn_id)
        df_ts['img_nn_id'], df_ts['seg_nn_id'] = [img_nn_ids, seg_nn_ids]
        df_ts.to_csv(data_dir + '/df_ts_pn.csv', index=False)
        
        ### test dataset 2
        img_nn_ids = []
        seg_nn_ids = []
        for i, (img_dir, seg_dir) in enumerate(zip(df_ts2['img_dir'], df_ts2['seg_dir'])):
            img_nn_id = 'TS2_' + str(f'{i:04}') + '_0000.nii.gz'
            seg_nn_id = 'TS2_' + str(f'{i:04}') + '.nii.gz'
            print(i, img_nn_id)
            img_save_dir = img_ts2_dir + '/' + img_nn_id
            seg_save_dir = seg_ts2_dir + '/' + seg_nn_id
            img = sitk.ReadImage(img_dir)
            sitk.WriteImage(img, img_save_dir)
            seg = sitk.ReadImage(seg_dir)
            sitk.WriteImage(seg, seg_save_dir)
            img_nn_ids.append(img_nn_id)
            seg_nn_ids.append(seg_nn_id)
        df_ts2['img_nn_id'], df_ts2['seg_nn_id'] = [img_nn_ids, seg_nn_ids]
        df_ts2.to_csv(data_dir + '/df_ts2_pn.csv', index=False)

        # img dataset without seg
        img_nn_ids = []
        for i, img_dir in enumerate(df_noseg['img_dir']):
            img_nn_id = 'TCIA_' + str(f'{i:04}') + '_0000.nii.gz'
            print(i, img_nn_id)
            img_save_dir = img_noseg_dir + '/' + img_nn_id
            img = sitk.ReadImage(img_dir)
            sitk.WriteImage(img, img_save_dir)
            img_nn_ids.append(img_nn_id)
        df_noseg['img_nn_id'] = img_nn_ids
        df_noseg.to_csv(data_dir + '/df_no_seg_pn.csv', index=False)
    
    ### external test dataset: test2
    elif data_type == 'test5':
        img_nn_ids = []
        seg_nn_ids = []
        for i, (img_dir, seg_dir) in enumerate(zip(df['img_dir'], df['seg_dir'])):
            img_nn_id = 'TS5_' + str(f'{i:04}') + '_0000.nii.gz'
            seg_nn_id = 'TS5_' + str(f'{i:04}') + '.nii.gz'
            print(i, img_nn_id)
            img_save_dir = img_ts5_dir + '/' + img_nn_id
            seg_save_dir = seg_ts5_dir + '/' + seg_nn_id
            img = sitk.ReadImage(img_dir)
            sitk.WriteImage(img, img_save_dir)
            seg = sitk.ReadImage(seg_dir)
            sitk.WriteImage(seg, seg_save_dir)
            img_nn_ids.append(img_nn_id)
            seg_nn_ids.append(seg_nn_id)
        df['img_nn_id'], df['seg_nn_id'] = [img_nn_ids, seg_nn_ids]
        df.to_csv(data_dir + '/df_ts5_pn.csv', index=False)
        # img dataset without seg
        img_nn_ids = []
        for i, img_dir in enumerate(df_noseg['img_dir']):
            img_nn_id = 'BWH_' + str(f'{i:04}') + '_0000.nii.gz'
            print(i, img_nn_id)
            img_save_dir = img_noseg_dir + '/' + img_nn_id
            img = sitk.ReadImage(img_dir)
            sitk.WriteImage(img, img_save_dir)
            img_nn_ids.append(img_nn_id)
        df_noseg['img_nn_id'] = img_nn_ids
        df_noseg.to_csv(data_dir + '/df_no_seg_pn5.csv', index=False)


if __name__ == '__main__':

    task = 'Task502_tot_p_n'
    data_type = 'train'
    project = 'kannlab'
    if project == 'kannlab':
        proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck'
        data_dir = proj_dir + '/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/' + task
    if project == 'harvard':
        proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
        data_dir = proj_dir + '/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/' + task
    elif project == 'hecktor':
        proj_dir = '/mnt/aertslab/USERS/Zezhong/hecktor2022/DATA2'
        data_dir = proj_dir + '/nnUNet_raw_data_base/nnUNet_raw_data/' + task
    
    prepare_nnUNet_data(proj_dir, data_dir, data_type)









