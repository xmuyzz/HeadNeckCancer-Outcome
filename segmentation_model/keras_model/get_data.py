import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import glob
from sklearn.model_selection import train_test_split


def train_data(proj_dir, crop_shape):

    #img_crop_dir = proj_dir + '/HKTR_TCIA_DFCI/TOT/crop_img'
    #seg_crop_dir = proj_dir + '/HKTR_TCIA_DFCI/TOT/crop_seg'
    #img_crop_dirs = [i for i in sorted(glob.glob(img_crop_dir + '/*nii.gz'))]
    #seg_crop_dirs = [i for i in sorted(glob.glob(seg_crop_dir + '/*nii.gz'))]
    img_crop_dirs = [i for i in glob.glob(proj_dir + '/TCIA/img_crop_160/*nrrd')]
    seg_crop_dirs = [i for i in glob.glob(proj_dir + '/TCIA/seg_pn_crop_160/*nrrd')]
    img_dirs = []
    seg_dirs = []
    img_ids = []
    cohorts = []
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
    df = pd.DataFrame({'cohort': cohorts, 'img_id': img_ids, 'img_dir': img_dirs, 'seg_dir': seg_dirs})
    print(df)
    # set DFCI data as external test
    #df_ts2 = df[df['cohort']=='DFCI']
    #df_tr_val_ts = df[df['cohort']!='DFCI']
    # set 10% data for internal test and 10% for val
    #df_tr_val, df_ts = train_test_split(df_tr_val_ts, test_size=0.9, random_state=1234)
    df_tr_val, df_ts = train_test_split(df, test_size=0.8, random_state=1234)
    df_tr, df_val = train_test_split(df_tr_val, test_size=0.2, random_state=1234)
    
    # train dataset
    print('\n--- loading train data ---')
    imgs = []
    segs = []
    for i, (img_dir, seg_dir) in enumerate(zip(df_tr['img_dir'], df_tr['seg_dir'])):
        img_id = img_dir.split('/')[-1]
        seg_id = seg_dir.split('/')[-1]
        if img_id == seg_id:
            print(i, img_id)
            img = sitk.ReadImage(img_dir)
            seg = sitk.ReadImage(seg_dir)
            img_arr = sitk.GetArrayFromImage(img)
            seg_arr = sitk.GetArrayFromImage(seg)
            #print(img_arr.shape)
            #print(seg_arr.shape)
            # Sanity check
            #print(seg_arr.max())
            img_arr = crop_arr(img_arr, crop_shape)
            seg_arr = crop_arr(seg_arr, crop_shape)
            #print(img_arr.shape)
            assertions(img_arr, seg_arr, img_id)
            imgs.append(img_arr)
            segs.append(seg_arr)
    tr_data = {'img': np.array(imgs), 'seg': np.array(segs)}
    
    # val dataset
    print('\n--- loading val data ---')
    imgs = []
    segs = []
    for i, (img_dir, seg_dir) in enumerate(zip(df_val['img_dir'], df_val['seg_dir'])):
        img_id = img_dir.split('/')[-1]
        seg_id = seg_dir.split('/')[-1]
        if img_id == seg_id:
            print(i, img_id)
            img = sitk.ReadImage(img_dir)
            seg = sitk.ReadImage(seg_dir)
            img_arr = sitk.GetArrayFromImage(img)
            seg_arr = sitk.GetArrayFromImage(seg)
            img_arr = crop_arr(img_arr, crop_shape)
            seg_arr = crop_arr(seg_arr, crop_shape)
            img_arr = img_arr.reshape(1, *img_arr.shape)
            seg_arr = seg_arr.reshape(1, *seg_arr.shape)
            # binary label to be 0 and 1
            #seg_arr[seg_arr > 1] = 1
            #print(img_arr.shape)
            #print(seg_arr.shape)
            # Sanity check
            assertions(img_arr, seg_arr, img_id)
            imgs.append(img_arr)
            segs.append(seg_arr)
    val_data = {'img': np.array(imgs), 'seg': np.array(segs)}
    
    return tr_data, val_data


def test_data(proj_dir, crop_shape):

    img_crop_dir = proj_dir + '/HKTR_TCIA_DFCI/TOT/crop_img'
    seg_crop_dir = proj_dir + '/HKTR_TCIA_DFCI/TOT/crop_seg'
    img_crop_dirs = [i for i in sorted(glob.glob(img_crop_dir + '/*nii.gz'))]
    seg_crop_dirs = [i for i in sorted(glob.glob(seg_crop_dir + '/*nii.gz'))]
    img_dirs = []
    seg_dirs = []
    img_ids = []
    cohorts = []
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
    df = pd.DataFrame({'cohort': cohorts, 'img_id': img_ids, 'img_dir': img_dirs, 'seg_dir': seg_dirs})
    print(df)
    # set DFCI data as external test
    df_ts2 = df[df['cohort']=='DFCI']
    df_tr_val_ts = df[df['cohort']!='DFCI']
    # set 10% data for internal test and 10% for val
    df_tr_val, df_ts = train_test_split(df_tr_val_ts, test_size=0.1, random_state=1234)
    df_tr, df_val = train_test_split(df_tr_val, test_size=0.1, random_state=1234)

    print('--- loading test dataset ---')
    ts1_data = []
    for i, (img_dir, seg_dir) in enumerate(zip(df_ts['img_dir'], df_ts['seg_dir'])):
        img_id = img_dir.split('/')[-1].split('.')[0]
        img = sitk.ReadImage(img_dir)
        img_arr = sitk.GetArrayFromImage(img)
        seg = sitk.ReadImage(seg_dir)
        seg_arr = sitk.GetArrayFromImage(seg)
        img_arr = crop_arr(img_arr, crop_shape)
        seg_arr = crop_arr(seg_arr, crop_shape)
        seg_arr[seg_arr > 1] = 1
        # Sanity check
        assertions(img_arr, seg_arr, img_id)
        ts1_data.append({
            'patient_id': img_id,
            'img_sitk_obj': img,
            'img_arr': img_arr,
            'seg_sitk_obj': seg})

    print('--- loading test dataset 2 ---')
    ts2_data = []
    for i, (img_dir, seg_dir) in enumerate(zip(df_ts2['img_dir'], df_ts2['seg_dir'])):
        img_id = img_dir.split('/')[-1].split('.')[0]
        img = sitk.ReadImage(img_dir)
        img_arr = sitk.GetArrayFromImage(img)
        seg = sitk.ReadImage(seg_dir)
        seg_arr = sitk.GetArrayFromImage(seg)
        img_arr = crop_arr(img_arr, crop_shape)
        seg_arr = crop_arr(seg_arr, crop_shape)
        seg_arr[seg_arr > 1] = 1
        # Sanity check
        assertions(img_arr, seg_arr, img_id)
        ts2_data.append({
            'patient_id': img_id,
            'img_sitk_obj': img,
            'img_arr': img_arr,
            'seg_sitk_obj': seg})

    return ts1_data, ts2_data


def assertions(arr_image, arr_label, patient_id):
    assert arr_image.shape == arr_label.shape, 'image and label do not have the same shape.'
    assert arr_label.min() == 0, 'label min is not 0 @ {}'.format(patient_id)
    #assert arr_label.max() == 3, 'label max is not 3 @ {}'.format(patient_id)
    #assert len(np.unique(arr_label))==3, 'lenght of label unique vals is not 3 @ {}_{}'.format(dataset, patient_id)


def crop_arr(arr, crop_shape):
    start_z = arr.shape[0]//2 - crop_shape[0]//2
    start_y = arr.shape[1]//2 - crop_shape[1]//2
    start_x = arr.shape[2]//2 - crop_shape[2]//2
    #
    arr = arr[start_z:start_z + crop_shape[0],
              start_y:start_y + crop_shape[1],
              start_x:start_x + crop_shape[2]]
    return arr



