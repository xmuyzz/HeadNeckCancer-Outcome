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
        img_crop_dir = proj_dir + '/data/BWH/crop_img'
        seg_crop_dir = proj_dir + '/data/BWH/crop_seg_p_n'
    elif data_type == 'test2':
        img_crop_dir = proj_dir + '/DFCI/new_curation/img_crop_172x172x76'
        seg_crop_dir = proj_dir + '/DFCI/new_curation/seg_crop_172x172x76'
    elif data_type == 'test5':
        img_crop_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH/crop_img'
        seg_crop_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH/crop_seg_pn'
    img_crop_dirs = [i for i in sorted(glob.glob(img_crop_dir + '/*nrrd'))]
    seg_crop_dirs = [i for i in sorted(glob.glob(seg_crop_dir + '/*nrrd'))]
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
        df_tr, df_ts = train_test_split(df, test_size=0.2)
        
        ### train dataset
        img_nn_ids = []
        seg_nn_ids = []
        for i, (img_dir, seg_dir) in enumerate(zip(df_tr['img_dir'], df_tr['seg_dir'])):
            img_nn_id = 'TR_' + str(f'{i:04}') + '_0000.nii.gz'
            seg_nn_id = 'TR_' + str(f'{i:04}') + '.nii.gz'
            print(i, img_nn_id)
            img_save_dir = img_tr_dir + '/' + img_nn_id
            seg_save_dir = seg_tr_dir + '/' + seg_nn_id
            ## process img data
            img = sitk.ReadImage(img_dir)
            img = sitk.GetArrayFromImage(img)
            img = img.transpose(1, 2, 0)
            img = img.reshape(*img.shape, 1)
            print(img.shape)
            img = nib.Nifti1Image(img, affine=np.eye(4))
            nib.save(img, img_save_dir)
            ## process seg data
            shutil.copyfile(seg_dir, seg_save_dir)
            img_nn_ids.append(img_nn_id)
            seg_nn_ids.append(seg_nn_id)
        df_tr['img_nn_id'], df_tr['seg_nn_id'] = [img_nn_ids, seg_nn_ids]
        df_tr.to_csv(os.path.join(data_dir, 'df_tr_pn.csv'), index=False)
        
        ### test dataset
        img_nn_ids = []
        seg_nn_ids = []
        for i, (img_dir, seg_dir) in enumerate(zip(df_ts['img_dir'], df_ts['seg_dir'])):
            img_nn_id = 'TS_' + str(f'{i:04}') + '_0000.nii.gz'
            seg_nn_id = 'TS_' + str(f'{i:04}') + '.nii.gz'
            print(i, img_nn_id)
            img_save_dir = img_ts_dir + '/' + img_nn_id
            seg_save_dir = seg_ts_dir + '/' + seg_nn_id
            ## process img file
            img = sitk.ReadImage(img_dir)
            img = sitk.GetArrayFromImage(img)
            img = img.transpose(1, 2, 0)
            img = img.reshape(*img.shape, 1)
            print(img.shape)
            img = nib.Nifti1Image(img, affine=np.eye(4))
            nib.save(img, img_save_dir)
            #sitk.WriteImage(img, img_save_dir)
            ## process seg file
            shutil.copyfile(seg_dir, seg_save_dir)
            img_nn_ids.append(img_nn_id)
            seg_nn_ids.append(seg_nn_id)
        df_ts['img_nn_id'], df_ts['seg_nn_id'] = [img_nn_ids, seg_nn_ids]
        df_ts.to_csv(os.path.join(data_dir, 'df_ts_pn.csv'), index=False)
        # img dataset without seg
        img_nn_ids = []
        for i, img_dir in enumerate(df_noseg['img_dir']):
            img_nn_id = 'TCIA_' + str(f'{i:04}') + '_0000.nii.gz'
            print(i, img_nn_id)
            img_save_dir = img_noseg_dir + '/' + img_nn_id
            img = sitk.ReadImage(img_dir)
            img = sitk.GetArrayFromImage(img)
            img = img.transpose(1, 2, 0)
            img = img.reshape(*img.shape, 1)
            img = nib.Nifti1Image(img, affine=np.eye(4))
            nib.save(img, img_save_dir)
            #sitk.WriteImage(img, img_save_dir)
            img_nn_ids.append(img_nn_id)
        df_noseg['img_nn_id'] = img_nn_ids
        df_noseg.to_csv(os.path.join(data_dir, 'df_no_seg_pn.csv'), index=False)
    
    ### external test dataset: test2
    elif data_type == 'test2':
        img_nn_ids = []
        seg_nn_ids = []
        for i, (img_dir, seg_dir) in enumerate(zip(df['img_dir'], df['seg_dir'])):
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
        df['img_nn_id'], df['seg_nn_id'] = [img_nn_ids, seg_nn_ids]
        df.to_csv(os.path.join(data_dir, 'df_ts2_pn.csv'), index=False)
        # img dataset without seg
        img_nn_ids = []
        for i, img_dir in enumerate(df_noseg['img_dir']):
            img_nn_id = 'DFCI_' + str(f'{i:04}') + '_0000.nii.gz'
            print(i, img_nn_id)
            img_save_dir = img_noseg_dir + '/' + img_nn_id
            img = sitk.ReadImage(img_dir)
            sitk.WriteImage(img, img_save_dir)
            img_nn_ids.append(img_nn_id)
        df_noseg['img_nn_id'] = img_nn_ids
        df_noseg.to_csv(os.path.join(data_dir, 'df_no_seg_pn2.csv'), index=False)

    ### external test dataset: test3
    elif data_type == 'test5':
        img_nn_ids = []
        seg_nn_ids = []
        print('start data preparation:')
        print(df)
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
        df.to_csv(data_dir + '/output/df_ts5_pn.csv', index=False)

if __name__ == '__main__':

    task = 'Task503_bwh_p_n'
    data_type = 'train'
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck'
    data_dir = proj_dir + '/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/' + task

    prepare_nnUNet_data(proj_dir, data_dir, data_type)









