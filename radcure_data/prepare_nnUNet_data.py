import numpy as np
import os
import glob
import pickle
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import shutil


def prepare_nnUNet_data(data_dir, save_dir):

    #img_ts_dir = save_dir + '/imagesTs_radcure'
    img_ts_dir = save_dir + '/imagesTx_bwh'
    if not os.path.exists(img_ts_dir):
        os.makedirs(img_ts_dir)

    img_crop_dirs = [i for i in sorted(glob.glob(data_dir + '/*nii.gz'))]
    img_dirs = []
    img_ids = []
    cohorts = []
    for img_dir in img_crop_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        cohort = img_id.split('_')[0]
        img_dirs.append(img_dir)
        img_ids.append(img_id)
        cohorts.append(cohort)
        print(img_id)
    df = pd.DataFrame({'cohort': cohorts, 'img_id': img_ids, 'img_dir': img_dirs})
    print(df)
    
    ### test dataset: test
    img_nn_ids = []
    for i, img_dir in enumerate(df['img_dir']):
        img_nn_id = 'TS_' + str(f'{i:04}') + '_0000.nii.gz'
        print(i, img_nn_id)
        img_save_dir = img_ts_dir + '/' + img_nn_id
        img = sitk.ReadImage(img_dir)
        sitk.WriteImage(img, img_save_dir)
        img_nn_ids.append(img_nn_id)
    df['img_nn_id'] = img_nn_ids
    df.to_csv(os.path.join(save_dir, 'tx_bwh_pn.csv'), index=False)


if __name__ == '__main__':

    task = 'Task502_tot_p_n'
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck'
    #data_dir = proj_dir + '/data/MAASTRO/crop_img'
    data_dir = proj_dir + '/data/BWH_TOT/crop_img'
    save_dir = proj_dir + '/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/' + task

    prepare_nnUNet_data(data_dir, save_dir)









