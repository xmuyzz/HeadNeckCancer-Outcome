import numpy as np
import os
import glob
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from bbox import get_bbox_3D
from resize_3d import resize_3d
from respacing import respacing
from bbox import get_bbox_3D



def get_tumor_vol(proj_dir, tumor_type, img_type, img_size):
    """ save bbox_img to folders
    """

    nnUNet_dir = proj_dir + '/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task502_tot_p_n'
    save_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file'   

    for data_set in ['tr_radcure', 'tx_bwh_pr', 'tx_maastro_pr']: 
        if data_set == 'tr_gt':
            img_dir = nnUNet_dir + '/imagesTr'
            seg_dir = nnUNet_dir + '/labelsTr'
        if data_set == 'tr_pred':
            img_dir = nnUNet_dir + '/imagesTr'
            seg_dir = nnUNet_dir + '/predsTr'
        elif data_set == 'tr_radcure':
            img_dir = nnUNet_dir + '/imagesTs_radcure'
            seg_dir = nnUNet_dir + '/predsTs_radcure' 
            save_folder = 'radcure'     
        elif data_set == 'ts_gt':
            img_dir = nnUNet_dir + '/imagesTs'
            seg_dir = nnUNet_dir + '/labelsTs'
        elif data_set == 'ts_pr':
            img_dir = nnUNet_dir + '/imagesTs'
            seg_dir = nnUNet_dir + '/predsTs'
        elif data_set == 'tx_bwh_gt':
            img_dir = nnUNet_dir + '/imagesTx_bwh'
            seg_dir = nnUNet_dir + '/labelsTx_bwh'
        elif data_set == 'tx_bwh_pr':
            img_dir = nnUNet_dir + '/imagesTx_bwh'
            seg_dir = nnUNet_dir + '/predsTx_bwh'
            save_folder = 'bwh'
        elif data_set == 'tx_masstro_gt':
            img_dir = nnUNet_dir + '/imagesTx_maastro'
            seg_dir = nnUNet_dir + '/labelsTx_maastro'
            save_folder = 'maastro'
        elif data_set == 'tx_maastro_pr':
            img_dir = nnUNet_dir + '/imagesTx_maastro'
            seg_dir = nnUNet_dir + '/predsTx_maastro'
            save_folder = 'maastro'

        # run core function and save data
        print('working on dataset:', data_set)
        bad_ids = []
        IDs = []
        tumor_vols = []
        node_vols = []
        for i, seg_path in enumerate(glob.glob(seg_dir + '/*nii.gz')):
            #id = seg_path.split('/')[-1].split('.')[0]
            id = seg_path.split('/')[-1]
            print(i, id)
            try:
                seg = sitk.ReadImage(seg_path)
                arr = sitk.GetArrayFromImage(seg)
                tumor_vol = np.count_nonzero(arr == 1) * 3 # mm^3
                node_vol = np.count_nonzero(arr == 2) * 3 # mm^3
                print('tumor volume: %s' %tumor_vol, 'node volume: %s' %node_vol)
                IDs.append(id)
                tumor_vols.append(tumor_vol)
                node_vols.append(node_vol)
            except Exception as e:
                print(id, e)
                bad_ids.append(id)
        print('bad data:', bad_ids)
        # IDss.extend(IDs)
        # tumor_volss.extend(tumor_vols)
        # node_volss.extend(node_vols)
        # print(len(IDss), len(tumor_volss), len(node_volss))
        df = pd.DataFrame({'nn_id': IDs, 'tumor volume': tumor_vols, 'node volume': node_vols})
        #print(df)
        save_path = save_dir + '/' + save_folder + '/tumor_volume.csv'
        df.to_csv(save_path, index=False)


if __name__ == '__main__':

    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck'
    img_type = 'attn122'   #'attn_img' # bbox_img, mask_img
    img_size = 'full'
    tumor_type = 'pn'


    get_tumor_vol(proj_dir, tumor_type, img_type, img_size)
    