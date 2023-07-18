import numpy as np
import os
import glob
import pickle
import pandas as pd
import nibabel as nib
import SimpleITK as sitk



def get_data(proj_dir, pro_data_dir, tumor_type, save_nii):
    
    """
    save np arr for masked img for CT scans 
    args:
        tumor_type {'string'} - tumor + node or tumor
        data_dir {'path'} - tumor+node label dir CHUM cohort
        arr_dir {path} - tumor+node label dir CHUS cohort
    return:
        images with preprocessing;        
    """

    # priamry tumor + node
    pn_img_tr_dir = os.path.join(proj_dir, 'Task501_PN/imagesTr')
    pn_seg_tr_dir = os.path.join(proj_dir, 'Task501_PN/labelsTr')
    pn_img_ts_dir = os.path.join(proj_dir, 'Task501_PN/imagesTs')
    pn_seg_ts_dir = os.path.join(proj_dir, 'Task501_PN/labelsTs')
    if not os.path.exists(pn_img_tr_dir):
        os.makedirs(pn_img_tr_dir)
    if not os.path.exists(pn_seg_tr_dir):
        os.makedirs(pn_seg_tr_dir)
    if not os.path.exists(pn_img_ts_dir):
        os.makedirs(pn_img_ts_dir)
    if not os.path.exists(pn_seg_ts_dir):
        os.makedirs(pn_seg_ts_dir)
    # primary tumor only
    p_img_tr_dir = os.path.join(proj_dir, 'Task502_P/imagesTr')
    p_seg_tr_dir = os.path.join(proj_dir, 'Task502_P/labelsTr')
    p_img_ts_dir = os.path.join(proj_dir, 'Task502_P/imagesTs')
    p_seg_ts_dir = os.path.join(proj_dir, 'Task502_P/labelsTs')
    if not os.path.exists(p_img_tr_dir):
        os.makedirs(p_img_tr_dir)
    if not os.path.exists(p_seg_tr_dir):
        os.makedirs(p_seg_tr_dir) 
    if not os.path.exists(p_img_ts_dir):
        os.makedirs(p_img_ts_dir) 
    if not os.path.exists(p_seg_ts_dir):
        os.makedirs(p_seg_ts_dir)
    # node only
    n_img_tr_dir = os.path.join(proj_dir, 'Task503_N/imagesTr')
    n_seg_tr_dir = os.path.join(proj_dir, 'Task503_N/labelsTr')
    n_img_ts_dir = os.path.join(proj_dir, 'Task503_N/imagesTs')
    n_seg_ts_dir = os.path.join(proj_dir, 'Task503_N/labelsTs')
    if not os.path.exists(n_img_tr_dir):
        os.makedirs(n_img_tr_dir)
    if not os.path.exists(n_seg_tr_dir):
        os.makedirs(n_seg_tr_dir) 
    if not os.path.exists(n_img_ts_dir):
        os.makedirs(n_img_ts_dir) 
    if not os.path.exists(n_seg_ts_dir):
        os.makedirs(n_seg_ts_dir)

    # load dirss list from pickle
    if tumor_type == 'pn':
        assert tumor_type in ['pn', 'p', 'n'], print('wrong tumor type!')
        print(tumor_type)
        dirsss = []
        for fn in ['img_pn_dirss.pkl', 'seg_pn_dirss.pkl']:
            fn = os.path.join(pro_data_dir, fn)
            with open(fn, 'rb') as f:
                dirss = pickle.load(f)
            dirsss.append(dirss)
        img_dirss = dirsss[0]
        seg_dirss = dirsss[1]
    elif tumor_type == 'p':
        assert tumor_type in ['pn', 'p', 'n'], print('wrong tumor type!')
        print(tumor_type)
        dirsss = []
        for fn in ['img_p_dirss.pkl', 'seg_p_dirss.pkl']:
            fn = os.path.join(pro_data_dir, fn)
            with open(fn, 'rb') as f:
                dirss = pickle.load(f)
            dirsss.append(dirss)
        img_dirss = dirsss[0]
        seg_dirss = dirsss[1]
        #print('dirss:', dirss)
        #print('img_dirss:', img_dirss)
        #print('seg_dirss:', seg_dirss)
    elif tumor_type == 'n':
        assert tumor_type in ['pn', 'p', 'n'], print('wrong tumor type!')
        print(tumor_type)
        dirsss = []
        for fn in ['img_n_dirss.pkl', 'seg_n_dirss.pkl']:
            fn = os.path.join(pro_data_dir, fn)
            with open(fn, 'rb') as f:
                dirss = pickle.load(f)
            dirsss.append(dirss)
        img_dirss = dirsss[0]
        seg_dirss = dirsss[1]

    # load image and label to get numpy arrays
    IDs = []
    img_fns = []
    seg_fns = []
    cohorts = ['CHUM', 'CHUS', 'PMH', 'MDACC']
    for cohort, img_dirs, seg_dirs in zip(cohorts, img_dirss, seg_dirss):
        ## CHUM and CHUS cohort
        #print('img_dirs:', img_dirs)
        #print('seg_dirs:', seg_dirs)
        if cohort in ['CHUM', 'CHUS']:
            print('CHUM and CHUS dataset:')
            if tumor_type == 'pn':
                img_save_dir = pn_img_ts_dir
                seg_save_dir = pn_seg_ts_dir
            elif tumor_type == 'p':
                img_save_dir = p_img_ts_dir
                seg_save_dir = p_seg_ts_dir
            if tumor_type == 'n':
                img_save_dir = n_img_ts_dir
                seg_save_dir = n_seg_ts_dir
            count = 0
            for img_dir, seg_dir in zip(img_dirs, seg_dirs):
                ## img andc seg numbers are not equal
                #print(img_dir)
                #print(seg_dir)
                img_id = img_dir.split('/')[-1].split('-')[1] + \
                         img_dir.split('/')[-1].split('-')[2].split('_')[0]
                seg_id = seg_dir.split('/')[-1].split('-')[1] + \
                         seg_dir.split('/')[-1].split('-')[2].split('_')[0]
                if img_id == seg_id:
                    count += 1
                    print(count, seg_id)
                    img_fn = 'OPC_' + str(f'{count:03}') + '_0000.nii.gz'
                    seg_fn = 'OPC_' + str(f'{count:03}') + '.nii.gz'
                    IDs.append(img_id)
                    img_fns.append(img_fn)
                    seg_fns.append(seg_fn)
                    if save_nii:
                        try:
                            img_nrrd = sitk.ReadImage(img_dir)
                            img = sitk.GetArrayFromImage(img_nrrd)
                            seg_nrrd = sitk.ReadImage(seg_dir)
                            seg = sitk.GetArrayFromImage(seg_nrrd)
                            img_fn = 'OPC_' + str(f'{count:03}') + '_0000.nii.gz'
                            seg_fn = 'OPC_' + str(f'{count:03}') + '.nii.gz'
                            print(img_fn)
                            #print(seg_fn)
                            img = nib.Nifti1Image(img, affine=np.eye(4))
                            seg = nib.Nifti1Image(seg, affine=np.eye(4))
                            nib.save(img, os.path.join(img_save_dir, img_fn))
                            nib.save(seg, os.path.join(seg_save_dir, seg_fn))
                        except Exception as e:
                            print(e, img_id)
                else:
                    #print(seg_arr.shape)
                    print('mismatched data:', seg_fn, img_fn)
                    continue
            continue
    
        ##PMH cohort
        elif cohort == 'PMH':
            print('PMH dataset:')
            if tumor_type == 'pn':
                img_save_dir = pn_img_tr_dir
                seg_save_dir = pn_seg_tr_dir
            elif tumor_type == 'p':
                img_save_dir = p_img_tr_dir
                seg_save_dir = p_seg_tr_dir
            if tumor_type == 'n':
                img_save_dir = n_img_tr_dir
                seg_save_dir = n_seg_tr_dir
            count = count
            for img_dir, seg_dir in zip(img_dirs, seg_dirs):
                ## img andc seg numbers are not equal
                #print(img_dir)
                #print(seg_dir)
                img_id = 'PMH' + img_dir.split('/')[-1].split('-')[1].split('_')[0][2:]
                seg_id = 'PMH' + seg_dir.split('/')[-1].split('-')[1].split('_')[0][2:]
                if img_id == seg_id:
                    count += 1
                    print(count, seg_id)
                    IDs.append(img_id)
                    img_fns.append(img_fn)
                    seg_fns.append(seg_fn)
                    if save_nii:
                        try:
                            img_nrrd = sitk.ReadImage(img_dir)
                            img = sitk.GetArrayFromImage(img_nrrd)
                            seg_nrrd = sitk.ReadImage(seg_dir)
                            seg = sitk.GetArrayFromImage(seg_nrrd)
                            img_fn = 'OPC_' + str(f'{count:03}') + '_0000.nii.gz'
                            seg_fn = 'OPC_' + str(f'{count:03}') + '.nii.gz'
                            print(img_fn)
                            #print(seg_fn)
                            img = nib.Nifti1Image(img, affine=np.eye(4))
                            seg = nib.Nifti1Image(seg, affine=np.eye(4))
                            nib.save(img, os.path.join(img_save_dir, img_fn))
                            nib.save(seg, os.path.join(seg_save_dir, seg_fn))
                        except Exception as e:
                            print(e, img_id)
                else:
                    #print(seg_arr.shape)
                    print('mismatched data:', seg_fn, img_fn)
                    continue
            continue
        
        ## MDACC cohort
        elif cohort == 'MDACC':
            print('MDACC dataset:')
            if tumor_type == 'pn':
                img_save_dir = pn_img_tr_dir
                seg_save_dir = pn_seg_tr_dir
            elif tumor_type == 'p':
                img_save_dir = p_img_tr_dir
                seg_save_dir = p_seg_tr_dir
            if tumor_type == 'n':
                img_save_dir = n_img_tr_dir
                seg_save_dir = n_seg_tr_dir
            #count = 0
            for img_dir, seg_dir in zip(img_dirs, seg_dirs):
                ## img andc seg numbers are not equal
                #print(img_dir)
                #print(seg_dir)
                img_id = 'MDACC' + img_dir.split('/')[-1].split('-')[2].split('_')[0][1:]
                seg_id = 'MDACC' + seg_dir.split('/')[-1].split('-')[2].split('_')[0][1:]
                if img_id == seg_id:
                    count += 1
                    print(count, seg_id)
                    img_fn = 'OPC_' + str(f'{count:03}') + '_0000.nii.gz'
                    seg_fn = 'OPC_' + str(f'{count:03}') + '.nii.gz'
                    print(img_fn)
                    #print(seg_fn)
                    IDs.append(img_id)
                    img_fns.append(img_fn)
                    seg_fns.append(seg_fn)
                    if save_nii:
                        try:
                            img_nrrd = sitk.ReadImage(img_dir)
                            img = sitk.GetArrayFromImage(img_nrrd)
                            seg_nrrd = sitk.ReadImage(seg_dir)
                            seg = sitk.GetArrayFromImage(seg_nrrd)
                            img_fn = 'OPC_' + str(f'{count:03}') + '_0000.nii.gz'
                            seg_fn = 'OPC_' + str(f'{count:03}') + '.nii.gz'
                            print(img_fn)
                            #print(seg_fn)
                            img = nib.Nifti1Image(img, affine=np.eye(4))
                            seg = nib.Nifti1Image(seg, affine=np.eye(4))
                            nib.save(img, os.path.join(img_save_dir, img_fn))
                            nib.save(seg, os.path.join(seg_save_dir, seg_fn))
                        except Exception as e:
                            print(e, img_id)
                else:
                    #print(seg_arr.shape)
                    print('mismatched data:', seg_fn, img_fn)
                    continue
            pass        
        df = pd.DataFrame({'ID': IDs, 'img_fn': img_fns, 'seg_fn': seg_fns})
        df.to_csv(os.path.join(proj_dir, 'nnUNet_ID.csv'), index=False) 
        print('successfully save numpy files!!')


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data'
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir, exist_ok=True)
    pro_data_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/pro_data'
    for tumor_type in ['pn', 'p', 'n']:
        get_data(
            proj_dir=proj_dir, 
            pro_data_dir=pro_data_dir, 
            tumor_type=tumor_type,
            save_nii=True)
    print('complete')





