import numpy as np
import os
import pandas as pd
import shutil
import glob
import SimpleITK as sitk



def transfer_HKTR_data(proj_dir):
    
    data_dir = proj_dir + '/hecktor2022/DATA2/hecktor2022_data'
    img_dirs1 = [i for i in sorted(glob.glob(data_dir + '/imagesTr/' + '*CT.nii.gz'))]
    img_dirs2 = [i for i in sorted(glob.glob(data_dir + '/imagesHoldOut/' + '*CT.nii.gz'))]
    img_dirs = img_dirs1 + img_dirs2
    seg_dirs1 = [i for i in sorted(glob.glob(data_dir + '/labelsTr/' + '*nii.gz'))]
    seg_dirs2 = [i for i in sorted(glob.glob(data_dir + '/labelsHoldOut/' + '*nii.gz'))]
    seg_dirs = seg_dirs1 + seg_dirs2

    img_save_dir = proj_dir + '/HN_OUTCOME/HKTR_TCIA_DFCI/HKTR/raw_img'
    seg_save_dir = proj_dir + '/HN_OUTCOME/HKTR_TCIA_DFCI/HKTR/raw_seg'
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(seg_save_dir):
        os.makedirs(seg_save_dir)
    
    print('\n--- copy img files ---')
    for i, img_dir in enumerate(img_dirs):
        fn = img_dir.split('/')[-1].split('__')[0]
        fn = fn.replace('-', '_')
        fn = fn + '.nii.gz'
        print(i, fn)
        save_path = img_save_dir + '/' + fn
        shutil.copyfile(img_dir, save_path)
    
    print('\n--- copy seg files ---')
    for i, seg_dir in enumerate(seg_dirs):
        fn = seg_dir.split('/')[-1]
        fn = fn.replace('-', '_')
        print(i, fn)
        save_path = seg_save_dir + '/' + fn
        shutil.copyfile(seg_dir, save_path)


def transfer_TCIA_data(proj_dir):

    CHUS_dir = raw_data_dir + '/CHUS_files/interpolated'
    CHUM_dir = raw_data_dir + '/CHUM_files/interpolated'
    MDACC_dir = raw_data_dir + '/MDACC_files/interpolated'
    PMH_dir = raw_data_dir + '/PMH_files/interpolated'
    save_img_dir = proj_dir + '/HN_OUTCOME/HKTR_TCIA_DFCI/TCIA/itp_img'
    save_seg_pn_dir = proj_dir + '/HN_OUTCOME/HKTR_TCIA_DFCI/TCIA/itp_seg_pn'
    save_seg_p_dir = proj_dir + '/HN_OUTCOME/HKTR_TCIA_DFCI/TCIA/itp_seg_p'
    save_seg_n_dir = proj_dir + '/HN_OUTCOME/HKTR_TCIA_DFCI/TCIA/itp_seg_n'
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    if not os.path.exists(save_seg_pn_dir):
        os.makedirs(save_seg_pn_dir)
    if not os.path.exists(save_seg_p_dir):
        os.makedirs(save_seg_p_dir)
    if not os.path.exists(save_seg_n_dir):
        os.makedirs(save_seg_n_dir)

    count = 0
    #data_dirs = [CHUM_dir, CHUS_dir, MDACC_dir, PMH_dir]
    #cohorts = ['CHUM', 'CHUS', 'MDA', 'PMH']
    data_dirs = [MDACC_dir, PMH_dir]
    cohorts = ['MDA', 'PMH']
    for data_dir, cohort in zip(data_dirs, cohorts):
        for data_path in sorted(glob.glob(data_dir + '/*nrrd')):
            if cohort in ['CHUM', 'CHUS']:
                ID = data_path.split('/')[-1].split('_')[1].split('-')[2]
            elif cohort == 'MDA':
                ID = data_path.split('/')[-1].split('_')[1].split('-')[2][1:]
            elif cohort == 'PMH':
                ID = data_path.split('/')[-1].split('_')[1].split('-')[1][2:]
            fn = cohort + '_' + ID + '.nii.gz'
            data_type = data_path.split('/')[-1].split('_')[2]
            if data_type == 'ct':
                count += 1
                print(count, fn)
                #save_fn = save_img_dir + '/' + fn
                #print(path)
                #print(dst_dir)
                #data = sitk.ReadImage(data_path)
                #sitk.WriteImage(data, save_fn)
            elif data_type == 'label':
                count += 1
                label = data_path.split('/')[-1].split('_')[3]
                if label == 'interpolated':
                    save_fn = save_seg_pn_dir + '/' + fn
                elif label == 'n':
                    save_fn = save_seg_n_dir + '/' + fn
                elif label == 'p':
                    save_fn = save_seg_p_dir + '/' + fn
                print(count, fn)
                #print(dst_dir) 
                data = sitk.ReadImage(data_path)
                sitk.WriteImage(data, save_fn)


def transfer_DFCI_data(proj_dir):

    data_dir = proj_dir + '/HN_OUTCOME/DFCI/new_curation'
    img_dirs = [i for i in sorted(glob.glob(data_dir + '/raw_img/' + '*nrrd'))]
    seg_dirs = [i for i in sorted(glob.glob(data_dir + '/raw_seg_p_n/' + '*nrrd'))]

    img_save_dir = proj_dir + '/HN_OUTCOME/HKTR_TCIA_DFCI/DFCI/raw_img'
    seg_save_dir = proj_dir + '/HN_OUTCOME/HKTR_TCIA_DFCI/DFCI/raw_seg'
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(seg_save_dir):
        os.makedirs(seg_save_dir)

    print('\n--- copy img files ---')
    img_ids = []
    new_ids = []
    i = 0
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        for seg_dir in seg_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                i += 1
                new_id = 'DFCI' + '_' + str(f'{i:03}') + '.nii.gz'
                print(i, new_id)
                img_save_path = img_save_dir + '/' + new_id
                seg_save_path = seg_save_dir + '/' + new_id
                img = sitk.ReadImage(img_dir)
                sitk.WriteImage(img, img_save_path)
                seg = sitk.ReadImage(seg_dir)
                sitk.WriteImage(seg, seg_save_path)
                # save IDs for reference
                img_ids.append(img_id)
                new_ids.append(new_id)
    df = pd.DataFrame({'old ID': img_ids, 'new ID': new_ids})
    df.to_csv(data_dir + '/DFCI_IDs.csv')


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong'
    raw_data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated'
    dataset = 'TCIA'

    if dataset == 'HKTR':
        transfer_HKTR_data(proj_dir)
    elif dataset == 'TCIA':
        transfer_TCIA_data(proj_dir)
    elif dataset == 'DFCI':
        transfer_DFCI_data(proj_dir)









