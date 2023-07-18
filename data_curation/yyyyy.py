import os
import pandas as pd
import glob
import shutil
import pydicom as dicom
from distutils.dir_util import copy_tree
from subprocess import call

def image_header():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/HN_Dicom_Export/10087175377'
    data = 'CT.1.2.840.113619.2.55.3.380389780.37.1304454653.924.1.dcm'
    ds = dicom.read_file(proj_dir + '/' + data, force=True)
    print(ds)
    print(ds[0x0008, 0x103e])
    print(str(ds[0x0008, 0x103e]))
    if 'H+N SCAN/SCRAM' in str(ds[0x0008, 0x103e]):
        print('this is a head neck scan')

def check():
    bwh_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH/raw_img'
    OPC3_dir = '/mnt/kannlab_rfa/Ben/NewerHNScans/OPX'
    bwh_ids = []
    for img_dir in glob.glob(bwh_dir + '/*nrrd'):
        bwh_id = img_dir.split('.')[-2]
        print(bwh_id)
        bwh_ids.append(bwh_id)
    opc2_ids = []
    for folder in os.listdir(OPC3_dir):
        print(folder)
        opc2_ids.append(folder)
    ids = [i for i in opc2_ids if i not in bwh_ids]
    print(ids)


def move_data():
    id_img_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/DFCI/new_curation/raw_img'
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data'
    raw_img_dir = proj_dir + '/BWH/raw_img'
    out_img_dir = proj_dir + '/OPC1/raw_img'
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    names = []
    Es = []
    for i, img_dir in enumerate(glob.glob(id_img_dir + '/*nrrd')):
        name = img_dir.split('/')[-1]
        print(i, name)
        raw_img_path = raw_img_dir + '/' + name
        out_img_path = out_img_dir + '/' + name
        try:
            shutil.move(raw_img_path, out_img_path)
        except Exception as e:
            print(name, e)
            names.append(name)
            Es.append(e)
    print(names)
    print(Es)


def check_data():
    opc1_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/DFCI/new_curation/uncombined_seg'
    opc2_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/HN_Dicom_Export'
    opc1_ids = []
    opc2_ids = []
    for folder in os.listdir(opc1_dir):
        opc1_id = folder.split('_')[1]
        opc1_ids.append(opc1_id)
    for folder in os.listdir(opc2_dir):
        opc2_ids.append(folder)
    overlap = list(set(opc1_ids) & set(opc2_ids))
    print(len(overlap))
    print(len(opc1_ids))
    print(len(opc2_ids))


def change_name():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/BWH_TOT'
    pmrns = []
    for img_path in glob.glob(proj_dir + '/raw_img/*nrrd'):
        ID = img_path.split('/')[-1].split('.')[0]
        if ID.split('_'):
            pmrn = ID.split('_')[0]
            pmrns.append(pmrn)
    dups = list(set([i for i in pmrns if pmrns.count(i) > 1]))
    print(dups)
    for img_path in glob.glob(proj_dir + '/raw_img/*nrrd'):
        ID = img_path.split('/')[-1].split('.')[0]
        if ID.split('_'):
            pmrn = ID.split('_')[0]
            if pmrn not in dups:
                print(ID)
                save_path = proj_dir + '/raw_img/' + pmrn + '.nrrd'
                os.rename(img_path, save_path)


def change_folder_name():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/BWH_TOT'
    pmrns = []
    for gtv_path in os.listdir(proj_dir + '/raw_gtv'):
        ID = gtv_path.split('/')[-1].split('.')[0]
        if ID.split('_'):
            pmrn = ID.split('_')[0]
            pmrns.append(pmrn)
    dups = list(set([i for i in pmrns if pmrns.count(i) > 1]))
    print(dups)
    for gtv_path in glob.glob(proj_dir + '/raw_gtv/*'):
        ID = gtv_path.split('/')[-1].split('.')[0]
        if ID.split('_'):
            pmrn = ID.split('_')[0]
            if pmrn not in dups:
                print(ID)
                save_path = proj_dir + '/raw_gtv/' + pmrn
                os.rename(gtv_path, save_path)


def change_folder():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/OPC2/dcm'
    for img_dir in glob.glob(proj_dir + '/*'):
        folder = img_dir.split('/')[-1]
        if folder.split('_')[-1] == 'HN':
            ID = folder.split('_')[0] + '_' + folder.split('_')[1]
        else:
            ID = folder
        print(ID)
        save_dir = proj_dir + '/' + ID
        os.rename(img_dir, save_dir)


def tot_dice():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/BWH_TOT/clinical_data'
    df0 = pd.read_csv(proj_dir + '/seg_df.csv')
    df0.drop(['p', 'n', 'pn'], axis=1, inplace=True)
    df = pd.read_csv(proj_dir + '/dice_sum.csv')
    df = df[df['nn_id'].notna()]
    df = df0.merge(df, how='left', on='PMRN').reset_index()
    df = df[df['nn_id'].notna()]
    df.to_csv(proj_dir + '/sum.csv', index=False)


def get_dcm_header():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data'
    df = pd.read_csv(proj_dir + '/BWH_TOT/clinical_data/sum.csv')
    dates = []
    for count, ID in enumerate(df['ID']):
        print(count, ID)
        data_dir = proj_dir + '/OPC2/dcm/' + ID
        dcms = [i for i in os.listdir(data_dir)]
        ds = dicom.read_file(data_dir + '/' + dcms[0], force=True)
        date = ds[0x0008, 0x0020][-8:]
        dates.append(date)
        print(date)
    print(dates)
    df['CT date'] = dates
    df.to_csv(proj_dir + '/BWH_TOT/clinical_data/sum.csv', index=False)

def xxx():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data'
    df0 = pd.read_csv(proj_dir + '/BWH_TOT/clinical_data/sum.csv')
    df1 = pd.read_csv(proj_dir + '/BWH_TOT/clinical_data/bwh_meta.csv', encoding='unicode_escape', low_memory=False)
    df = df0.merge(df1, how='left', on='PMRN')
    pmrns = df['PMRN'].to_list()
    dups = list(set([i for i in pmrns if pmrns.count(i) > 1]))
    print(dups)
    IDs = []
    for ID in df['ID']:
        pmrn = ID.split('_')[0]
        if int(pmrn) not in dups:
            new_id = pmrn
            IDs.append(new_id)
        else:
            print(ID)
            new_id = ID
            IDs.append(new_id)
    print(IDs)
    df['id'] = IDs
    df.to_csv(proj_dir + '/BWH_TOT/clinical_data/sum_meta.csv')



def review_data():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/BWH_TOT'
    raw_img_dir = proj_dir + '/raw_img'
    raw_gtv_dir = proj_dir + '/raw_gtv'
    save_img_dir = proj_dir + '/review_data/raw_img'
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    df = pd.read_csv(proj_dir + '/clinical_data/sum_meta.csv')
    IDs = []
    errors = []
    for i, ID in enumerate(df['id']):
        print(i, ID)
        ID = str(ID)
        try:
            # copy raw_img
            print('copy img ...')
            raw_img_path = raw_img_dir + '/' + ID + '.nrrd'
            save_img_path = save_img_dir + '/' + ID + '.nrrd'
            shutil.copyfile(raw_img_path, save_img_path)
            # copy gtv folder
            print('copy gtv ...')
            save_gtv_dir = proj_dir + '/review_data/raw_gtv/' + ID
            if not os.path.exists(save_gtv_dir):
                os.makedirs(save_gtv_dir)
            for raw_gtv_path in glob.glob(raw_gtv_dir + '/' +ID + '/*nrrd'):
                gtv = raw_gtv_path.split('/')[-1]
                save_gtv_path = save_gtv_dir + '/' + gtv 
                shutil.copyfile(raw_gtv_path, save_gtv_path) 
        except Exception as e:
            print(ID, e)
            IDs.append(ID)
            errors.append(e)
    print(IDs)
    print(errors)


def review_data2():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck'
    nnUNet_dir = proj_dir + '/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task502_tot_p_n'
    review_dir = proj_dir + '/data/BWH_TOT/review_data'
    save_img_dir = review_dir + '/nnUNet_img'
    save_seg_dir = review_dir + '/nnUNet_seg'
    save_pre_dir = review_dir + '/nnUNet_pre'
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    if not os.path.exists(save_seg_dir):
        os.makedirs(save_seg_dir)
    if not os.path.exists(save_pre_dir):
        os.makedirs(save_pre_dir)
    df = pd.read_csv(review_dir + '/sum_meta.csv')
    for i, ID in enumerate(df['nn_id']):
        print(i, ID)
        img_path = nnUNet_dir + '/imagesTs4/' + ID + '_0000.nii.gz'
        seg_path = nnUNet_dir + '/labelsTs4/' + ID + '.nii.gz'
        pre_path = nnUNet_dir + '/predsTs4/' + ID + '.nii.gz'
        save_img_path = save_img_dir + '/' + ID + '.nii.gz'
        save_seg_path = save_seg_dir + '/' + ID + '.nii.gz'
        save_pre_path = save_pre_dir + '/' + ID + '.nii.gz'
        shutil.copyfile(img_path, save_img_path)
        shutil.copyfile(seg_path, save_seg_path)
        shutil.copyfile(pre_path, save_pre_path)

if __name__ == '__main__':
    #xxx()
    review_data2()



