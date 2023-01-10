import os
import pandas as pd
import glob
import shutil


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



if __name__ == '__main__':

    check_data()





