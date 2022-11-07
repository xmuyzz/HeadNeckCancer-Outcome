import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom as dicom


def image_header():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/HN_Dicom_Export/10087175377'
    data = 'CT.1.2.840.113619.2.55.3.380389780.37.1304454653.924.1.dcm'
    ds = dicom.read_file(proj_dir + '/' + data, force=True)
    print(ds)
    print(ds[0x0008, 0x103e])
    print(str(ds[0x0008, 0x103e]))
    if 'H+N SCAN/SCRAM' in str(ds[0x0008, 0x103e]):
        print('this is a head neck scan')


def seg_header():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/HN_Dicom_Export/10087175377'
    data = 'RTSTRUCT.1.2.246.352.71.4.1039211570.152922.20111020123613.dcm'
    ds = dicom.read_file(proj_dir + '/' + data, force=True)
    print(ds)
    print(ds[0x0008, 0x1030])
    print(str(ds[0x0008, 0x1030]))


def find():
    proj_dir = '/mnt/kannlab_rfa/Ben/NewerHNScans/OPX'
    count = 0
    for root, dirs, files in os.walk(proj_dir):
        if not dirs:
            segs = []
            ID = root.split('/')[-1]
            for img_dir in glob.glob(root + '/*dcm'):
                seg = img_dir.split('/')[-1].split('.')[0]
                if seg == 'RTSTRUCT':
                    segs.append(seg)
                if len(segs) > 1:
                    count += 1
                    print(count, ID)

def bwh():
    bwh2_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH2/raw_img'
    bwh3_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH3/raw_img'
    file2s = []
    for dir2 in glob.glob(bwh2_dir + '/*nrrd'):
        file2 = dir2.split('/')[-1]
        file2s.append(file2)
    count = 0
    for dir3 in glob.glob(bwh3_dir + '/*nrrd'):
        file3 = dir3.split('/')[-1]
        count += 1
        print(count)
        if file3 in file2s:
            print('repeated scan:', file3)


if __name__ == '__main__':

    #image_header()
    #seg_header()
    bwh()
