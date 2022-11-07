import os
import glob
import numpy as np
import shutil
import pydicom as dicom


def find_repeat(proj_dir):

    folders = []
    for folder in os.listdir(proj_dir):
        keys = []
        for img_dir in glob.glob(proj_dir + '/' + folder + '/*dcm'):
            key = img_dir.split('/')[-1].split('.')[0]
            if key == 'RTSTRUCT':
                keys.append(key)
            if len(keys) > 1:
                print(folder, len(keys))
                folders.append(folder)
    print('duplicate scans:', len(folders), folders)


def move_ct_to_folder(proj_dir):

    for folder in os.listdir(proj_dir):
        print(folder)
        img_dirs = []
        IDs = []
        for img_dir in glob.glob(proj_dir + '/' + folder + '/*dcm'):
            data_type = img_dir.split('/')[-1].split('.')[0]
            if data_type == 'CT':
                ID = img_dir.split('/')[-1].split('.')[-4]
                img_dirs.append(img_dir)
                IDs.append(ID)
                keys = list(set(IDs))
                #print(keys)
                for key in keys:
                    sub_folder = proj_dir + '/' + folder + '/' + key
                    if not os.path.exists(sub_folder):
                        os.makedirs(sub_folder)
        for img_dir, ID in zip(img_dirs, IDs):
            fn = img_dir.split('/')[-1]
            save_dir = proj_dir + '/' + folder + '/' + ID + '/' + fn
            shutil.move(img_dir, save_dir)


def seg_dicom(proj_dir):

    folders = []
    for folder in os.listdir(proj_dir):
        #print(folder)
        img_dirs = []
        for img_dir in glob.glob(proj_dir + '/' + folder + '/*dcm'):
            img_dirs.append(img_dir)
            if len(img_dirs) > 1:
                print(len(img_dirs))
                print(folder)
                try:
                    for img_dir in img_dirs:
                        ds = dicom.read_file(img_dir)
                        print(ds[0x3006, 0x0016])
                        print(ds)
                except Exception as e:
                    print(folder, e)
                #print(str(ds[0x0008, 0x1030]))


def test(proj_dir):
    case = '10047042048'
    img1 = 'CT.1.2.840.113619.2.55.3.3752517765.117.1202764551.849.1.dcm'
    img2 = 'CT.1.2.840.113619.2.55.3.3752517765.117.1202764552.1.153.dcm' 
    seg1 = 'RTSTRUCT.1.2.246.352.71.4.1039211570.2306282.20080222181706.dcm'
    seg2 = 'RTSTRUCT.1.2.246.352.71.4.1039211570.2321412.20080303162609.dcm'
    img1_dir = proj_dir + '/' + case + '/' + img1
    img2_dir = proj_dir + '/' + case + '/' + img2
    seg1_dir = proj_dir + '/' + case + '/' + seg1
    seg2_dir = proj_dir + '/' + case + '/' + seg2
    paths = [img1_dir, img2_dir, seg1_dir, seg2_dir]
    names = ['img1', 'img2', 'seg1', 'seg2']
    for path, name in zip(paths, names):
        ds = dicom.read_file(path)
        print(name, ds.StudyDate)


def test_seg(proj_dir):
    seg_dir = proj_dir + '/10021237143/RTSTRUCT.1.2.246.352.71.4.953043541824.350796.20140529103833.dcm'
    seg_ds = dicom.read_file(seg_dir)
    #print(seg_ds[0x0020, 0x0052])
    #print(seg_ds[0x3006, 0x0010][0])
    print(seg_ds)
    print(seg_ds['StudyDescription'])
    #print(str(seg_ds[0x0020, 0x0052]).split('.')[-4])
    #print(str(seg_ds[0x0020, 0x000e]).split('.')[-4])
    #print(seg_ds)


def move_seg_to_folder(proj_dir):

    bad_data = []
    for folder in os.listdir(proj_dir):
        #print(folder)
        seg_dirs = []
        for seg_dir in sorted(glob.glob(proj_dir + '/' + folder + '/*dcm')):
            seg_dirs.append(seg_dir)
            if seg_dirs != []:
                print(seg_dirs)
                #try:
                for seg_dir in seg_dirs:
                    ds = dicom.read_file(seg_dir)
                    print(ds[0x0020, 0x0052])
                    id0 = str(ds[0x0020, 0x0052]).split('.')[-4]
                    id1 = str(ds[0x0020, 0x0052]).split('.')[-5]
                    for x in [id0, id1]:
                        if len(x) > 5:
                            ID = x
                    print(ID)
                    fn = seg_dir.split('/')[-1]
                    save_dir = proj_dir + '/' + folder + '/' + ID + '/' + fn
                    shutil.move(seg_dir, save_dir)
                    print('sucessfully transfer rtstruct file!')
                #except Exception as e:
                #    print(folder, e)
                #    bad_data.append(folder + '_' + ID)
    print(bad_data)


def get_scan_type(proj_dir):
    
    bad_data = []
    count = 0
    for root, paths, files in os.walk(proj_dir):
        if not paths:
            count += 1
            #print(count, root)
            img_dirs = [i for i in sorted(glob.glob(root + '/*dcm'))]
            ds = dicom.read_file(img_dirs[0])
            try:
                study = str(ds['SeriesDescription'])
                #print(study)
                #if 'H+N' not in study or 'Head&Neck' not in study:
                if 'H+N' in study or 'Head' in study or 'H&N' in study:
                    save_dir = root + '_HN'
                    os.rename(root, save_dir)
                    print('successfully changed folder name!')
                else:
                    #print(ds['StudyDescription'])
                    print(ds['SeriesDescription'])
                    #print(ds['StudyDate'])
            except Exception as e:
                print(e)
                bad_data.append(root)
    print(bad_data)


if __name__ == '__main__':

    #proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/HN_Dicom_Export'
    #proj_dir = '/mnt/kannlab_rfa/Ben/NewerHNScans/OPX'
    proj_dir = '/mnt/kannlab_rfa/Ben/HN_NonOPC_Dicom_Export'
    test(proj_dir)
    #move_seg_to_folder(proj_dir)
    #test_seg(proj_dir)
    #get_scan_type(proj_dir)










