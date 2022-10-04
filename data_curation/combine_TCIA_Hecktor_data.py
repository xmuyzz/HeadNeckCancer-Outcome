import sys
import os
import pydicom
import glob
import SimpleITK as sitk
import nibabel as nib
import numpy as np


def combine_hecktor_tcia(proj_dir):

    """
    save nrrd to nii 
    """
    TCIA_img_dir = proj_dir + '/TCIA/img'
    TCIA_seg_dir = proj_dir + '/TCIA/seg_p_n'
    Heck_img_dir = proj_dir + '/Hecktor_TCIA_data/raw_img'
    Heck_seg_dir = proj_dir + '/Hecktor_TCIA_data/raw_seg'

    # rename MDACC_xxx.nii.gz with MDA_xxx.nii.gz
    count = 0
    for data_dir in [TCIA_img_dir, TCIA_seg_dir]:
        for root, subdirs, files in os.walk(data_dir):
            for fn in files:
                old_path = os.path.join(root, fn)
                if 'MDACC' in fn:
                    count += 1
                    new_fn = 'MDA_' + fn.split('_')[1]
                    print(count, fn)
                    old_path = os.path.join(root, fn)
                    new_path = os.path.join(root, new_fn)
                    print(count, new_fn)
                    os.rename(old_path, new_path)
    print('complete renaming!')

    # get hecktor data id
    img_ids = []
    for img_dir in sorted(glob.glob(Heck_img_dir + '/*nii.gz')):
        img_id = img_dir.split('/')[-1].split('.')[0]
        img_ids.append(img_id)
    print('\ntotal hecktor data:', len(img_ids))
    seg_ids = []
    for seg_dir in sorted(glob.glob(Heck_seg_dir + '/*nii.gz')):
        seg_id = seg_dir.split('/')[-1].split('.')[0]
        seg_ids.append(seg_id)
    print('\ntotal hecktor data:', len(seg_ids))

    # move TCIA data to hecktor
    count = 0
    print('---combine img data----')
    for data_dir in sorted(glob.glob(TCIA_img_dir + '/*nrrd')):
        data_id = data_dir.split('/')[-1].split('.')[0]
        if data_id not in img_ids:
            count += 1
            print(count, data_id)
            data = sitk.ReadImage(data_dir)
            sitk.WriteImage(data, Heck_img_dir + '/' + data_id + '.nii.gz')
    print('---combine seg data----')
    for data_dir in sorted(glob.glob(TCIA_seg_dir + '/*nrrd')):
        data_id = data_dir.split('/')[-1].split('.')[0]
        if data_id not in seg_ids:
            count += 1
            print(count, data_id)
            data = sitk.ReadImage(data_dir)
            sitk.WriteImage(data, Heck_seg_dir + '/' + data_id + '.nii.gz')


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    
    combine_hecktor_tcia(proj_dir)




