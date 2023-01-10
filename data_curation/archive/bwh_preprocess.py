import sys
import os
import pydicom
import glob
import SimpleITK as sitk
import pandas as pd
import numpy as np
from dcm_to_nrrd import dcm_to_nrrd
from rtstruct_to_nrrd import rtstruct_to_nrrd
from combine_structures import combine_structures
from interpolate import interpolate
from crop_image import crop_top, crop_top_image_only, crop_full_body
from registration import nrrd_reg_rigid
import SimpleITK as sitk
import shutil
import nibabel as ni


def change_img_name(proj_dir):
    count = 0
    for root, subdirs, files in os.walk(proj_dir + '/raw_img'):
        for fn in files:
            count += 1
            #print(fn)
            old_path = os.path.join(root, fn)
            #fn = fn.replace('-', '_')
            new_fn = fn.split('_')[1] + '.nrrd'
            print(count, new_fn)
            new_path = os.path.join(root, new_fn)
            print(count, new_path)
            os.rename(old_path, new_path) 


def get_seg_info(proj_dir):
    """
    COMBINING MASKS 
    """
    dfci_img_dir = proj_dir + '/raw_img'
    dfci_seg_dir = proj_dir + '/uncombined_seg'
    output_dir = proj_dir + '/output_data' 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _pn_list = ['cm', 'mm', '+', '**', 'z', 'none', 'resident', 'pre', 'dfrm', 'Dfmd', 'dfmd','Dfrmd', 
                'CTV', 'Pre', 'Expansion', 'virtual', 'Virtual', 'LX', 'TEMP', 'expand', 'larynx', 'vGTV']
    _N_list = ['TONSIL', 'Prim', 'NPX', 'OPX', 'FINAL', 'Pre', 'Tonsil', 'New', 'Dfrmd', 'TNSL']
    _n_list = ['tonsil', 'TONSIL', 'expand', 'noFDG', 'resident', 'Central', 'central', 'Joint', 'PT', 
               'nodose', 'Tnsil', 'Tonsil', 'yinitialGTVp', 'Pre', 'larynx', 'Combined', 'p', 'new']
    n_list = ['L 2', 'R 3', 'L_lvl', 'R_lvl', 'parotid', 'L2', 'parap', 'LII', 'RII', 'R_2']
    p_list = ['BOT', 'P', 'p', 'pn', 'Tonsil', 'Primary', 'TONSIL', 'prim', 'primary', 'RTONSIL', 'PRIMARY', 
              'tonsil', 'Prim', 'phar', 'OPX', 'HPX', 'palate', 'larynx', 'Central', 'central', 'TNSL', 'tumor', '_p_70_n'] 
    _p_list = ['NK', 'LN', 'parotid', 'parap', 'neck', 'Neck', 'RP', 'Node', 'RII', 'LII', 'R_2', 'GTVn']
    seg_IDsss = []
    for tumor_type in ['pn', 'n', 'p']:
        if tumor_type == 'pn':
            pat_IDs = []
            seg_IDss = []    
            for img_dir in sorted(glob.glob(dfci_img_dir + '/*nrrd')):
                seg_dirs = []
                seg_IDs = []
                pat_id = img_dir.split('/')[-1].split('.')[0]
                #print(pat_id)
                for folder in os.listdir(dfci_seg_dir):
                    ID = folder.split('.')[0]
                    if ID == pat_id:
                        #print(ID)
                        for seg_dir in glob.glob(os.path.join(dfci_seg_dir, folder) + '/*nrrd'):
                            seg_name = seg_dir.split('/')[-1].split('.')[0]
                            if 'GTV' in seg_name and not any(x in seg_name for x in _pn_list):
                                seg_dirs.append(seg_dir)
                                seg_IDs.append(seg_name)
                        print(pat_id, seg_IDs)
                pat_IDs.append(pat_id)
                seg_IDss.append(seg_IDs)
        elif tumor_type == 'n':
            pat_IDs = []
            seg_IDss = []
            for img_dir in sorted(glob.glob(dfci_img_dir + '/*nrrd')):
                seg_dirs = []
                seg_IDs = []
                pat_id = img_dir.split('/')[-1].split('.')[0]
                #print(pat_id)
                for folder in os.listdir(dfci_seg_dir):
                    ID = folder.split('.')[0]
                    if ID == pat_id:
                        #print(ID)
                        for seg_dir in glob.glob(os.path.join(dfci_seg_dir, folder) + '/*nrrd'):
                            seg_name = seg_dir.split('/')[-1].split('.')[0]
                            if 'GTV' in seg_name and not any(x in seg_name for x in _pn_list):
                                if 'N' in seg_name and not any(x in seg_name for x in _N_list):
                                    seg_dirs.append(seg_dir)
                                    seg_IDs.append(seg_name)
                                elif 'n' in seg_name and not any(x in seg_name for x in _n_list):
                                    seg_dirs.append(seg_dir)
                                    seg_IDs.append(seg_name)
                                elif any(x in seg_name for x in n_list):
                                    seg_dirs.append(seg_dir)
                                    seg_IDs.append(seg_name)
                        print(pat_id, seg_IDs)
                pat_IDs.append(pat_id)
                seg_IDss.append(seg_IDs)
        elif tumor_type == 'p':
            pat_IDs = []
            seg_IDss = []
            for img_dir in sorted(glob.glob(dfci_img_dir + '/*nrrd')):
                seg_dirs = []
                seg_IDs = []
                pat_id = img_dir.split('/')[-1].split('.')[0]
                #print(pat_id)
                for folder in os.listdir(dfci_seg_dir):
                    ID = folder.split('.')[0]
                    if ID == pat_id:
                        #print(ID)
                        for seg_dir in glob.glob(os.path.join(dfci_seg_dir, folder) + '/*nrrd'):
                            seg_name = seg_dir.split('/')[-1].split('.')[0]
                            if 'GTV' in seg_name and not any(x in seg_name for x in _pn_list):
                                if any(x in seg_name for x in p_list) and not any(x in seg_name for x in _p_list):
                                    seg_dirs.append(seg_dir)
                                    seg_IDs.append(seg_name)
                        print(pat_id, seg_IDs)
                pat_IDs.append(pat_id)
                seg_IDss.append(seg_IDs)
        seg_IDsss.append(seg_IDss)
    df = pd.DataFrame({'pat id': pat_IDs, 'pn': seg_IDsss[0], 'n': seg_IDsss[1], 'p': seg_IDsss[2]})
    fn = output_dir + '/GTV_seg.csv'
    df.to_csv(fn, index=True)


def combine_mask2(proj_dir, tumor_type):
    """
    COMBINING MASKS 
    """
    raw_img_dir = proj_dir + '/raw_img'
    uncombined_seg_dir = proj_dir + '/uncombined_seg'
    seg_n_save_dir = proj_dir + '/raw_seg_n'
    seg_p_save_dir = proj_dir + '/raw_seg_p'
    seg_pn_save_dir = proj_dir + '/raw_seg_pn'
    if not os.path.exists(seg_n_save_dir):
        os.makedirs(seg_n_save_dir)
    if not os.path.exists(seg_p_save_dir):
        os.makedirs(seg_p_save_dir)
    if not os.path.exists(seg_pn_save_dir):
        os.makedirs(seg_pn_save_dir)
    if tumor_type == 'n':
        seg_save_dir = seg_n_save_dir
    if tumor_type == 'p':
        seg_save_dir = seg_p_save_dir
    if tumor_type == 'pn':
        seg_save_dir = seg_pn_save_dir
    
    df = pd.read_csv(proj_dir + '/output_data/GTV_seg.csv', converters={tumor_type: pd.eval})
    img_ids = []
    seg_namess = []
    bad_data = []
    count = 0
    for img_dir in sorted(glob.glob(raw_img_dir + '/*nrrd')):
        seg_names = []
        seg_dirs = []
        #img_id = img_dir.split('/')[-1].split('_')[1]
        img_id = img_dir.split('/')[-1].split('.')[0]
        #print(img_id)
        for seg_folder in os.listdir(uncombined_seg_dir):
            #print(seg_folder)
            #seg_id = seg_folder.split('_')[1]
            seg_id = str(seg_folder)
            #print(seg_id)
            if seg_id == img_id:
                count += 1
                print(count, 'ID:', seg_id)
                for df_id, gtv_list in zip(df['pat id'], df[tumor_type]):
                    if df_id == seg_id:
                        print(df_id)
                        print(gtv_list)
                        if gtv_list:
                            for gtv in gtv_list:
                                gtv_dir = uncombined_seg_dir + '/' + seg_folder + '/' + gtv + '.nrrd'
                                #print(gtv_dir)
                                seg_dirs.append(gtv_dir)
                        else:
                            print('empty GTV list!')
                try:
                    combined_mask = combine_structures(
                        patient_id=seg_id, 
                        mask_arr=seg_dirs, 
                        path_to_reference_image_nrrd=img_dir, 
                        binary=2, 
                        return_type='sitk_object', 
                        output_dir=seg_save_dir)
                    print('combine successfully')
                except Exception as e:
                    print(seg_id, e)
                    bad_data.append(seg_id)
    print('bad data:', bad_data)


def combine_mask(proj_dir, tumor_type):
    """
    COMBINING MASKS 
    """
    raw_img_dir = proj_dir + '/raw_img'
    uncombined_seg_dir = proj_dir + '/uncombined_seg'
    seg_n_save_dir = proj_dir + '/raw_seg_n'
    seg_p_save_dir = proj_dir + '/raw_seg_p'
    seg_pn_save_dir = proj_dir + '/raw_seg_pn'
    if not os.path.exists(seg_n_save_dir):
        os.makedirs(seg_n_save_dir)
    if not os.path.exists(seg_p_save_dir):
        os.makedirs(seg_p_save_dir)
    if not os.path.exists(seg_pn_save_dir):
        os.makedirs(seg_pn_save_dir)
    if tumor_type == 'n':
        seg_save_dir = seg_n_save_dir
        csv_save_dir = proj_dir + '/combined_seg_n.csv'
    if tumor_type == 'p':
        seg_save_dir = seg_p_save_dir
        csv_save_dir = proj_dir + '/combined_seg_p.csv'
    if tumor_type == 'pn':
        seg_save_dir = seg_pn_save_dir
        csv_save_dir = proj_dir + '/combined_seg_pn.csv'
    img_ids = []
    seg_namess = []
    count = 0
    for img_dir in sorted(glob.glob(raw_img_dir + '/*nrrd')):
        seg_names = []
        seg_dirs = []
        #img_id = img_dir.split('/')[-1].split('_')[1]
        img_id = img_dir.split('/')[-1].split('.')[0]
        #print(img_id)
        for seg_folder in os.listdir(uncombined_seg_dir):
            #print(seg_folder)
            #seg_id = seg_folder.split('_')[1]
            seg_id = str(seg_folder)
            #print(seg_id)
            if seg_id == img_id:
                count += 1
                print(count, 'ID:', seg_id)
                for seg_dir in glob.glob(uncombined_seg_dir + '/' + seg_folder + '/*nrrd'):
                    seg_name = seg_dir.split('/')[-1].split('.')[0]
                    #print(seg_dir)
                    #print(seg_name)
                    if tumor_type == 'pn':
                        if 'GTV' in seg_name and 'cm' not in seg_name and 'mm' not in seg_name \
                            and '+' not in seg_name and '**' not in seg_name and 'z' not in seg_name \
                            and 'resident' not in seg_name and 'pre' not in seg_name and 'Dfmd' not in seg_name:
                            seg_dirs.append(seg_dir)
                            seg_names.append(seg_name)
                    #print('seg names:', seg_names)
                    #print(seg_dir)
                    elif tumor_type == 'n':
                        if 'GTV' in seg_name and 'cm' not in seg_name and 'mm' not in seg_name \
                            and '+' not in seg_name and '**' not in seg_name and 'z' not in seg_name \
                            and 'resident' not in seg_name and 'pre' not in seg_name and 'Dfmd' not in seg_name:
                            if 'N' in seg_name and 'TONSIL' not in seg_name and 'Prim' not in seg_name \
                                and 'NPX' not in seg_name and 'Neck' not in seg_name and 'FINAL' not in seg_name:
                                seg_dirs.append(seg_dir)
                                seg_names.append(seg_name)
                            elif 'n' in seg_name and 'tonsil' not in seg_name and 'Tonsil' not in seg_name \
                                and 'expand' not in seg_name and 'noFDG' not in seg_name and 'resident' \
                                not in seg_name and 'neck' not in seg_name:
                                seg_dirs.append(seg_dir)
                                seg_names.append(seg_name)
                    elif tumor_type == 'p':
                        if 'GTV' in seg_name and 'cm' not in seg_name and 'mm' not in seg_name \
                            and '+' not in seg_name and '**' not in seg_name and 'z' not in seg_name \
                            and 'resident' not in seg_name and 'pre' not in seg_name and 'Dfmd' not in seg_name:
                            if 'BOT' in seg_name or 'P' in seg_name or 'Tonsil' in seg_name \
                                or 'Primary' in seg_name or 'TONSIL' in seg_name or 'prim' in seg_name \
                                or 'primary' in seg_name or 'RTONSIL' in seg_name or 'PRIMARY' in seg_name \
                                or 'tonsil' in seg_name or 'Prim' in seg_name or 'phar' in seg_name \
                                or 'OPX' in seg_name or 'HPX' in seg_name or 'palate' in seg_name \
                                or 'phar' in seg_name:
                                seg_dirs.append(seg_dir)
                                seg_names.append(seg_name)

                try:
                    combined_mask = combine_structures(
                        patient_id=seg_id, 
                        mask_arr=seg_dirs, 
                        path_to_reference_image_nrrd=img_dir, 
                        binary=2, 
                        return_type='sitk_object', 
                        output_dir=seg_save_dir)
                    print('combine successfully')
                except Exception as e:
                    print(seg_id, e)
                    if seg_names == []:
                        print('no GTV!')
        img_ids.append(img_id)
        seg_namess.append(seg_names)
    # save mask information
    df = pd.DataFrame({'pat id': img_ids, 'seg id': seg_namess})
    df.to_csv(csv_save_dir, index=False)            


def get_pn_seg(proj_dir):
    """
    1) combine p_seg and n_seg to a 4d nii image;
    2) p_seg and n_seg in different channels;
    Args:
        proj_dir {path} -- project path
    Returns:
        save nii files
    Raise issues:
        none
    """
    p_seg_path = proj_dir + '/raw_seg_p'
    n_seg_path = proj_dir + '/raw_seg_n'
    pn_seg_path = proj_dir + '/raw_seg_pn'
    p_n_seg_path = proj_dir + '/raw_seg_p_n'
    img_path = proj_dir + '/raw_img'
    if not os.path.exists(p_n_seg_path):
        os.makedirs(p_n_seg_path)
    fns = [i for i in sorted(os.listdir(pn_seg_path))]
    for i, fn in enumerate(fns):
        try:
            pat_id = fn.split('.')[0]
            print(i, pat_id)
            # image
            img_dir = img_path + '/' + fn
            img = sitk.ReadImage(img_dir)
            arr = sitk.GetArrayFromImage(img)
            # primary tumor
            p_seg_dir = p_seg_path + '/' + fn
            if os.path.exists(p_seg_dir):
                p_seg = sitk.ReadImage(p_seg_dir)
                p_seg = sitk.GetArrayFromImage(p_seg)
                p_seg[p_seg != 0] = 1
                #print('p_seg shape:', p_seg.shape)
            else:
                print('no primary segmentation...')
                p_seg = np.zeros(shape=arr.shape)
            # node
            n_seg_dir = n_seg_path + '/' + fn
            if os.path.exists(n_seg_dir):
                n_seg = sitk.ReadImage(n_seg_dir)
                n_seg = sitk.GetArrayFromImage(n_seg)
                n_seg[n_seg != 0] = 2
            else:
                print('no node segmentation...')
                n_seg = np.zeros(shape=arr.shape)
            # combine P and N to one np arr
            p_n_seg = np.add(p_seg, n_seg).astype(int)
            # change dtype, otherwise nrrd cannot be read
            p_n_seg = np.asarray(p_n_seg, dtype='uint8')
            # some voxels from P and N have overlap
            p_n_seg[p_n_seg == 3] = 1
            sitk_obj = sitk.GetImageFromArray(p_n_seg)
            sitk_obj.SetSpacing(img.GetSpacing())
            sitk_obj.SetOrigin(img.GetOrigin())
            # write new nrrd
            writer = sitk.ImageFileWriter()
            writer.SetFileName(p_n_seg_path + '/' + pat_id + '.nrrd')
            writer.SetUseCompression(True)
            writer.Execute(sitk_obj)
        except Exception as e:
            print(pat_id, e)


def interp_reg_crop(proj_dir, root_dir, tumor_type, image_format, crop_shape):
    """
    Rigid Registration - followed by top crop
    """
    print('------start registration--------')
    img_raw_dir = proj_dir + '/raw_img'
    seg_p_n_raw_dir = proj_dir + '/raw_seg_p_n'
    seg_pn_raw_dir = proj_dir + '/raw_seg_pn'
    seg_p_raw_dir = proj_dir + '/raw_seg_p'
    seg_n_raw_dir = proj_dir + '/raw_seg_n'

    img_crop_dir = proj_dir + '/crop_img_160'
    seg_p_n_crop_dir = proj_dir + '/crop_seg_p_n_160'
    seg_pn_crop_dir = proj_dir + '/crop_seg_pn_160'
    seg_p_crop_dir = proj_dir + '/crop_seg_p'
    seg_n_crop_dir = proj_dir + '/crop_seg_n'
    if not os.path.exists(img_crop_dir):
        os.makedirs(img_crop_dir)
    if not os.path.exists(seg_p_n_crop_dir):
        os.makedirs(seg_p_n_crop_dir)
    if not os.path.exists(seg_pn_crop_dir):
        os.makedirs(seg_pn_crop_dir)
    if not os.path.exists(seg_p_crop_dir):
        os.makedirs(seg_p_crop_dir)
    if not os.path.exists(seg_n_crop_dir):
        os.makedirs(seg_n_crop_dir)

    img_dirs = [i for i in sorted(glob.glob(img_raw_dir + '/*nrrd'))]
    seg_p_n_dirs = [i for i in sorted(glob.glob(seg_p_n_raw_dir + '/*nrrd'))]
    seg_pn_dirs = [i for i in sorted(glob.glob(seg_pn_raw_dir + '/*nrrd'))]
    seg_p_dirs = [i for i in sorted(glob.glob(seg_p_raw_dir + '/*nrrd'))]
    seg_n_dirs = [i for i in sorted(glob.glob(seg_n_raw_dir + '/*nrrd'))]
    if tumor_type == 'p_n':
        seg_dirs = seg_p_n_dirs
        seg_crop_dir = seg_p_n_crop_dir
    elif tumor_type == 'pn':
        seg_dirs = seg_pn_dirs
        seg_crop_dir = seg_pn_crop_dir
    elif tumor_type == 'p':
        seg_dirs = seg_p_dirs
        seg_crop_dir = seg_p_crop_dir
    elif tumor_type == 'n':
        seg_dirs = seg_n_dirs
        seg_crop_dir = seg_n_crop_dir
    img_ids = []
    bad_ids = []
    bad_scans = []
    count = 0
    # get register template
    fixed_img_dir = root_dir + '/DFCI/img_interp/10020741814.nrrd'
    fixed_img = sitk.ReadImage(fixed_img_dir, sitk.sitkFloat32)
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        #print(img_id)
        for seg_dir in seg_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            #print(seg_id)
            if img_id == seg_id:
                img_ids.append(img_id)
                count += 1
                print(count, img_id)
                # load img and seg
                img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
                seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
                # --- crop full body scan ---
                z_img = img.GetSize()[2]
                z_seg = seg.GetSize()[2]
                if z_img < 105:
                    print('This is an incomplete scan!')
                    bad_scans.append(seg_id)
                else:
                    if z_img > 200:
                        img = crop_full_body(img, int(z_img * 0.65))
                        seg = crop_full_body(seg, int(z_seg * 0.65))
                    try:
                        # --- interpolation for image and seg to 1x1x3 ---
                        # interpolate images
                        print('interplolate')
                        img_interp = interpolate(
                            patient_id=img_id, 
                            path_to_nrrd=img_dir, 
                            interpolation_type='linear', #"linear" for image
                            new_spacing=(1, 1, 3), 
                            return_type='sitk_obj', 
                            output_dir='',
                            image_format=image_format)
                        # interpolate segs
                        seg_interp = interpolate(
                            patient_id=img_id, 
                            path_to_nrrd=seg_dir, 
                            interpolation_type='nearest_neighbor', # nearest neighbor for label
                            new_spacing=(1, 1, 3), 
                            return_type='sitk_obj', 
                            output_dir='',
                            image_format=image_format)                
                        # --- registration for image and seg to 1x1x3 ---    
                        # register images
                        print('register')
                        reg_img, fixed_img, moving_img, final_transform = nrrd_reg_rigid( 
                            patient_id=img_id, 
                            moving_img=img_interp, 
                            output_dir='', 
                            fixed_img=fixed_img,
                            image_format=image_format)
                        # register segmentations
                        reg_seg = sitk.Resample(
                            seg_interp, 
                            fixed_img, 
                            final_transform, 
                            sitk.sitkNearestNeighbor, 
                            0.0, 
                            moving_img.GetPixelID())
                        # --- crop ---
                        print('cropping')
                        crop_top(
                            patient_id=img_id,
                            img=reg_img,
                            seg=reg_seg,
                            crop_shape=crop_shape,
                            return_type='sitk_object',
                            output_img_dir=img_crop_dir,
                            output_seg_dir=seg_crop_dir,
                            image_format=image_format)
                        print('successfully crop!')
                    except Exception as e:
                        bad_ids.append(img_id)
                        print(img_id, e)
        print('bad ids:', bad_ids)
        print('incomplete scans:', bad_scans)


def crop(proj_dir, tumor_type, crop_shape, image_format):    
    """
With TOP-CROP HPC ### NEED TO RUN FOR image_crop, image_crop_p, and image_crop_n  
    WILL ONLY WORK WITH SPACING = 1,1,3
    """
    print('------start cropping--------')
    img_reg_dir = proj_dir + '/reg_img2'
    seg_p_n_reg_dir = proj_dir + '/reg_seg_p_n2'
    seg_pn_reg_dir = proj_dir + '/reg_seg_pn2'
    seg_p_reg_dir = proj_dir + '/reg_seg_p2'
    seg_n_reg_dir = proj_dir + '/reg_seg_n2'
    img_crop_dir = proj_dir + '/crop_img'
    seg_p_n_crop_dir = proj_dir + '/crop_seg_p_n2'
    seg_pn_crop_dir = proj_dir + '/crop_seg_pn2'
    seg_p_crop_dir = proj_dir + '/crop_seg_p2'
    seg_n_crop_dir = proj_dir + '/crop_seg_n2'
    if not os.path.exists(img_crop_dir):
        os.makedirs(img_crop_dir)
    if not os.path.exists(seg_p_n_crop_dir):
        os.makedirs(seg_p_n_crop_dir)
    if not os.path.exists(seg_pn_crop_dir):
        os.makedirs(seg_pn_crop_dir)
    if not os.path.exists(seg_p_crop_dir):
        os.makedirs(seg_p_crop_dir)
    if not os.path.exists(seg_n_crop_dir):
        os.makedirs(seg_n_crop_dir)
    img_reg_dirs = [i for i in sorted(glob.glob(img_reg_dir + '/*nrrd'))]
    seg_p_n_reg_dirs = [i for i in sorted(glob.glob(seg_p_n_reg_dir + '/*nrrd'))]
    seg_pn_reg_dirs = [i for i in sorted(glob.glob(seg_pn_reg_dir + '/*nrrd'))]
    seg_p_reg_dirs = [i for i in sorted(glob.glob(seg_p_reg_dir + '/*nrrd'))]
    seg_n_reg_dirs = [i for i in sorted(glob.glob(seg_n_reg_dir + '/*nrrd'))]
    if tumor_type == 'p_n':
        seg_dirs = seg_p_n_reg_dirs
        seg_crop_dir = seg_p_n_crop_dir
    elif tumor_type == 'pn':
        seg_dirs = seg_pn_reg_dirs
        seg_crop_dir = seg_pn_crop_dir
    elif tumor_type == 'p':
        seg_dirs = seg_p_reg_dirs
        seg_crop_dir = seg_p_crop_dir
    elif tumor_type == 'n':
        seg_dirs = seg_n_reg_dirs
        seg_crop_dir = seg_n_crop_dir
    # registration for image and seg
    img_ids = []
    count = 0
    for img_dir in img_reg_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        for seg_dir in seg_dirs:
            seg_id = seg_dir.split('/')[-1].split('.')[0]
            if img_id == seg_id:
                count += 1
                print(count, img_id)
                img_ids.append(img_id)
                try:
                    crop_top(
                        patient_id=img_id,
                        img_dir=img_dir,
                        seg_dir=seg_dir,
                        crop_shape=crop_shape,
                        return_type='sitk_object',
                        output_img_dir=img_crop_dir,
                        output_seg_dir=seg_crop_dir,
                        image_format=image_format)
                except Exception as e:
                    print(e, 'crop failed!')


def main():

    root_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    #proj_dir = root_dir + '/DFCI/new_curation'
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH'
    tumor_type = 'p_n'
    image_format = 'nrrd'
    crop_shape = (160, 160, 64)
    #crop_shape = (172, 172, 76)
    #step = 'combine_mask'

    #for step in ['combine_mask', 'get_pn_seg', 'interp_reg_crop']:
    for step in ['get_seg_info']:
        if step == 'change_name':
            change_img_name(proj_dir)
        elif step == 'get_seg_info':
            get_seg_info(proj_dir)
        elif step == 'combine_mask':
            for tumor_type in ['p']:
                combine_mask2(proj_dir, tumor_type)
        elif step == 'get_pn_seg':
            get_pn_seg(proj_dir)
        elif step == 'interp_reg_crop':
            interp_reg_crop(proj_dir, root_dir, tumor_type, image_format, crop_shape)
        elif step == 'crop':
            crop(proj_dir, tumor_type, crop_shape, image_format)


if __name__ == '__main__':
    
    main()










