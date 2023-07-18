import numpy as np
import pandas as pd
import os
import glob
import SimpleITK as sitk
from rtstruct_to_nrrd import rtstruct_to_nrrd
from combine_structures import combine_structures
from interpolate import interpolate
from crop_image import crop_top, crop_top_image_only, crop_full_body
from registration import nrrd_reg_rigid

def get_gtv_info():
    proj_dir = '/mnt/kannlab_rfa/Frank/Data/Autosegm'
    rtstruct_dir = proj_dir + '/Maastro/RTSTRUCT_02_ShortID'
    df = pd.read_csv(proj_dir + '/Maastro_RT_Struct_overview_merge.csv')
    IDs = []
    datess = []
    for i, ID in enumerate(os.listdir(rtstruct_dir)):
        if not ID.startswith('.'):
            print(i, ID)
            IDs.append(ID)
            dates = []
            for date in os.listdir(rtstruct_dir + '/' + ID + '/RTSTRUCT'):
                if not date.startswith('.'):
                    dates.append(date)
                    print(dates)
            datess.append(dates)
    df1 = pd.DataFrame({'MUMC2021 OPC.Patient Study ID': IDs, 'CT dates': datess})
    print(df1)
    df = df.merge(df1, how='left', on='MUMC2021 OPC.Patient Study ID').reset_index()
    df.to_csv(proj_dir + '/Maastro_sum.csv')


def get_raw_gtv():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/MAASTRO'
    raw_img_dir = proj_dir + '/raw_img'
    rtstruct_dir = proj_dir + '/rtstruct'
    raw_gtv_dir = proj_dir + '/raw_gtv'
    IDs = []
    datess = []
    for i, ID in enumerate(os.listdir(rtstruct_dir)):
        if not os.path.exists(raw_gtv_dir + '/' + ID):
            if not ID.startswith('.'):
                print(i, ID)
                raw_img_path = raw_img_dir + '/Maastro_' + ID + '_CT_raw_raw_raw_xx.nrrd'
                if not os.path.exists(raw_img_path):
                    print('CT not available:', ID)
                else:
                    for i, rt_dir in enumerate(glob.glob(rtstruct_dir + '/' + ID + '/RTSTRUCT/*/*.dcm')):
                        rtstruct_to_nrrd(
                            patient_id=ID,
                            path_to_rtstruct=rt_dir,
                            path_to_image=raw_img_path,
                            output_dir=raw_gtv_dir)

def get_raw_gtv2():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/MAASTRO'
    raw_img_dir = proj_dir + '/raw_img'
    rtstruct_dir = proj_dir + '/rtstruct'
    raw_gtv_dir = proj_dir + '/raw_gtv'
    IDs = []
    datess = []
    for i, ID in enumerate(os.listdir(rtstruct_dir)):
        #if not os.path.exists(raw_gtv_dir + '/' + ID):
        if ID in ['MUMC041', 'MUMC060', 'MUMC067']:
            if not ID.startswith('.'):
                print(i, ID)
                raw_img_path = raw_img_dir + '/Maastro_' + ID + '_CT_raw_raw_raw_xx.nrrd'
                if not os.path.exists(raw_img_path):
                    print('CT not available:', ID)
                else:
                    for j, rt_dir in enumerate(glob.glob(rtstruct_dir + '/' + ID + '/RTSTRUCT/*/*.dcm')):
                        print(j)
                        output_dir = rtstruct_dir + '/' + ID + '/' + str(j)
                        rtstruct_to_nrrd(
                            patient_id=ID,
                            path_to_rtstruct=rt_dir,
                            path_to_image=raw_img_path,
                            output_dir=output_dir)

def get_gtv_names():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/MAASTRO'
    raw_img_dir = proj_dir + '/raw_img'
    rtstruct_dir = proj_dir + '/rtstruct'
    raw_gtv_dir = proj_dir + '/raw_gtv'
    IDs = []
    GTVss = []
    volss = []
    excludes = ['SUV', 'PTV', 'NS', 'PET', 'pet', 'HR', 'sum', 'FDG', 'cr', 'CTV']
    for i, ID in enumerate(os.listdir(raw_gtv_dir)):
        if not ID.startswith('.'):
            print(i, ID)
            IDs.append(ID)
            GTVs = []
            vols = []
            for data_dir in glob.glob(raw_gtv_dir + '/' + ID + '/*nrrd'):
                name = data_dir.split('/')[-1].split('.')[0]
                if 'GTV' in name and not any(x in name for x in excludes):
                    GTVs.append(name)
                    img = sitk.ReadImage(data_dir)
                    space = img.GetSpacing()
                    voxel= np.prod(space)
                    arr = sitk.GetArrayFromImage(img)
                    vol = voxel * np.sum(arr)
                    vol = int(vol)
                    print(vol)
                    vols.append(vol)
            GTVss.append(GTVs)
            volss.append(vols)
    df = pd.DataFrame({'ID': IDs, 'GTV': GTVss, 'Volume': volss})
    df0 = pd.read_csv(proj_dir + '/Maastro_sum_FH.csv', index_col=0)
    df0.drop(['GTV'], axis=1, inplace=True)
    df = df0.merge(df, on='ID', how='left').reset_index()
    df = df[df['GTV'].notna()]
    ids = []
    for ID, vol in zip(df['ID'], df['Volume']):
        if not vol or 0 in vol:
            ids.append(ID)
    df = df[~df['ID'].isin(ids)]
    df.drop(['index'], axis=1, inplace=True)
    # get GTV_P and GTV_N
    ps = ['GTVp1', 'GTV-1', 'GTVp_1', 'GTVp1b']
    gtv_pss = []
    gtv_nss = []
    for gtvs in df['GTV']:
        gtv_ps = []
        gtv_ns = []
        for gtv in gtvs:
            if gtv in ps:
                gtv_ps.append(gtv)
            else:
                gtv_ns.append(gtv)
        gtv_pss.append(gtv_ps)
        gtv_nss.append(gtv_ns)
    df['GTV_p'], df['GTV_n'] = gtv_pss, gtv_nss
    df = df[['ID', 'CT dates', 'GTV_p', 'GTV_n', 'GTV', 'Volume']]
    df.to_csv(proj_dir + '/Maastro_GTV.csv', index=False)


def combine_mask(tumor_type):
    """
    COMBINING MASKS 
    """
    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/MAASTRO'
    raw_img_dir = proj_dir + '/raw_img'
    raw_gtv_dir = proj_dir + '/raw_gtv'
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
    df = pd.read_csv(proj_dir + '/clinical/Maastro_GTV.csv', converters={tumor_type: pd.eval})
    i = 0
    bad_data = []
    for ID, gtvs in zip(df['ID'], df[tumor_type]):
        gtv_dirs = []
        #if ID == 'MUMC017':
        i += 1
        print(i, ID)
        img_dir = raw_img_dir + '/Maastro_' + ID + '_CT_raw_raw_raw_xx.nrrd'
        for gtv in gtvs:
            #print(img_dir)
            gtv_dir = raw_gtv_dir + '/' + ID + '/' + gtv + '.nrrd'
            gtv_dirs.append(gtv_dir)
            #print(gtv_dirs)
            try:
                combined_mask = combine_structures(
                    patient_id=ID,
                    mask_arr=gtv_dirs,
                    path_to_reference_image_nrrd=img_dir,
                    binary=2,
                    return_type='sitk_object',
                    output_dir=seg_save_dir)
                print('combine successfully')
            except Exception as e:
                print(ID, e)
                bad_data.append(ID)
    print('bad data:', bad_data)


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
    p_seg_dir = proj_dir + '/raw_seg_p'
    n_seg_dir = proj_dir + '/raw_seg_n'
    pn_seg_dir = proj_dir + '/raw_seg_pn'
    p_n_seg_dir = proj_dir + '/raw_seg_p_n'
    raw_img_dir = proj_dir + '/raw_img'
    if not os.path.exists(p_n_seg_dir):
        os.makedirs(p_n_seg_dir)
    fns = [i for i in sorted(os.listdir(pn_seg_dir))]
    for i, fn in enumerate(fns):
        try:
            pat_id = fn.split('.')[0]
            print(i, pat_id)
            # image
            img_dir = raw_img_dir + '/Maastro_' + pat_id + '_CT_raw_raw_raw_xx.nrrd'
            #img_dir = img_path + '/' + fn
            img = sitk.ReadImage(img_dir)
            arr = sitk.GetArrayFromImage(img)
            # primary tumor
            p_seg_path = p_seg_dir + '/' + fn
            if os.path.exists(p_seg_path):
                p_seg = sitk.ReadImage(p_seg_path)
                p_seg = sitk.GetArrayFromImage(p_seg)
                p_seg[p_seg != 0] = 1
                #print('p_seg shape:', p_seg.shape)
            else:
                print('no primary segmentation...')
                p_seg = np.zeros(shape=arr.shape)
            # node
            n_seg_path = n_seg_dir + '/' + fn
            if os.path.exists(n_seg_path):
                n_seg = sitk.ReadImage(n_seg_path)
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
            # some voxels from P and N have overlap
            p_n_seg[p_n_seg == 3] = 1
            # clear overlap with airway by setting threshold -800 HU
            img_seg = arr * p_n_seg
            p_n_seg[img_seg <= -500] = 0
            sitk_obj = sitk.GetImageFromArray(p_n_seg)
            sitk_obj.SetSpacing(img.GetSpacing())
            sitk_obj.SetOrigin(img.GetOrigin())
            # write new nrrd
            writer = sitk.ImageFileWriter()
            writer.SetFileName(p_n_seg_dir + '/' + pat_id + '.nrrd')
            writer.SetUseCompression(True)
            writer.Execute(sitk_obj)
        except Exception as e:
            print(pat_id, e)


def interp_reg_crop(proj_dir, tumor_type, image_format, crop_shape):
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
        seg_raw_dir = seg_p_n_raw_dir
        seg_crop_dir = seg_p_n_crop_dir
    elif tumor_type == 'pn':
        seg_dirs = seg_pn_dirs
        seg_raw_dir = seg_pn_raw_dir
        seg_crop_dir = seg_pn_crop_dir
    elif tumor_type == 'p':
        seg_dirs = seg_p_dirs
        seg_raw_dir = seg_p_raw_dir
        seg_crop_dir = seg_p_crop_dir
    elif tumor_type == 'n':
        seg_dirs = seg_n_dirs
        seg_raw_dir = seg_n_raw_dir
        seg_crop_dir = seg_n_crop_dir
    img_ids = []
    bad_ids = []
    bad_scans = []
    Es = []
    count = 0
    # get register templatei
    root_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    fixed_img_dir = root_dir + '/DFCI/img_interp/10020741814.nrrd'
    fixed_img = sitk.ReadImage(fixed_img_dir, sitk.sitkFloat32)
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('_')[1]
        if img_id == 'MUMC143':
            seg_dir = seg_raw_dir + '/' + img_id + '.nrrd'
            count += 1
            print(count, img_id)
            # load img and seg
            if not os.path.exists(seg_dir):
                print('seg does not exist:', img_id)
            else:
                img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
                seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
                # --- crop full body scan ---
                z_img = img.GetSize()[2]
                z_seg = seg.GetSize()[2]
                if z_img < 105:
                    print('This is an incomplete scan!')
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
                        bad_scans.append(img_id)
                        Es.append(e)
                        print(img_id, e)
    print('bad scans:', bad_scans, Es)


if __name__ == '__main__':

    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/MAASTRO'
    tumor_type = 'p_n'
    image_format = 'nrrd'
    crop_shape = (160, 160, 64)
    #crop_shape = (172, 172, 76)
    
    #get_gtv_names()
    #get_raw_gtv2()
    #combine_mask(tumor_type)
    #get_pn_seg(proj_dir)
    interp_reg_crop(proj_dir, tumor_type, image_format, crop_shape)

            


