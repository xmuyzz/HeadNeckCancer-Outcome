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


def max_bbox(proj_dir, tumor_type):
    """
    get the max lenths of w, h, d of bbox
    Args:
      tumor_type - required: tumor + node or tumor;
      Cdata_dir  - required: tumor+node label dir CHUM cohort;
    Returns:
        lenths of width, height and depth of max bbox.
    """
    ## get the max lengths of r, c, z
    nnUNet_dir = proj_dir + '/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task502_tot_p_n'
    tr_seg_dir = nnUNet_dir + '/labelsTr'
    ts_seg_dir = nnUNet_dir + '/labelsTs'
    tr_seg_paths = [i for i in glob.glob(tr_seg_dir + '/*.nii.gz')]
    ts_seg_paths = [i for i in glob.glob(ts_seg_dir + '/*.nii.gz')]
    #print(tr_seg_paths)
    seg_paths = tr_seg_paths + ts_seg_paths
    #print(seg_paths)
    
    d_lens = []
    h_lens = []
    w_lens = []
    empty_segs = []
    for i, seg_path in enumerate(seg_paths):
        print(i)
        seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)
        seg_arr = sitk.GetArrayFromImage(seg)
        if tumor_type == 'pn':
            seg_arr = np.where(seg_arr != 0)
        elif tumor_type == 'p':
            seg_arr = np.where(seg_arr == 1)
        elif tumor_type == 'n':
            seg_arr = np.where(seg_arr == 2)
        #print(label_dir.split('/')[-1])
        #print(label_arr.shape)
        if np.any(seg_arr):
            dmin, dmax, hmin, hmax, wmin, wmax = get_bbox_3D(seg_arr)
            d_len = dmax - dmin
            h_len = hmax - hmin
            w_len = wmax - wmin
            d_lens.append(d_len)
            h_lens.append(h_len)
            w_lens.append(w_len)
        else:
            print('empty seg file:', seg_path.split('/')[-1])
            empty_segs.append(seg_path.split('/')[-1])
            continue
    d_max = max(d_lens)
    h_max = max(h_lens)
    w_max = max(w_lens)
    print('d_max:', d_max)
    print('h_max:', h_max)
    print('w_max:', w_max)
    print(empty_segs)

    return d_max, h_max, w_max


def bbox_img(img_path, seg_path, save_img_type, ID, max_bbox, output_dir):
    """
    get cnosistent 3D tumor&node data using masking, bbox and padding
    Args:
      img_dir {path} -- dir for image in nrrd format
      label_dir {path} -- dir for label in nrrd format
    Returns:
        Preprocessed images in nii.gz or np.array formats;
    """
    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img)
    seg = sitk.ReadImage(seg_path)
    seg_arr = sitk.GetArrayFromImage(seg)
 
    # get 3d bounding box
    #--------------------
    dmin, dmax, hmin, hmax, wmin, wmax = get_bbox_3D(seg_arr)

    #dmin, dmax, hmin, hmax, wmin, wmax = [0, 160, 0, 160, 0, 60]
    #print(dmin, dmax, hmin, hmax, wmin, wmax)

    # choose mask_img or bbox_img
    if save_img_type == 'mask_img':
        #apply mask to image
        if tumor_type == 'pn':
            masked_arr = np.where(seg_arr != 0, img_arr, seg_arr)
            #masked_arr = np.where(seg_arr != 0) * img_arr
        elif tumor_type == 'p':
            seg_arr[seg_arr == 2] = 0
            masked_arr = np.where(seg_arr == 1, img_arr, seg_arr)
            #masked_arr = np.where(seg_arr==1) * img_arr
        elif tumor_type == 'n':
            seg_arr[seg_arr == 1] = 0
            masked_arr = np.where(seg_arr == 2, img_arr, seg_arr)
        #masked_arr = norm_img * seg_arr
        img_bbox = masked_arr[dmin:dmax+1, hmin:hmax+1, wmin:wmax+1]
    elif save_img_type == 'attn_img':
        #apply mask to image
        if tumor_type == 'pn':
            seg_arr[seg_arr == 1] = 2
            seg_arr[seg_arr == 2] = 2
            seg_arr[seg_arr == 0] = 1
            attn_arr = seg_arr * img_arr
            MAX, MIN = attn_arr.max(), attn_arr.min()
            attn_arr = (attn_arr - MIN) / (MAX - MIN)
        elif tumor_type == 'p':
            seg_arr[seg_arr == 2] = 0
            masked_arr = np.where(seg_arr == 1, img_arr, seg_arr)
            #masked_arr = np.where(seg_arr==1) * img_arr
        elif tumor_type == 'n':
            seg_arr[seg_arr == 1] = 0
            masked_arr = np.where(seg_arr == 2, img_arr, seg_arr)
        #masked_arr = norm_img * seg_arr
        img_bbox = attn_arr[dmin:dmax+1, hmin:hmax+1, wmin:wmax+1]
    elif save_img_type == 'bbox_img':
        img_bbox = img_arr[dmin:dmax+1, hmin:hmax+1, wmin:wmax+1]
    elif save_img_type == '2channel':
        out_img = np.stack((img_arr, seg_arr), axis=3)
    
    #print('masked_arr:', masked_arr.shape)
    #print('img_bbox:', img_bbox.shape)
    
    # padding to match max bbox
    #--------------------------
    padding = False
    if padding:
        d_pad = max_bbox[0] - img_bbox.shape[0]
        h_pad = max_bbox[1] - img_bbox.shape[1]
        w_pad = max_bbox[2] - img_bbox.shape[2]
        # keep consistent bbox size
        pad_1s = []
        pad_2s = []
        for pad in [d_pad, h_pad, w_pad]:
            assert pad >= 0, print('pad:', pad, img_bbox.shape)
            if pad % 2 == 0:
                pad_1 = pad_2 = pad // 2
            else:
                pad_1 = pad // 2
                pad_2 = pad // 2 + 1
            pad_1s.append(pad_1)
            pad_2s.append(pad_2)
        # numpy padding
        img_pad = np.pad(
            array=img_bbox, 
            pad_width=((pad_1s[0], pad_2s[0]), (pad_1s[1], pad_2s[1]), (pad_1s[2], pad_2s[2])), 
                        mode='constant',
                        constant_values=[(0, 0), (0, 0), (0, 0)])
    #print('img_pad:', img_pad.shape)
    #output_img = sitk.GetImageFromArray(img_pad)
    
    ##### temporary solution:
    ##### use whole input image
    #output_img = sitk.GetImageFromArray(attn_arr)
    output_img = sitk.GetImageFromArray(out_img)
    #img = img_pad.transpose(1, 2, 0)
    #img = np.transpose(img_pad, axes=[1, 2, 0])
    
    fn = output_dir + '/' + ID + '.nii.gz'
    output_img.SetSpacing(img.GetSpacing())
    output_img.SetOrigin(img.GetOrigin())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(fn)
    writer.SetUseCompression(True)
    writer.Execute(output_img)
    
    return output_img


def save_bbox_img(proj_dir, data_set, tumor_type, img_type):
    """ save bbox_img to folders
    """
    print('tumor type:', tumor_type)
    print('data set:', data_set)
    print('save_img_type:', img_type)

    nnUNet_dir = proj_dir + '/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task502_tot_p_n'
    data_dir = proj_dir + '/outcome/data/' + img_type
    
    save_dir = data_dir + '/' + data_set + '_' + tumor_type 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if data_set == 'tr':
        img_dir = nnUNet_dir + '/imagesTr'
        seg_dir = nnUNet_dir + '/labelsTr'
    elif data_set == 'ts_gt':
        img_dir = nnUNet_dir + '/imagesTs'
        seg_dir = nnUNet_dir + '/labelsTs'
    elif data_set == 'ts_pr':
        img_dir = nnUNet_dir + '/imagesTs'
        seg_dir = nnUNet_dir + '/predsTs'
    elif data_set == 'tx1_gt':
        img_dir = nnUNet_dir + '/imagesTx1'
        seg_dir = nnUNet_dir + '/labelsTx1'
    elif data_set == 'tx1_pr':
        img_dir = nnUNet_dir + '/imagesTx1'
        seg_dir = nnUNet_dir + '/predsTx1'
    elif data_set == 'tx2_gt':
        img_dir = nnUNet_dir + '/imagesTx2'
        seg_dir = nnUNet_dir + '/labelsTx2'
    elif data_set == 'tx2_pr':
        img_dir = nnUNet_dir + '/imagesTx2'
        seg_dir = nnUNet_dir + '/predsTx2'

    ids = []
    for i, seg_path in enumerate(glob.glob(seg_dir + '/*nii.gz')):
        id = seg_path.split('/')[-1].split('.')[0]
        print(i, id)
        img_path = img_dir + '/' + id + '_0000.nii.gz'
        try:
            bbox_img(img_path, seg_path, save_img_type=img_type, ID=id, 
                max_bbox=(60, 160, 160), output_dir=save_dir)
        except Exception as e:
            print(id, e)
            ids.append(id)
    print('bad data:', ids)


if __name__ == '__main__':

    proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck'
    img_type = '2channel'   #'attn_img' # bbox_img, mask_img

    #max_bbox(proj_dir, tumor_type)
    for tumor_type in ['pn']:
        for data_set in ['tr', 'ts_pr', 'ts_gt']:
            save_bbox_img(proj_dir, data_set, tumor_type, img_type)
    

