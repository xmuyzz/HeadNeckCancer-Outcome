import pickle
import os
import numpy as np
import json
import nibabel as nib
import SimpleITK as sitki
import SimpleITK as sitk
from bbox import get_bbox_3D
from respacing import respacing

if __name__ == '__main__':

    out_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/test'
    data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated/CHUM_files'
    img_dir = os.path.join(data_dir, 'image_reg/HN-CHUM-001_registered.nrrd')
    seg_dir = os.path.join(data_dir, 'label_reg/HN-CHUM-001_label_registered.nrrd')
    

    """repacing
    """
    do_respacing = True
    if do_respacing:
        # respacing for image
        img_arr = respacing(
            nrrd_dir=img_dir,
            interp_type='linear',
            new_spacing=[2, 2, 2],
            patient_id=None,
            return_type='npy',
            save_dir=None
            )
        seg_arr = respacing(
            nrrd_dir=seg_dir,
            interp_type='nearest_neighbor',
            new_spacing=[2, 2, 2],
            patient_id=None,
            return_type='npy',
            save_dir=None
            )
    else:
        img_nrrd = sitk.ReadImage(img_dir)
        img_arr = sitk.GetArrayFromImage(img_nrrd)
        seg_nrrd = sitk.ReadImage(seg_dir)
        seg_arr = sitk.GetArrayFromImage(seg_nrrd)


    """normalize CT image
    """
    norm_type = 'np_clip'
    data = img_arr
    data[data <= -1024] = -1024
    # strip skull, skull UHI = ~700
    #data[data > 700] = 0
    # normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
    if norm_type == 'np_interp':
        norm_img = np.interp(data, [-200, 200], [0, 1])
    elif norm_type == 'np_clip':
        data = np.clip(data, a_min=-200, a_max=200)
        MAX, MIN = data.max(), data.min()
        norm_img = (data - MIN) / (MAX - MIN)

    """apply mask to image
    """
    arr = np.where(seg_arr==1, norm_img, seg_arr)
    #print('masked_arr:', masked_arr[:, :, 27])
    #print('seg_arr:', seg_arr[:, :, 27])
    #arr = norm_img * seg_arr
    #print('img arr:', img_arr[:, :, 68])

    """get 3d bounding box
    """
    dmin, dmax, hmin, hmax, wmin, wmax = get_bbox_3D(arr)
    print(arr[dmin:dmax, :, :].shape)
    print(arr[:, hmin:hmax, :].shape)
    print(arr[:, :, wmin:wmax].shape)
    img_bbox = arr[dmin:dmax+1, hmin:hmax+1, wmin:wmax+1]
    print('img_bbox:', img_bbox.shape)

    """padding
    """
    padding = True
    if padding:
        d_max = 100
        h_max = 100
        w_max = 100
        arr = img_bbox
        d_pad = d_max - arr.shape[0]
        h_pad = h_max - arr.shape[1]
        w_pad = w_max - arr.shape[2]
        # keep consistent bbox size
        pad_1s = []
        pad_2s = []
        for pad in [d_pad, h_pad, w_pad]:
            assert pad >= 0, print('pad:', pad, arr.shape)
            if pad % 2 == 0:
                pad_1 = pad_2 = pad // 2
            else:
                pad_1 = pad // 2
                pad_2 = pad // 2 + 1
            pad_1s.append(pad_1)
            pad_2s.append(pad_2)
        # numpy padding
        img_pad = np.pad(
            array=arr,
            pad_width=((pad_1s[0], pad_2s[0]),
                       (pad_1s[1], pad_2s[1]),
                       (pad_1s[2], pad_2s[2])),
            mode='constant',
            constant_values=[(0, 0), (0, 0), (0, 0)]
            )
        print('img_pad:', img_pad.shape)
        img = img_pad
        #img = img_pad.transpose(1, 2, 0)
        #img = np.transpose(img_pad, axes=[1, 2, 0])
    else:
        img = img_bbox
    print('img:', img.shape)


    fn = 'seg_test.nii.gz'
    img = nib.Nifti1Image(img, affine=np.eye(4))
    nib.save(img, os.path.join(out_dir, fn))




