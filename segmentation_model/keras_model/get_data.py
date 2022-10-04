import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import glob



def get_data(test_img_dir, test_seg_dir): 
    """
    Used for testing only. The image sitk object info is needed during test time. 
    To avoid reading the image nrrd twice, it is read here.
    """
    print('load test data...')
    count = 0
    data = []
    img_dirs = [i for i in sorted(glob.glob(test_img_dir + "/*nrrd"))]
    seg_dirs = [i for i in sorted(glob.glob(test_seg_dir + "/*nrrd"))]
    for seg_dir in seg_dirs:
        seg_id = seg_dir.split("/")[-1].split(".")[0]
        for img_dir in img_dirs:
            img_id = img_dir.split("/")[-1].split(".")[0]
            if seg_id == img_id:
                count += 1
                print(count, img_id)
                pat_id = img_dir.split("/")[-1].split(".")[0]
                img_sitk_obj = sitk.ReadImage(img_dir)
                arr_img = sitk.GetArrayFromImage(img_sitk_obj)
                arr_img_interp = np.interp(arr_img, [-1024, 3071], [0, 1])
                # get label
                seg_sitk_obj = sitk.ReadImage(seg_dir)
                arr_seg = sitk.GetArrayFromImage(seg_sitk_obj)
                # assertions
                assert arr_img.shape == arr_seg.shape, "image and label do not have the same shape."
                assert arr_seg.min() == 0, "label min is not 0 @ {}_{}".format(pat_id)
                assert arr_seg.max() == 1, "label max is not 1 @ {}_{}".format(pat_id)
                assert len(np.unique(arr_seg))==2, "length of label unique vals is not 2 @ {}_{}".format(pat_id)
                # append to list
                data.append({
                    "patient_id": pat_id,
                    "image_sitk_obj": img_sitk_obj,
                    "image": arr_img_interp,
                    "seg_sitk_obj": seg_sitk_obj})
            #print ("{}".format(pat_id))
    return data







