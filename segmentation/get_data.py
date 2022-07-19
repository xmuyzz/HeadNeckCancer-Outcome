import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import glob



def get_data(dfci_img_dir, dfci_seg_dir, img_type, dataset): 
    """
    Used for testing only. The image sitk object info is needed during test time. 
    To avoid reading the image nrrd twice, it is read here.
    """
    count = 0
    data = []
    img_dirs = [i for i in sorted(glob.glob(dfci_img_dir + "/*nrrd"))]
    seg_dirs = [i for i in sorted(glob.glob(dfci_seg_dir + "/*nrrd"))]
    for img_dir, seg_dir in zip(img_dirs, seg_dirs):
        count += 1
        print(count)
        pat_id = img_dir.split("/")[-1].split(".")[0]
        img_sitk_obj = sitk.ReadImage(img_dir)
        arr_img = sitk.GetArrayFromImage(img_sitk_obj)
        arr_img_interp = np.interp(arr_img, [-1024, 3071], [0, 1])
        # get label
        seg_sitk_obj = sitk.ReadImage(seg_dir)
        arr_seg = sitk.GetArrayFromImage(seg_sitk_obj)
        # assertions
        assert arr_img.shape == arr_seg.shape, "image and label do not have the same shape."
        assert arr_seg.min() == 0, "label min is not 0 @ {}_{}".format(pat_id, dataset)
        assert arr_seg.max() == 1, "label max is not 1 @ {}_{}".format(pat_id, dataset)
        assert len(np.unique(arr_seg))==2, "length of label unique vals is not 2 @ {}_{}".format(pat_id, dataset)
        # append to list
        if img_type == "ct":
            data.append({
                "patient_id": pat_id,
                "dataset": dataset,
                "image_sitk_obj": img_sitk_obj,
                "image": arr_img_interp,
                "seg_sitk_obj": seg_sitk_obj})
        print ("{}".format(pat_id))
    return data







