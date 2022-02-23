"""
  ----------------------------------------------
  DeepContrast - run DeepContrast pipeline step1
  ----------------------------------------------
  ----------------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.6.8
  ----------------------------------------------
  
  Deep-learning-based IV contrast detection
  in CT scans - all param.s are read
  from a config file stored under "/config"
  
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import yaml
import argparse
from get_data.input_arr import input_arr
from get_data.label import label
from get_data.img_label_df import img_label_df
from get_data.split_dataset import split_dataset
from get_data.max_bbox import max_bbox
from get_data.get_dir import get_dir


if __name__ == '__main__':

    data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated'
    proj_dir = '/mnt/HDD_6TB/HN_Outcome'
    data_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    clinical_data_file = 'HN_clinical_meta_data.csv'
    save_label = 'label.csv'
    norm_type = 'np_clip'
    tumor_type = 'primary_node'
    input_type = 'masked_img'
    input_channel = 1
    save_img_type = 'nii'
    run_max_bbox = False
    run_data = False

    ## get img, seg dir
    img_dirss, seg_pn_dirss, exclude_pn, exclude_p = get_dir(
        data_dir=data_dir,
        tumor_type=tumor_type
        )

    if run_data:
        input_arr(
            data_dir=data_dir, 
            proj_dir=proj_dir,
            aimlab_dir=aimlab_dir,
            norm_type=norm_type, 
            tumor_type=tumor_type, 
            input_type=input_type,
            input_channel=input_channel,
            run_max_bbox=run_max_bbox,
            img_dirss=img_dirss,
            seg_pn_dirss=seg_pn_dirss,
            toyset=toyset,
            save_img_type=save_img_type
            )

    label(
        proj_dir=proj_dir,
        clinical_data_file=clinical_data_file, 
        save_label=save_label
        )

    img_label_df(
        proj_dir=proj_dir,
        aimlab_dir=aimlab_dir,
        save_img_type=save_img_type
        )

    split_dataset(proj_dir=proj_dir)




