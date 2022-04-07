import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle
from time import gmtime, strftime
from datetime import datetime
import timeit
import yaml
import argparse
from get_data.input_arr import input_arr
from get_data.label import label
from get_data.img_label_df import img_label_df
from get_data.split_dataset import split_dataset
from get_data.get_dir import get_dir


if __name__ == '__main__':

    data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated'
    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    out_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    clinical_data_file = 'HN_clinical_meta_data.csv'
    save_label = 'label.csv'
    norm_type = 'np_clip'
    tumor_type = 'primary_node'
    input_img_type = 'masked_img'
    input_channel = 1
    save_img_type = 'nii'
    new_spacing = (2, 2, 2)
    run_max_bbox = False
    run_data = True
    split_data_only = False
    run_getdir = True
    run_label = False

    for tumor_type in ['primary_node']:
        for input_img_type in ['raw_img']:
            if split_data_only:
                split_dataset(
                    proj_dir=proj_dir,
                    tumor_type=tumor_type,
                    input_img_type=input_img_type)
            else:
                ## get img, seg dir
                if run_data:
                    if run_getdir:
                        get_dir(
                            data_dir=data_dir,
                            proj_dir=proj_dir,
                            tumor_type=tumor_type)
                    input_arr(
                        data_dir=data_dir, 
                        proj_dir=proj_dir,
                        new_spacing=new_spacing,
                        norm_type=norm_type, 
                        tumor_type=tumor_type, 
                        input_img_type=input_img_type,
                        input_channel=input_channel,
                        run_max_bbox=run_max_bbox,
                        save_img_type=save_img_type)
                if run_label: 
                    label(
                        proj_dir=proj_dir,
                        clinical_data_file=clinical_data_file, 
                        save_label=save_label)

                img_label_df(
                    proj_dir=proj_dir,
                    tumor_type=tumor_type,
                    input_img_type=input_img_type,
                    save_img_type=save_img_type)
                
                split_dataset(
                    proj_dir=proj_dir,
                    tumor_type=tumor_type,
                    input_img_type=input_img_type)







