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
from opts import parse_opts


def main(opt):

    run_max_bbox = False
    run_data = False
    split_data_only = False
    run_getdir = True
    run_label = True
    run_get_dataset = False

    for tumor_type in ['primary_node']:
        for input_img_type in ['raw_img']:
            if split_data_only:
                split_dataset(
                    proj_dir=opt.proj_dir,
                    tumor_type=opt.tumor_type,
                    input_img_type=opt.input_data_type)
            else:
                ## get img, seg dir
                if run_data:
                    if run_getdir:
                        get_dir(
                            data_dir=opt.data_dir,
                            proj_dir=opt.proj_dir,
                            tumor_type=opt.tumor_type)
                    input_arr(
                        data_dir=opt.data_dir, 
                        proj_dir=opt.proj_dir,
                        new_spacing=opt.new_spacing,
                        norm_type=opt.norm_type, 
                        tumor_type=opt.tumor_type, 
                        input_img_type=opt.input_img_type,
                        input_channel=iopt.nput_channel,
                        run_max_bbox=run_max_bbox,
                        save_img_type=opt.save_data_type)
                if run_label: 
                    label(
                        proj_dir=opt.proj_dir,
                        clinical_data_file=opt.clinical_data_file, 
                        save_label=opt.save_label)
                if run_get_dataset:
                    img_label_df(
                        proj_dir=opt.proj_dir,
                        tumor_type=opt.tumor_type,
                        input_img_type=opt.input_data_type,
                        save_img_type=opt.save_img_type)
                    split_dataset(
                        proj_dir=opt.proj_dir,
                        tumor_type=opt.tumor_type,
                        input_img_type=opt.input_data_type)


if __name__ == '__main__':

    opt = parse_opts()

    main(opt)





