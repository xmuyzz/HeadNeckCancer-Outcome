import os
import numpy as np
import pandas as pd
from data_prepro import data_prepro
from model_pred import model_pred


if __name__ == '__main__':
 
    data_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/DFCI/new_curation/img_reg'
    model_dir = '/home/xmuyzz/Harvard_AIM/HNCancer/contrast_detection/saved_models'
    output_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task505_PN/output'
    print('\n--- MODEL INFERENCE ---\n')
    
    # data preprocessing
    df_img, img_arr = data_prepro(body_part='HeadNeck', data_dir=data_dir)

    # model prediction
    model_pred(
        body_part='HeadNeck',
        save_csv=True,
        model_dir=model_dir,
        out_dir=output_dir,
        df_img=df_img,
        img_arr=img_arr)
    
    print('Model prediction done!')
