import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold, train_test_split


def split_dataset(proj_dir, tumor_type, input_img_type):

    if tumor_type == 'pn':
        if input_img_type == 'masked_img':
            df_fn = 'df_img_label_pn_masked.csv'
            fn_tr = 'df_pn_masked_tr.csv'
            fn_va = 'df_pn_masked_va.csv'
            #fn_ts = 'df_pn_masked_ts.csv'
        elif input_img_type == 'raw_img':
            df_fn = 'df_img_label_pn_raw.csv'
            fn_tr = 'df_pn_raw_tr.csv'
            fn_va = 'df_pn_raw_va.csv'
            #fn_ts = 'df_pn_raw_ts.csv'
    if tumor_type == 'p':
        if input_img_type == 'masked_img':
            df_fn = 'df_img_label_p_masked.csv'
            fn_tr = 'df_p_masked_tr.csv'
            fn_va = 'df_p_masked_va.csv'
            #fn_ts = 'df_p_masked_ts.csv'
        elif input_img_type == 'raw_img':
            df_fn = 'df_img_label_p_raw.csv'
            fn_tr = 'df_p_raw_tr.csv'
            fn_va = 'df_p_raw_va.csv'
            #fn_ts = 'df_p_raw_ts.csv'
    if tumor_type == 'n':
        if input_img_type == 'masked_img':
            df_fn = 'df_img_label_n_masked.csv'
            fn_tr = 'df_n_masked_tr.csv'
            fn_va = 'df_n_masked_va.csv'
            #fn_ts = 'df_n_masked_ts.csv'
        elif input_img_type == 'raw_img':
            df_fn = 'df_img_label_n_raw.csv'
            fn_tr = 'df_n_raw_tr.csv'
            fn_va = 'df_n_raw_va.csv'
            #fn_ts = 'df_n_raw_ts.csv'
    df = pd.read_csv(prep_data_dir + '/prep_data/' + df_fn)

    df_tr, df_va = train_test_split(df, test_size=0.1)
    print('train data shape:', df_tr.shape)
    print('val data shape:', df_va.shape)
    #print('test data shape:', df_ts.shape)
    df_tr.to_csv(pro_data_dir + '/' + fn_tr, index=False)
    df_va.to_csv(pro_data_dir + '/' + fn_va, index=False)
    #df_ts.to_csv(pro_data_dir + '/' + fn_ts, index=False)
    print('train, val and test dfs have been saved!!!')