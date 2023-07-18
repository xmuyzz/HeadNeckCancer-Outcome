import os
import pandas as pd
import numpy as np


def get_dataset(tumor_type, input_data_type):
    if tumor_type == 'primary_node':
        if input_img_type == 'masked_img':
            df_fn = 'df_img_label_pn_masked.csv'
            fn_tr = 'df_pn_masked_tr.csv'
            fn_va = 'df_pn_masked_va.csv'
            fn_ts = 'df_pn_masked_ts.csv'
        elif input_img_type == 'raw_img':
            df_fn = 'df_img_label_pn_raw.csv'
            fn_tr = 'df_pn_raw_tr.csv'
            fn_va = 'df_pn_raw_va.csv'
            fn_ts = 'df_pn_raw_ts.csv'
    if tumor_type == 'primary':
        if input_img_type == 'masked_img':
            df_fn = 'df_img_label_p_masked.csv'
            fn_tr = 'df_p_masked_tr.csv'
            fn_va = 'df_p_masked_va.csv'
            fn_ts = 'df_p_masked_ts.csv'
        elif input_img_type == 'raw_img':
            df_fn = 'df_img_label_p_raw.csv'
            fn_tr = 'df_p_raw_tr.csv'
            fn_va = 'df_p_raw_va.csv'
            fn_ts = 'df_p_raw_ts.csv'
    if tumor_type == 'node':
        if input_img_type == 'masked_img':
            df_fn = 'df_img_label_n_masked.csv'
            fn_tr = 'df_n_masked_tr.csv'
            fn_va = 'df_n_masked_va.csv'
            fn_ts = 'df_n_masked_ts.csv'
            fn_ts = 'df_n_masked_test.csv'
        elif input_img_type == 'raw_img':
            df_fn = 'df_img_label_n_raw.csv'
            fn_tr = 'df_n_raw_tr.csv'
            fn_va = 'df_n_raw_va.csv'
            fn_ts = 'df_n_raw_ts.csv'
            fn_ta = 'df_n_raw_test.csv'

    return fn_tr, fn_va, fn_ts







