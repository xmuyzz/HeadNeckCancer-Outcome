import os
import pandas as pd
import numpy as np



def get_dataset(tumor_type, input_data_type):
    if tumor_type == 'primary_node':
        if input_data_type == 'masked_img':
            fns_train = [
                'df_pn_masked_train0.csv',
                'df_pn_masked_train1.csv',
                'df_pn_masked_train2.csv',
                'df_pn_masked_train3.csv',
                'df_pn_masked_train4.csv']
            fns_val = [
                'df_pn_masked_val0.csv',
                'df_pn_masked_val1.csv',
                'df_pn_masked_val2.csv',
                'df_pn_masked_val3.csv',
                'df_pn_masked_val4.csv']
            fn_test = 'df_pn_masked_test.csv'
        elif input_data_type == 'raw_img':
            fns_train = [
                'df_pn_raw_train0.csv',
                'df_pn_raw_train1.csv',
                'df_pn_raw_train2.csv',
                'df_pn_raw_train3.csv',
                'df_pn_raw_train4.csv']
            fns_val = [
                'df_pn_raw_val0.csv',
                'df_pn_raw_val1.csv',
                'df_pn_raw_val2.csv',
                'df_pn_raw_val3.csv',
                'df_pn_raw_val4.csv']
            fn_test = 'df_pn_raw_test.csv'
    if tumor_type == 'primary':
        if input_data_type == 'masked_img':
            fns_train = [
                'df_p_masked_train0.csv',
                'df_p_masked_train1.csv',
                'df_p_masked_train2.csv',
                'df_p_masked_train3.csv',
                'df_p_masked_train4.csv']
            fns_val = [
                'df_p_masked_val0.csv',
                'df_p_masked_val1.csv',
                'df_p_masked_val2.csv',
                'df_p_masked_val3.csv',
                'df_p_masked_val4.csv']
            fn_test = 'df_p_maksed_test.csv'
        elif input_data_type == 'raw_img':
            fns_train = [
                'df_p_raw_train0.csv',
                'df_p_raw_train1.csv',
                'df_p_raw_train2.csv',
                'df_p_raw_train3.csv',
                'df_p_raw_train4.csv']
            fns_val = [
                'df_p_raw_val0.csv',
                'df_p_raw_val1.csv',
                'df_p_raw_val2.csv',
                'df_p_raw_val3.csv',
                'df_p_raw_val4.csv']
            fn_test = 'df_p_raw_test.csv'
    if tumor_type == 'node':
        if input_data_type == 'masked_img':
            fns_train = [
                'df_n_masked_train0.csv',
                'df_n_masked_train1.csv',
                'df_n_masked_train2.csv',
                'df_n_masked_train3.csv',
                'df_n_masked_train4.csv']
            fns_val = [
                'df_n_masked_val0.csv',
                'df_n_masked_val1.csv',
                'df_n_masked_val2.csv',
                'df_n_masked_val3.csv',
                'df_n_masked_val4.csv']
            fn_test = 'df_n_masked_test.csv'
        elif input_data_type == 'raw_img':
            fns_train = [
                'df_n_raw_train0.csv',
                'df_n_raw_train1.csv',
                'df_n_raw_train2.csv',
                'df_n_raw_train3.csv',
                'df_n_raw_train4.csv']
            fns_val = [
                'df_n_raw_val0.csv',
                'df_n_raw_val1.csv',
                'df_n_raw_val2.csv',
                'df_n_raw_val3.csv',
                'df_n_raw_val4.csv']
            fn_test = 'df_n_raw_test.csv'

    return fns_train, fns_val, fn_test




