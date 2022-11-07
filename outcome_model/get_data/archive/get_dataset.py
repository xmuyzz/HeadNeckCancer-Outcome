import os
import pandas as pd
import numpy as np


def get_dataset(tumor_type, input_data_type):
    if tumor_type == 'pn':
        if input_data_type == 'mask_img':
            fns_train = [
                'pn_mask_tr0.csv',
                'pn_mask_tr1.csv',
                'pn_mask_tr2.csv',
                'pn_mask_tr3.csv',
                'pn_mask_tr4.csv']
            fns_val = [
                'pn_mask_val0.csv',
                'pn_mask_val1.csv',
                'pn_mask_val2.csv',
                'pn_mask_val3.csv',
                'pn_mask_val4.csv']
            fn_test = 'pn_mask_ts.csv'
        elif input_data_type == 'bbox_img':
            fns_train = [
                'pn_bbox_tr0.csv',
                'pn_bbox_tr1.csv',
                'pn_bbox_tr2.csv',
                'pn_bbox_tr3.csv',
                'pn_bbox_tr4.csv']
            fns_val = [
                'pn_bbox_val0.csv',
                'pn_bbox_val1.csv',
                'pn_bbox_val2.csv',
                'pn_bbox_val3.csv',
                'pn_bbox_val4.csv']
            fn_test = 'pn_bbox_ts.csv'
    if tumor_type == 'p':
        if input_data_type == 'mask_img':
            fns_train = [
                'p_mask_tr0.csv',
                'p_mask_tr1.csv',
                'p_mask_tr2.csv',
                'p_mask_tr3.csv',
                'p_mask_tr4.csv']
            fns_val = [
                'p_mask_val0.csv',
                'p_mask_val1.csv',
                'p_mask_val2.csv',
                'p_mask_val3.csv',
                'p_mask_val4.csv']
            fn_test = 'p_mask_ts.csv'
        elif input_data_type == 'bbox_img':
            fns_train = [
                'p_bbox_tr0.csv',
                'p_bbox_tr1.csv',
                'p_bbox_tr2.csv',
                'p_bbox_tr3.csv',
                'p_bbox_tr4.csv']
            fns_val = [
                'p_bbox_val0.csv',
                'p_bbox_val1.csv',
                'p_bbox_val2.csv',
                'p_bbox_val3.csv',
                'p_bbox_val4.csv']
            fn_test = 'p_bbox_ts.csv'
    if tumor_type == 'n':
        if input_data_type == 'mask_img':
            fns_train = [
                'n_mask_tr0.csv',
                'n_mask_tr1.csv',
                'n_mask_tr2.csv',
                'n_mask_tr3.csv',
                'n_mask_tr4.csv']
            fns_val = [
                'n_mask_val0.csv',
                'n_mask_val1.csv',
                'n_mask_val2.csv',
                'n_mask_val3.csv',
                'n_mask_val4.csv']
            fn_test = 'n_mask_ts.csv'
        elif input_data_type == 'bbox_img':
            fns_train = [
                'n_bbox_tr0.csv',
                'n_bbox_tr1.csv',
                'n_bbox_tr2.csv',
                'n_bbox_tr3.csv',
                'n_bbox_tr4.csv']
            fns_val = [
                'n_bbox_val0.csv',
                'n_bbox_val1.csv',
                'n_bbox_val2.csv',
                'n_bbox_val3.csv',
                'n_bbox_val4.csv']
            fn_test = 'n_bbox_ts.csv'

    return fns_tr, fns_val, fn_ts




