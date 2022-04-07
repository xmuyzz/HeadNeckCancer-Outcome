import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold, train_test_split


def split_dataset(proj_dir, tumor_type, input_img_type):
    
    """
    Split dataset into train, tuning and val with 5-fold cross-validation;
    Args:
        proj_dir {path} -- path to project folder;
    
    Returns:
        Dataframes of train, tune and val containing data path and labels;
    
    """

    # create df for data and pat_id to match labels
    if tumor_type == 'primary_node':
        if input_img_type == 'masked_img':
            df_fn = 'df_img_label_pn_masked.csv'
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
        elif input_img_type == 'raw_img':
            df_fn = 'df_img_label_pn_raw.csv'
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
        if input_img_type == 'masked_img':
            df_fn = 'df_img_label_p_masked.csv'
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
            fn_test = 'df_p_masked_test.csv'
        elif input_img_type == 'raw_img':
            df_fn = 'df_img_label_p_raw.csv'
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
        if input_img_type == 'masked_img':
            df_fn = 'df_img_label_n_masked.csv'
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
        elif input_img_type == 'raw_img':
            df_fn = 'df_img_label_n_raw.csv'
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
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    df = pd.read_csv(os.path.join(pro_data_dir, df_fn))
    ## MDACC and PMH cohorts for training, tuning and internal validation
    df_develop = df.loc[df['group_id'].isin(['MDACC', 'PMH'])]
    ## CHUM and CHUS cohorts for external test
    df_test = df.loc[df['group_id'].isin(['CHUM', 'CHUS'])]

    # train_test_split for internal test
    #df_train_ = df_development.sample(frac=0.8, random_state=200)
    #df_test_in = df.drop(train.index)
    
    ## k-fold cross valiadtion on development set
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    dfs_train = []
    dfs_val = []
    #print(df_development[0:10])
    for idx_train, idx_val in kf.split(df_develop):
        df_train = df_develop.iloc[idx_train, :]
        df_val = df_develop.iloc[idx_val, :]
        dfs_train.append(df_train)
        dfs_val.append(df_val)
    print('train data shape:', dfs_train[0].shape)
    print('val data shape:', dfs_val[0].shape)
    print('test data shape:', df_test.shape)

    ## save train, val, test dfs
    for df_train, df_val, fn_train, fn_val in zip(dfs_train, dfs_val, fns_train, fns_val):
        df_train.to_csv(os.path.join(pro_data_dir, fn_train), index=False)
        df_val.to_csv(os.path.join(pro_data_dir, fn_val), index=False)
        
    df_test.to_csv(os.path.join(pro_data_dir, fn_test), index=False)
    print('train, val and test dfs have been saved!!!')








