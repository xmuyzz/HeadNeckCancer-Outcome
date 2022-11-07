import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold, train_test_split


def split_dataset(proj_dir, tumor_type, input_img_type, test_split_type='internal_split'):
    
    """
    Split dataset into train, tuning and val with 5-fold cross-validation;
    Args:
        proj_dir {path} -- path to project folder;    
    Returns:
        Dataframes of train, tune and val containing data path and labels;
    """
    if tumor_type == 'pn':
        fns_train = ['pn_tr0.csv', 'pn_tr1.csv', 'pn_tr2.csv', 'pn_tr3.csv', 'pn_tr4.csv']
        fns_val = ['pn_val0.csv', 'pn_val1.csv', 'pn_val2.csv', 'pn_val3.csv', 'pn_val4.csv']
        fn_test = 'pn_ts.csv'
   if tumor_type == 'p':
        fns_train = ['p_tr0.csv', 'p_tr1.csv', 'p_tr2.csv', 'p_tr3.csv', 'p_tr4.csv']
        fns_val = ['p_val0.csv', 'p_val1.csv', 'p_val2.csv', 'p_val3.csv', 'p_val4.csv']
        fn_test = 'p_ts.csv'
    if tumor_type == 'n':
        fns_train = ['n_tr0.csv', 'n_tr1.csv', 'n_tr2.csv', 'n_tr3.csv', 'n_tr4.csv']
        fns_val = ['n_val0.csv', 'n_val1.csv', 'n_val2.csv', 'n_val3.csv', 'n_val4.csv']
        fn_test = 'n_ts.csv'
    
    df = pd.read_csv(proj_dir + '/pro_data/' + df_fn))
    if test_split_type == 'internal_split':
        # train_test_split for internal test
        df_trval = df.sample(frac=0.8, random_state=1234)
        df_ts = df.drop(train.index)
    elif test_split_type == 'external_split':
        ## MDACC and PMH cohorts for training, tuning and internal validation
        trval_IDs = ['MDACC', 'PMH']
        ts_IDs = ['CHUM', 'CHUS']
        df_trval = df.loc[df['group_id'].isin(tr_IDs)]
        df_ts = df.loc[df['group_id'].isin(ts_IDs)]

    # k-fold cross valiadtion on development set
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    dfs_tr = []
    dfs_val = []
    #print(df_development[0:10])
    for idx_tr, idx_val in kf.split(df_trval):
        df_tr = df_trval.iloc[idx_tr, :]
        df_val = df_trval.iloc[idx_val, :]
        dfs_tr.append(df_tr)
        dfs_val.append(df_val)
    print('train data shape:', dfs_tr[0].shape)
    print('val data shape:', dfs_val[0].shape)
    print('test data shape:', df_ts.shape)
    ## save train, val, test dfs
    for df_tr, df_val, fn_tr, fn_val in zip(dfs_tr, dfs_val, fns_tr, fns_val):
        df_train.to_csv(pro_data_dir + '/' + fn_tr, index=False)
        df_val.to_csv(pro_data_dir + '/' + fn_val, index=False)
    df_ts.to_csv(pro_data_dir, + '/' + fn_ts, index=False)
    print('train, val and test dfs have been saved!!!')








