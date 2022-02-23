import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold, train_test_split


def split_dataset(proj_dir):
    
    """
    Split dataset into train, tuning and val with 5-fold cross-validation;

    @Args:
        proj_dir {path} -- path to project folder;
    
    @Returns:
        Dataframes of train, tune and val containing data path and labels;
    
    """

    pn_masked_arr_dir = os.path.join(proj_dir, 'data/pn_masked_arr')
    pn_raw_arr_dir = os.path.join(proj_dir, 'data/pn_raw_arr')
    p_masked_arr_dir = os.path.join(proj_dir, 'data/PMH_files/p_masked_arr')
    p_raw_arr_dir = os.path.join(proj_dir, 'data/MDACC_files/p_raw_arr')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    
    df = pd.read_csv(os.path.join(pro_data_dir, 'df_img_label.csv'))
    ## MDACC and PMH cohorts for training, tuning and internal validation
    df_develop = df.loc[df['group_id'].isin(['MDACC', 'PMH'])]
    ## CHUM and CHUS cohorts for external test
    df_test = df.loc[df['group_id'].isin(['CHUM', 'CHUS'])]

    # train_test_split for internal test
    #df_train_ = df_development.sample(frac=0.8, random_state=200)
    #df_test_in = df.drop(train.index)
    
    ## k-fold cross valiadtion on development set
    #added some parameters
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    #result = next(kf.split(df_development), None)
    #print(result)
    dfs_train = []
    dfs_val = []
    #print(df_development[0:10])
    for idx_train, idx_val in kf.split(df_develop):
        #print(idx_train)
        #print(idx_val)
        df_train = df_develop.iloc[idx_train, :]
        df_val = df_develop.iloc[idx_val, :]
        dfs_train.append(df_train)
        dfs_val.append(df_val)
    
    print('train data shape:', dfs_train[0].shape)
    print('val data shape:', dfs_val[0].shape)
    print('test data shape:', df_test.shape)

    ## save train, val, test dfs
    fns_train = ['df_train0.csv', 'df_train1.csv', 'df_train2.csv', 
                 'df_train3.csv', 'df_train4.csv']
    fns_val = ['df_val0.csv', 'df_val1.csv', 'df_val2.csv', 
               'df_val3.csv', 'df_val4.csv']
    for df_train, df_val, fn_train, fn_val in zip(dfs_train, dfs_val, fns_train, fns_val):
        df_train.to_csv(os.path.join(pro_data_dir, fn_train), index=False)
        df_val.to_csv(os.path.join(pro_data_dir, fn_val), index=False)
        
    df_test.to_csv(os.path.join(pro_data_dir, 'df_test.csv'), index=False)
    print('train, val and test dfs have been saved!!!')








