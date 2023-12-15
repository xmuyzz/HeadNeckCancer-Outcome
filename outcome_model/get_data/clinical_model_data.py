import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold, train_test_split



def get_img_label(data_dir, csv_dir, label_file, surv_type, img_size, img_type, tumor_type):

    print('surv_type:', surv_type)
    print('img type:', img_type)
    print('tumor type:', tumor_type)
    
    mydata_dir = data_dir + '/' + img_size + '_' + img_type
    img_dir = mydata_dir + '/tot_pn'
    img_dirs = [i for i in sorted(glob.glob(img_dir + '/*nii.gz'))]

    fns = []
    for img_dir in img_dirs:
        fn = img_dir.split('/')[-1]
        fns.append(fn)
    df_img = pd.DataFrame({'nn_id': fns, 'img_dir': img_dirs})
    print('total img number:', df_img.shape[0])

    df_label = pd.read_csv(csv_dir + '/train_set/' + label_file)
    # keep OPC only
    df_label = df_label.loc[df_label['cancer_type']=='Oropharynx']
    print('total OPC cases:', df_label.shape[0])

    df = df_label.merge(df_img, how='left', on='nn_id')
    # exclude img_dir = none
    df = df[df['img_dir'].notna()]
    print('total df size:', df.shape)
    #print(df[0:20])
    
    # drop patients without clinical data: rfs, os, lr, dr
    df = df.dropna(subset=[surv_type + '_event', surv_type + '_time'])
    df1 = df.dropna(subset=['efs_event', 'efs_time'])
    df2 = df.dropna(subset=['os_event', 'os_time'])
    print('efs number:', df1.shape[0])
    print('os number:', df2.shape[0])

    # stratify tr and val based on rfs
    df_tr_, df_ts = train_test_split(df, test_size=0.2, stratify=df[surv_type + '_event'], random_state=1234)
    #print(df_tr)
    #df_tr, df_va = train_test_split(df_tr_, test_size=0.1, stratify=df[surv_type + '_event'], random_state=1234)
    df_tr, df_va = train_test_split(df_tr_, test_size=0.2, random_state=1234)
    print('train data shape:', df_tr.shape)
    print('val data shape:', df_va.shape)
    print('test data shape:', df_ts.shape)
    tr_fn = 'tr_clinical.csv'
    va_fn = 'va_clinical.csv'
    ts_fn = 'ts_clinical.csv'
    df_tr.to_csv(csv_dir + '/clinical_model/' + tr_fn, index=False)
    df_va.to_csv(csv_dir + '/clinical_model/' + va_fn, index=False)
    df_ts.to_csv(csv_dir + '/clinical_model/' + ts_fn, index=False)
    print('train and val dfs have been saved!!!')

    
if __name__ == '__main__':
    
    csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file'
    data_dir = '/home/xmuyzz/data/HNSCC/outcome'
    label_file = 'tr_tot.csv'
    task = 'efs'
    #data_set = 'ts_gt'
    tumor_type = 'pn'
    img_size = 'full'
    img_type = 'attn122'
    get_img_label(data_dir, csv_dir, label_file, task, img_size, img_type, tumor_type)