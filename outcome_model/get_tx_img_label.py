import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold, train_test_split
from opts import parse_opts



def get_img_label(opt, dataset, data_dir, csv_dir, surv_type, img_size, img_type, tumor_type):

    print('surv_type:', surv_type)
    print('img type:', img_type)
    print('tumor type:', tumor_type)

    task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_size + '_' + \
               opt.img_type + '_' + opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 
    
    mydata_dir = data_dir + '/' + img_size + '_' + img_type
    img_dir = mydata_dir + '/tx_' + dataset + '_pr_pn'
    img_dirs = [i for i in sorted(glob.glob(img_dir + '/*nii.gz'))]

    # get img dirs and save to df
    fns = []
    for img_dir in img_dirs:
        fn = img_dir.split('/')[-1]
        fns.append(fn)
    df_img = pd.DataFrame({'nn_id': fns, 'img_dir': img_dirs})
    print('total img number:', df_img.shape[0])
    #print(df_img[0:10])

    df_label = pd.read_csv(csv_dir + '/' + dataset + '/' + dataset + '_efs.csv')
    # keep OPC only
    #df_label = df_label.loc[df_label['cancer_type']=='Oropharynx']
    #print(df_label)
    print('total OPC cases:', df_label.shape[0])

    df = df_label.merge(df_img, how='left', on='nn_id')
    df_vol = pd.read_csv(csv_dir + '/' + dataset + '/tumor_volume.csv')
    df = df.merge(df_vol, how='left', on='nn_id')

    # exclude img_dir = none
    df = df[df['img_dir'].notna()]
    print('total df size:', df.shape)
    #print(df[0:20])
    
    # drop patients without clinical data: rfs, os, lr, dr
    df = df.dropna(subset=[surv_type + '_event', surv_type + '_time'])
    print('test data shape:', df.shape)
    fn = 'tx_' + dataset + '_img_label_' + tumor_type + '.csv'
    save_dir1 = csv_dir + '/' + surv_type
    save_dir2 = mydata_dir + '/' + surv_type
    save_dir3 = task_dir + '/tx_' + dataset
    for save_dir in [save_dir1, save_dir2, save_dir3]:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(save_dir + '/' + fn, index=False)
    print('train and val dfs have been saved!!!')


    
if __name__ == '__main__':

    opt = parse_opts()
    np.random.seed(42)

    #data_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data'
    csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file'
    data_dir = '/home/xmuyzz/data/HNSCC/outcome'
    dataset = 'bwh'
    task = 'efs'
    tumor_type = 'pn'
    img_size = 'full'
    img_type = 'attn122'
    for dataset in ['bwh', 'maastro']:
        get_img_label(opt, dataset, data_dir, csv_dir, task, img_size, img_type, tumor_type)