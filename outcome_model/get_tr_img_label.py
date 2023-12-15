import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
import sklearn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import minmax_scale
import torch
import torch.nn.functional as F
from opts import parse_opts


def get_img_label(opt, data_dir, csv_dir, label_file, surv_type, img_size, img_type, tumor_type):
    """
    create df for data and pat_id to match labels 
    Args:
        proj_dir {path} -- project dir;
        out_dir {path} -- output dir;
        save_img_type {str} -- image type: nii or npy;
    Returns:
        Dataframe with image dirs and labels;
    Raise errors:
        None
    """
    print('surv_type:', surv_type)
    print('img type:', img_type)
    print('tumor type:', tumor_type)

    task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_size + '_' + \
               opt.img_type + '_' + opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 
    
    mydata_dir = data_dir + '/' + img_size + '_' + img_type
    img_dir = mydata_dir + '/tot_pn'
    img_dirs = [i for i in sorted(glob.glob(img_dir + '/*nii.gz'))]

    fns = []
    for img_dir in img_dirs:
        fn = img_dir.split('/')[-1]
        fns.append(fn)
    df_img = pd.DataFrame({'nn_id': fns, 'img_dir': img_dirs})
    print('total img number:', df_img.shape[0])
    #print(df_img[0:10])
    #df_label = pd.read_csv(data_dir + '/csv_file/tot_label.csv')
    #df_label = pd.read_csv(csv_dir + '/TCIA_Radcure_label.csv')
    df_label = pd.read_csv(csv_dir + '/train_set/' + label_file)
    # keep OPC only
    print('tot HNC cases:', df_label.shape[0])
    #df_label = df_label.loc[df_label['cancer_type']=='Oropharynx']
    #print('total OPC cases:', df_label.shape[0])

    df = df_label.merge(df_img, how='left', on='nn_id')
    # add tumor volume 
    df_vol = pd.read_csv(csv_dir + '/train_set/tumor_volume.csv')
    df = df.merge(df_vol, how='left', on='nn_id')
    # exclude img_dir = none
    df = df[df['img_dir'].notna()]
    print('total df size:', df.shape)
    #print(df[0:20])
    
    df = df.dropna(subset=[surv_type + '_event', surv_type + '_time'])
    
    # test os and efs patient number
    df1 = df.dropna(subset=['efs_event', 'efs_time'])
    df2 = df.dropna(subset=['os_event', 'os_time'])
    print('efs number:', df1.shape[0])
    print('os number:', df2.shape[0])

    # normalize clinical variables
    #-----------------------------------------
    print('HPV:', df['HPV'].to_list())
    print('T-Stage-1234:', df['T-Stage-1234'].to_list())
    print('N-Stage-0123:', df['N-Stage-0123'].to_list())
    print('Age>65:', df['Age>65'].to_list())
    print('Age:', df['age'].to_list())
    print('Female:', df['Female'].to_list())
    print('Smoking>10py:', df['Smoking>10py'].to_list())

    # normalize clinical information without one hot encoding
    df = df.dropna(subset=['T-Stage-1234', 'N-Stage-0123', 'Age>65', 'Female', 'age'])
    df['T-Stage-1234'] = minmax_scale(df['T-Stage-1234']).astype(float)
    df['N-Stage-0123'] = minmax_scale(df['N-Stage-0123']).astype(float)
    df['Age>65'] = minmax_scale(df['Age>65']).astype(float)
    df['age'] = minmax_scale(df['age']).astype(float)
    df['Female'] = minmax_scale(df['Female']).astype(float)


    # one hot encoding for patient clinical variables
    #-------------------------------------------------
    # Step 1: Impute missing values (replace NaNs with a default value, e.g., 'Unknown')
    # Step 2: One-hot encode the clinical variable

    # age
    df['Age>65'].fillna(2, inplace=True)
    df['Age>65'] = df['Age>65'].astype(int)
    age = torch.tensor(df['Age>65'].to_numpy())
    num_classes = torch.max(age) + 1
    print('age num class:', num_classes)
    one_hot_encoded = F.one_hot(age, num_classes=5)
    #print('one hot encoding:', one_hot_encoded.tolist())
    df['Age>65_oh'] = one_hot_encoded.tolist()
    print(df['Age>65_oh'])

    # sex
    df['Female'].fillna(2, inplace=True)
    df['Female'] = df['Female'].astype(int)
    sex = torch.tensor(df['Female'].to_numpy())
    num_classes = torch.max(sex) + 1
    print('sex num class:', num_classes)
    one_hot_encoded = F.one_hot(sex, num_classes=5)
    #print('one hot encoding:', one_hot_encoded.tolist())
    df['Female_oh'] = one_hot_encoded.tolist()
    print(df['Female_oh'])

    # T-Stage-1234
    df['T-Stage-1234'].fillna(0, inplace=True)
    df['T-Stage-1234'] = df['T-Stage-1234'].astype(int)
    t_stage = torch.tensor(df['T-Stage-1234'].to_numpy())
    num_classes = torch.max(t_stage) + 1
    print('t_stage num class:', num_classes)
    one_hot_encoded = F.one_hot(t_stage, num_classes=5)
    #print('one hot encoding:', one_hot_encoded.tolist())
    df['T-Stage-1234_oh'] = one_hot_encoded.tolist()
    print(df['T-Stage-1234_oh'])

    # N-Stage-0123
    df['N-Stage-0123'].fillna(4, inplace=True)
    df['N-Stage-0123'] = df['N-Stage-0123'].astype(int)
    n_stage = torch.tensor(df['N-Stage-0123'].to_numpy())
    num_classes = torch.max(n_stage) + 1
    print('n_stage num class:', num_classes)
    one_hot_encoded = F.one_hot(n_stage, num_classes=5)
    #print('one hot encoding:', one_hot_encoded.tolist())
    df['N-Stage-0123_oh'] = one_hot_encoded.tolist()
    print(df['N-Stage-0123_oh'])

    # smoking status
    df['Smoking>10py'].fillna(2, inplace=True)
    df['Smoking>10py'] = df['Smoking>10py'].astype(int)
    smoking = torch.tensor(df['Smoking>10py'].to_numpy())
    num_classes = torch.max(smoking) + 1
    print('smoking num class:', num_classes)
    one_hot_encoded = F.one_hot(smoking, num_classes=5)
    #print('one hot encoding:', one_hot_encoded.tolist())
    df['Smoking>10py_oh'] = one_hot_encoded.tolist()
    print(df['Smoking>10py_oh'])

    # HPV status
    df['HPV'].fillna(2, inplace=True)
    df['HPV'] = df['HPV'].astype(int)
    hpv = torch.tensor(df['HPV'].to_numpy())
    num_classes = torch.max(hpv) + 1
    print('hpv num class:', num_classes)
    one_hot_encoded = F.one_hot(smoking, num_classes=5)
    #print('one hot encoding:', one_hot_encoded.tolist())
    df['HPV_oh'] = one_hot_encoded.tolist()
    print(df['HPV_oh'])


    # split tr and val based on entire dataset
    #-----------------------------------------------
    # df_tr_, df_ts = train_test_split(df, test_size=0.2, stratify=df[surv_type + '_event'], random_state=1234)
    # #print(df_tr)
    # #df_tr, df_va = train_test_split(df_tr_, test_size=0.1, stratify=df[surv_type + '_event'], random_state=1234)
    # df_tr, df_va = train_test_split(df_tr_, test_size=0.2, random_state=1234)

    ### split test set baesd on Radcure dataset
    #-------------------------------------------------------------
    #df0 = df[df['cohort'].isin(['RADCURE', 'MDA'])]
    df0 = df[df['cohort'].isin(['RADCURE'])]
    print('new data shape:', df.shape[0])
    #df0 = df0[df0['HPV'].isin([0, 1])]
    #print('new data shape:', df.shape[0])
    
    # df_xx, df_ts = train_test_split(df0, test_size=0.3, stratify=df0[surv_type + '_event'], random_state=1234)
    # df_tr_va = df[~df['ID'].isin(df_ts['ID'])]
    # df_tr, df_va = train_test_split(df_tr_va, test_size=0.2, stratify=df_tr_va[surv_type + '_event'], random_state=1234)

    # df_xx, df_ts = train_test_split(df0, test_size=0.6, stratify=df0[surv_type + '_event'], random_state=1234)
    # df_yy, df_va = train_test_split(df_xx, test_size=0.2, stratify=df_xx[surv_type + '_event'], random_state=1234)
    # df_tr = df[~df['ID'].isin(df_ts['ID'].to_list() + df_va['ID'].to_list())]

    df_xx, df_ts = train_test_split(df0, test_size=0.6, stratify=df0[surv_type + '_event'], random_state=1234)
    df_tr_va = df[~df['ID'].isin(df_ts['ID'].to_list())]
    df_tr, df_va = train_test_split(df_tr_va, test_size=0.2, stratify=df_tr_va[surv_type + '_event'], random_state=1234)
    
    print('df_tr size:', df_tr.shape[0])
    print('df_va size:', df_va.shape[0])
    print('df_ts size:', df_ts.shape[0])

    # add C3 muscle and adipose to radcure test dataset
    df_c3 = pd.read_csv(csv_dir + '/train_set/Radcure_CSA.csv')
    ids = []
    for id in df_c3['patient_id']:
        id = id.replace('-', '_')
        #print(id)
        ids.append(id)
    df_c3['ID'] = ids
    df_c3 = df_c3[['ID', 'Muscle_Area', 'Muscle_Density', 'Adipose_Area', 'Adipose_Density']]
    df_ts = df_ts.merge(df_c3, on='ID', how='left').reset_index()

    tr_fn = 'tr_img_label_' + tumor_type + '.csv'
    va_fn = 'va_img_label_' + tumor_type + '.csv'
    ts_fn = 'ts_img_label_' + tumor_type + '.csv'

    save_dir1 = csv_dir + '/' + surv_type
    save_dir2 = mydata_dir + '/' + surv_type
    for save_dir in [save_dir1, save_dir2]:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df_tr.to_csv(save_dir + '/' + tr_fn, index=False)
        df_va.to_csv(save_dir + '/' + va_fn, index=False)
        df_ts.to_csv(save_dir + '/' + ts_fn, index=False)

    # save img and label files to task dir
    tr_save_dir = task_dir + '/tr'
    va_save_dir = task_dir + '/va'
    ts_save_dir = task_dir + '/ts'
    for save_dir in [tr_save_dir, va_save_dir, ts_save_dir]:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    df_tr.to_csv(tr_save_dir + '/' + tr_fn, index=False)
    df_va.to_csv(va_save_dir + '/' + va_fn, index=False)
    df_ts.to_csv(ts_save_dir + '/' + ts_fn, index=False)
    
    print('train and val dfs have been saved!!!')

    
if __name__ == '__main__':

    opt = parse_opts()

    np.random.seed(42)
    _ = torch.manual_seed(123)
    
    csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file'
    data_dir = '/home/xmuyzz/data/HNSCC/outcome'
    #label_file = 'new_tot_label.csv'
    label_file = 'tr_tot.csv'
    task = 'efs'
    #data_set = 'ts_gt'
    tumor_type = 'pn'
    img_size = 'full'
    img_type = 'attn122'
    get_img_label(opt, data_dir, csv_dir, label_file, task, img_size, img_type, tumor_type)