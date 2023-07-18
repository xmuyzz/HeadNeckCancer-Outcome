import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk


proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file'

# TCIA label data
#-----------------
df = pd.read_csv(proj_dir + '/label.csv')
IDs = []
for i, pat_id in enumerate(df['pat_id']):
    ID = pat_id[:-3] + '_' + pat_id[-3:]
    if ID.split('_')[0] == 'MDACC':
        ID = 'MDA_' + ID.split('_')[1] 
    #print(i, ID)
    IDs.append(ID)
df['ID'] = IDs
rfs_events = []
rfs_times = []
for lr_event, lr_time, ds_event, ds_time in zip(df['lr_event'], df['lr_time'], df['ds_event'], df['ds_time']):
    if lr_event == 1 and ds_event == 0:
        rfs_event, rfs_time = 1, lr_time
    elif lr_event == 0 and ds_event == 1:
        rfs_event, rfs_time = 1, ds_time
    elif lr_event == 1 and ds_event == 1:
        rfs_event, rfs_time == 1, lr_time
    else:
        rfs_event, rfs_time = 0, lr_time
    rfs_events.append(rfs_event)
    rfs_times.append(rfs_time)
#print(df)
df['rfs_event'], df['rfs_time'] = rfs_events, rfs_times
df.to_csv(proj_dir + '/label.csv', index=False)

# hecktor RFS label data
#-----------------------
df1 = pd.read_csv(proj_dir + '/hecktor_rfs.csv')
IDs = []
for pat_id in df1['pat_id']:
    ID = pat_id.replace('-', '_')
    #print(ID)
    IDs.append(ID)
df1['ID'] = IDs
df1.columns = ['pat_id', 'rfs_event', 'rfs_time', 'ID']

# train and test set
#-------------------
csv0s = ['df_ts_pn.csv', 'df_tr_pn.csv']
csv1s = ['ts_label.csv', 'tr_label.csv']
for csv0, csv1 in zip(csv0s, csv1s):
    df2 = pd.read_csv(proj_dir + '/' + csv0)
    df2 = df2.merge(df1, how='left', on='ID')
    df2 = df2.merge(df, how='left', on='ID')
    #print(df2)
    rfs_events = []
    rfs_times = []
    for rfs_event_x, rfs_time_x, rfs_event_y, rfs_time_y in zip(df2['rfs_event_x'], df2['rfs_time_x'], 
                                                                df2['rfs_event_y'], df2['rfs_time_y']):
        if not np.isnan(rfs_event_x):
            #print('rfs_event_x:', rfs_event_x)
            rfs_event = rfs_event_x
            rfs_time = rfs_time_x
            print('Hecktor data:', rfs_event)
        else:
            rfs_event = rfs_event_y
            rfs_time = rfs_time_y
            print('TCIA data:', rfs_event)
        rfs_events.append(rfs_event)
        rfs_times.append(rfs_time)
    df2['rfs_event'] = rfs_events
    df2['rfs_time'] = rfs_times
    print(rfs_events)
    df2 = df2.drop(columns=['pat_id_x', 'pat_id_y', 'rfs_event_x', 'rfs_time_x', 
                            'rfs_event_y', 'rfs_time_y', 'img_dir', 'seg_dir'])
    df2.to_csv(proj_dir + '/' + csv1, index=False)






