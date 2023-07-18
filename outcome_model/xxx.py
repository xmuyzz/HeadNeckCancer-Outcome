import os
import pandas as pd
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torchtuples as tt
from time import strftime
from datetime import datetime
import pytz
import time

start = time.time()
print('start')
end = time.time()
duration = end - start
print('total time:', duration)


tz = pytz.timezone('US/Eastern')
print(datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'))

time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f')
print('\n%s: cnn_model: %s%s' % (datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'), 'resnet', 11))
print('\n%s: cnn_model: %s%s' % (datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'), 'resnet', 11))
print('\n%s: cnn_model: %s%s' % (datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'), 'resnet', 11))

def check():
    proj_dir ='/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file'
    #proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/pro_data'

    df = pd.read_csv(proj_dir + '/tr_img_label_pn.csv')
    #df = pd.read_csv(proj_dir + '/df_img_label_pn_raw.csv')
    df = df.dropna(subset=['rfs_time', 'rfs_event'])
    df = df[0:10] 
    times = torch.from_numpy(df['rfs_time'].to_numpy())
    events = torch.from_numpy(df['rfs_event'].to_numpy())
    #times = torch.from_numpy(df['lr_time'].to_numpy())
    #events = torch.from_numpy(df['lr_event'].to_numpy())

    datas = []
    for i, img_dir in enumerate(df['img_dir']):
        img = sitk.ReadImage(img_dir)
        img = sitk.GetArrayFromImage(img)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        print(i, img.shape)
        #img = np.stack((img), axis=0)
        #print(img.shape)
        time = times[i]
        event = events[i]
        img = torch.from_numpy(img).float()
        data = (img, (time, event))
        print(data)
        #print(data.shape)
        #data = tt.tuplefy(data)
        #print(data.shapes())
        datas.append(data)
        #data = tt.tuplefy(data).stack()
        #data = tt.tuplefy(data)
        #print(data.shapes)
    #print(datas)
    data = tt.tuplefy(datas).stack()
    print(data.shapes())
    #print(data)
    
