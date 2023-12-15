import pandas as pd
import os
import numpy as np


csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/bwh'

df1 = pd.read_csv(csv_dir + '/bwh_efs.csv')
df2 = pd.read_csv(csv_dir + '/bwh_adipose.csv')
df1['patient_id'] = df1['patient_id'].astype(str)
df2['patient_id'] = df2['patient_id'].astype(str)

df = df1.merge(df2, on='patient_id', how='left')

# adipose
ads = []
for adi in df['Auto_Adipose_Area']:
    if adi < df['Auto_Adipose_Area'].median():
        adipose = 0
        ads.append(adipose)
    else:
        adipose = 1
        ads.append(adipose)
df['adipose'] = ads 

df.to_csv(csv_dir + '/bwh_tot_labels.csv', index=False)
print('successfully saved bwh total label file in csv!!!')
