import pandas as pd
import glob 
import numpy as np


# proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/maastro'
# save_dir = '/home/xmuyzz/data/HNSCC/outcome/csv_file' 
# #save_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/test'
# df0 = pd.read_csv(proj_dir + '/maastro_clinical.csv')
# df1 = pd.read_csv(proj_dir + '/tx_maastro_pn.csv')
# df = df1.merge(df0, how='left', on='patient_id').reset_index()

# # remove 'img_dir' column
# df.drop(['img_dir'], axis=1, inplace=True)

# # nn id
# ids = []
# for id in df['img_nn_id']:
#     #print(id)
#     id = id.split('_0000.nii')[0] + '.nii.gz'
#     print(id)
#     ids.append(id)
# df['nn_id'] = ids

# # events and times
# times = []
# for fu in df['Follow-up time']:
#     time = fu*30
#     #print(time)
#     times.append(time)

# print(df['Status'])
# events = []
# for status in df['Status']:
#     if status == 'NED':
#         event = 0
#     elif status == 'Deceased':
#         event = 1
#     #print(event)
#     events.append(event)

# df['os_event'] = events
# df['os_time'] = times

# df.to_csv(save_dir + '/maastro_label.csv', index=False)
# print(df)
# print('successfully save lable file to CSV fodler!')


proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/bwh'
save_dir = '/home/xmuyzz/data/HNSCC/outcome/csv_file' 
#save_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/test'
df0 = pd.read_csv(proj_dir + '/bwh_clinical.csv')
df0 = df0.loc[df0['Site 1 (for staging)']=='Oropharynx']
df1 = pd.read_csv(proj_dir + '/tx_bwh_pn.csv')
df0['patient_id'] = df0['patient_id'].astype(str)
print('df0 size:', df0.shape[0])
df1['patient_id'] = df1['patient_id'].astype(str)
print('df1 size:', df1.shape[0]) 
df = df1.merge(df0, how='left', on='patient_id').reset_index()
df = df.loc[df['Site 1 (for staging)']=='Oropharynx']
df = df.dropna(subset='os_event')

# nn id
ids = []
for id in df['img_nn_id']:
    #print(id)
    id = id.split('_0000.nii')[0] + '.nii.gz'
    print(id)
    ids.append(id)
df['nn_id'] = ids

print('data size:', df.shape[0])
df.to_csv(save_dir + '/bwh_label.csv', index=False)