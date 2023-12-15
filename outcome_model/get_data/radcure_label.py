import numpy as np
import pandas as pd
import glob


# proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/data/radcure'
# df = pd.read_csv(proj_dir + '/RADCURE_TCIA_Clinical.csv')
# count = 0
# events = []
# times = []
# for status, fu in zip(df['Status'], df['Length FU']):
#     count += 1
#     time = int(float(fu)*365)
#     if status == 'Dead':
#         event = 1
#     elif status == 'Alive':
#         event = 0
#     print(count, time, event)
#     times.append(time)
#     events.append(event)
# df['os_event'], df['os_time'] = [events, times]
# print(df)
# df.to_csv(proj_dir + '/radcure_label.csv')


# combine racure label with nnUNet ID
#------------------------------------
proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file'
df = pd.read_csv(proj_dir + '/ID_radcure_nn.csv')
img_ids = []
for i, nn_id in enumerate(df['nn_id']):
    try:
        img_id = nn_id.split('_0000.')[0] + '.nii.gz'
        print(i, nn_id, img_id)
        img_ids.append(img_id)
    except Exception as e:
        print(nn_id, e)
df['nnUNet_id'] = img_ids
df = df[['nnUNet_id', 'radcure_id']]
df.columns = ['nn_id', 'radcure_id']
print(df)

df1 = pd.read_csv(proj_dir + '/radcure_label.csv')
df0 = df1.merge(df, on='radcure_id', how='left')
df0.to_csv(proj_dir + '/radcure_label.csv', index=False)




