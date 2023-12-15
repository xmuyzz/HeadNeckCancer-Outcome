import pandas as pd
import glob 
import numpy as np
import shutil
import os


proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/maastro' 
df = pd.read_csv(proj_dir + '/maastro_efs.csv')
ids = []
for id in df['nn_id']:
    nn_id = id.split('_.')[0] + '.nii.gz'
    print(nn_id)
    ids.append(nn_id)
df['nn_id'] = ids
df.to_csv(proj_dir + '/maastro_efs.csv', index=False)

# proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/original2_attn123' 
# count = 0
# for img_dir in glob.glob(proj_dir + '/imagesTs_radcure/*nii.gz'):
#     id = img_dir.split('/')[-1]
#     dataset = id.split('_')[0]
#     key = id.split('_')[1]
#     if dataset == 'TS':
#         count += 1
#         new_id = 'TSR_' + key
#         print(count, id, new_id)
#         save_dir = proj_dir + '/tr_pn/' + new_id
#         os.rename(img_dir, save_dir)


# # change image Ts ID in nnUNet
# nnUNet_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task502_tot_p_n'
# count = 0
# for img_dir in glob.glob(nnUNet_dir + '/imagesTs_radcure/*nii.gz'):
#     id = img_dir.split('/')[-1]
#     key = id.split('TS_')[1]
#     count += 1
#     new_id = 'TSR_' + key
#     print(count, id, new_id)
#     save_dir = nnUNet_dir + '/imagesTs_radcure/' + new_id
#     os.rename(img_dir, save_dir)


# # change seg Ts ID in nnUNet
# nnUNet_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task502_tot_p_n'
# count = 0
# for img_dir in glob.glob(nnUNet_dir + '/predsTs_radcure/*nii.gz'):
#     id = img_dir.split('/')[-1]
#     key = id.split('_')[1]
#     count += 1
#     new_id = 'TSR_' + key
#     print(count, id, new_id)
#     save_dir = nnUNet_dir + '/predsTs_radcure/' + new_id
#     os.rename(img_dir, save_dir)


# proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/original2_attn123' 
# #df = pd.read_csv(proj_dir + '/tr_label0.csv')
# df = pd.read_csv(proj_dir + '/radcure_label.csv')
# count = 0
# ids = []
# for id in df['nn_id']:
#     count += 1
#     if pd.isnull(id):
#         print(count, id)
#     if not pd.isnull(id):
#         dataset = id.split('_')[0]
#         key = id.split('_')[1]
#         if dataset == 'TS':
#             new_id = 'TSR' + '_' + key        
#         else:
#             new_id = id
#         print(count, new_id)
#         ids.append(new_id)
# df['nn_id'] = ids
# df.to_csv(proj_dir + '/radcure_label.csv', index=False)





