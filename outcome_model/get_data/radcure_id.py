import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
import shutil


#------------------------
# get radcure and opc ID
#-------------------------
# proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/attn123_original'
# df = pd.read_csv(proj_dir + '/ID_radcure_opc.csv')
# radcure_ids = []
# opc_ids = []
# for ID in df['ID']:
#     radcure_id = ID.split(',')[0]
#     opc_id = ID.split(',')[1]
#     print(radcure_id, opc_id)
#     radcure_ids.append(radcure_id)
#     opc_ids.append(opc_id)
# df['radcure_id'], df['opc_id'] = [radcure_ids, opc_ids]
# df.to_csv(proj_dir + '/ID_radcure_opc.csv')


# #-----------------------------
# # get radcure, opc, nnUNet ID
# #-----------------------------
# proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/attn123_original'
# df1 = pd.read_csv(proj_dir + '/ID_radcure_opc.csv')
# df2 = pd.read_csv(proj_dir + '/ID_radcure_nn.csv')
# df = df1.merge(df2, on='radcure_id', how='left').reset_index()
# df.to_csv(proj_dir + '/ID_radcure_opc_nn.csv', index=False)


#--------------------------------------
# remove OPC cases from radcure dataset
#--------------------------------------
proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/attn123_original'
df = pd.read_csv(proj_dir + '/ID_radcure_opc_nn.csv')
img_ids = []
for i, nn_id in enumerate(df['nn_id']):
    try:
        img_id = nn_id.split('_0000.')[0] + '.nii.gz'
        #print(i, nn_id, img_id)
        img_ids.append(img_id)
    except Exception as e:
        print(nn_id, e)

radcure_opc_dir = proj_dir + '/radcure_opc'
if not os.path.exists(radcure_opc_dir):
    os.makedirs(radcure_opc_dir)
count = 0
for img_path in glob.glob(proj_dir + '/tr_radcure_pn/*.nii.gz'):
    img_id = img_path.split('/')[-1]
    if img_id in img_ids:
        count += 1
        print(count, img_id)
        save_path = radcure_opc_dir + '/' + img_id
        shutil.move(img_path, save_path)
print('successfully transfer all opc file!!!')
