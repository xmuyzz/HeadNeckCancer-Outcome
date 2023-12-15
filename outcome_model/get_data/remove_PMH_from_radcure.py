import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
import shutil


proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/train_set'
df0 = pd.read_csv(proj_dir + '/new_tot_label.csv')
df1 = pd.read_csv(proj_dir + '/ID_radcure_opc_nn.csv')

pmh_ids = []
for opc_id in df1['opc_id']:
    pmh_id = 'PMH' + '_' + opc_id[-3:]
    print(pmh_id)
    pmh_ids.append(pmh_id)
df1['pmh_id'] = pmh_ids
print(len(pmh_ids))

df0 = df0[~df0['ID'].isin(pmh_ids)]
print(df0)


# get cohort name
cohorts = []
for x in df0['cohort']:
    if 'RADCURE' in x:
        #print(x)
        cohort = x.split('-')[0]
        #print(cohort)
        cohorts.append(cohort)
    else:
        cohort = x
        cohorts.append(cohort)
print(len(cohorts))
df0['cohort'] = cohorts

# get consistent ID
IDs = []
for ID in df0['ID']:
    if '-' in ID:
        id = ID.replace('-', '_')
        print(id)
        IDs.append(id)
    else:
        id = ID
        IDs.append(id)
        print(id)
df0['ID'] = IDs

print(df0)
df0.to_csv(proj_dir + '/tot_development_label.csv', index=False)

    
        

