import os
import numpy as np
import pandas as pd


proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/DFCI'
df1 = pd.read_csv(proj_dir + '/all_meta_data.csv', engine='python')
df2 = pd.read_csv(proj_dir + '/combined_seg.csv')
df2 = df2[['pat id']]
df2.columns = ['PMRN']
df = df2.merge(df1, how='left', on='PMRN')
df.to_csv(proj_dir + '/BWH_HN_metadata.csv', index=False)
