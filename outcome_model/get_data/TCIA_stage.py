import pandas as pd
import numpy as np
import os


proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/train_set'

df0 = pd.read_csv(proj_dir + '/new_tot_data.csv')
df1 = pd.read_csv(proj_dir + '/radcure_efs.csv')
df2 = pd.read_csv(proj_dir + '/TCIA_meta_data.csv')

df1 = df1[['patient_id', 'T', 'N']]
