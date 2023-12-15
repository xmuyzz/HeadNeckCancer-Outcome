import os
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
from time import localtime, strftime
from sklearn.cluster import KMeans
from opts import parse_opts
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index



csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/clinical_model'
df_tr = pd.read_csv(csv_dir + '/tr_img_label_pn.csv')
df_va = pd.read_csv(csv_dir + '/va_img_label_pn.csv')
df_ts = pd.read_csv(csv_dir + '/ts_img_label_pn.csv')

df_tr = df_tr.dropna(subset=['efs_time', 'efs_event', 'Age>65', 'Female', 'Smoking>10py', 'T-Stage', 'N-Stage'])
df_tr = df_tr[['efs_time', 'efs_event', 'Age>65', 'Female', 'Smoking>10py', 'T-Stage', 'N-Stage']]
df_va = df_va.dropna(subset=['efs_time', 'efs_event', 'Age>65', 'Female', 'Smoking>10py', 'T-Stage', 'N-Stage'])
df_va = df_va[['efs_time', 'efs_event', 'Age>65', 'Female', 'Smoking>10py', 'T-Stage', 'N-Stage']]
df_ts = df_ts.dropna(subset=['efs_time', 'efs_event', 'Age>65', 'Female', 'Smoking>10py', 'T-Stage', 'N-Stage'])
df_ts = df_ts[['efs_time', 'efs_event', 'Age>65', 'Female', 'Smoking>10py', 'T-Stage', 'N-Stage']]

# fitting coxph model
print('coxph df:')
cph = CoxPHFitter()
cph.fit(df_tr, duration_col='efs_time', event_col='efs_event')
cph.print_summary()

# c-index on val/test set
c_index = concordance_index(df_tr['efs_time'], -cph.predict_partial_hazard(df_tr), df_tr['efs_event'])
print('tr c-index:', round(c_index, 2))

c_index = concordance_index(df_va['efs_time'], -cph.predict_partial_hazard(df_va), df_va['efs_event'])
print('val c-index:', round(c_index, 2))

c_index = concordance_index(df_ts['efs_time'], -cph.predict_partial_hazard(df_ts), df_ts['efs_event'])
print('ts c-index:', round(c_index, 2))



# prediction
# x = cph.predict_survival_function(df_va)
# print(x)
# x = cph.predict_median(df_va)
# print(x)
# x = cph.predict_partial_hazard(df_va)
# print(x)







