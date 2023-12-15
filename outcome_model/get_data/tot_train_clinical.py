import pandas as pd
import os 
import numpy as np


# csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/train_set'
# df0 = pd.read_csv(csv_dir + '/tot_tr_label.csv')

# df1 = pd.read_csv(csv_dir + '/TCIA_meta_data.csv')
# ids = []
# for i, id in enumerate(df1['patientid']):
#     print(i)
#     if 'HN-' in id:
#         patient_id = id.split('-')[1] + '_' + id.split('-')[2]
#         print(patient_id)
#         ids.append(patient_id)
#     elif 'HNSCC' in id:
#         patient_id = 'MDA_' + id[-3:]
#         print(patient_id)
#         ids.append(patient_id)
#     elif 'OPC' in id:
#         patient_id = 'PMH_' + id[-3:]
#         print(patient_id)
#         ids.append(patient_id)
# print(ids)
# df1['ID'] = ids
# df1 = df1[['ID', 'gender', 'ageatdiag', 't-category', 'n-category', 'm-category', 'ajccstage']]
# df = df0.merge(df1, how='left', on='ID').reset_index()

# # get T-stage and N-stage for Radcure patients
# df2 = pd.read_csv(csv_dir + '/radcure_efs.csv')
# ids = []
# for id in df2['patient_id']:
#     pat_id = id.replace('-', '_')    
#     print(pat_id)
#     ids.append(pat_id)
# df2['ID'] = ids
# df2 = df2[['ID', 'T', 'N']]
# df = df.merge(df2, on='ID', how='left').reset_index()
# print(df)


# # HPV: 0 - negative, 1 - positive, 2 - unknown;
# hpvs = []
# for hpv in df['hpv']:
#     if hpv in ['Yes, Negative', 'Yes, negative', 'negative']:
#         x = 0
#         hpvs.append(x)
#     elif hpv in ['Yes, positive', 'Yes, Positive', 'positive']:
#         x = 1
#         hpvs.append(x)
#     else:
#         x = 2
#         hpvs.append(x)
# df['HPV'] = hpvs

# # AJCC-stages
# stages = []
# for x in df['stage']:
#     if x in ['IV', 'IVA', 'IVB']:
#         x = 4
#         stages.append(x)
#     elif hpv in ['III']:
#         x = 3
#         stages.append(x)
#     elif hpv in ['II']:
#         x = 2
#         stages.append(x)
#     elif hpv in ['I']:
#         x = 1
#         stages.append(x)
#     else:
#         x = hpv
#         stages.append(x)
# df['AJCC-Stage'] = stages

# # t-category
# Ts = []
# for t in df['t-category']:
#     if t in ['T4', 'T4a', 'T4b', '4a', '4']:
#         x = 4
#         Ts.append(x)
#     elif t in ['T3', '3']:
#         x = 3
#         Ts.append(x)
#     elif t in ['T2', '2']:
#         x = 2
#         Ts.append(x)
#     elif t in ['T1', '1']:
#         x = 1
#         Ts.append(x)
#     else:
#         x = t
#         Ts.append(x)
# df['T-Stage'] = Ts

# # n-category
# Ns = []
# for t in df['n-category']:
#     if t in ['N3', '3']:
#         x = 3
#         Ns.append(x)
#     elif t in ['N2', 'N2b', '2', '2a', '2b', '2c']:
#         x = 2
#         Ns.append(x)
#     elif t in ['N1', '1']:
#         x = 1
#         Ns.append(x)
#     elif t in ['N0', '0']:
#         x = 0
#         Ns.append(x)
#     else:
#         x = t
#         Ns.append(x)
# df['N-Stage'] = Ns

# print(df)
# df.to_csv(csv_dir + '/new_tot.csv', index=False)



csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/train_set'
df = pd.read_csv(csv_dir + '/new_tot.csv')

# AJCC-stages
stages = []
for s in df['ajccstage']:
    if s in ['IV', 'IVA', 'IVB', 'IVC', 'Stade IVA', 'Stade IVB', 'Stage IV']:
        x = 4
        stages.append(x)
    elif s in ['III', 'IIIA', 'IIIC', 'Stade III', 'Stage III']:
        x = 3
        stages.append(x)
    elif s in ['II', 'IIA', 'IIB', 'Stade II', 'StageII']:
        x = 2
        stages.append(x)
    elif s in ['I', 'IB', 'Stade I']:
        x = 1
        stages.append(x)
    else:
        x = ''
        stages.append(x)
print(stages)
df['AJCC-Stage'] = stages

# t-stage
Ts = []
for t in df['T-Stage']:
    if t in ['T4', 'T4a', 'T4b', '4']:
        x = 4
        Ts.append(x)
    elif t in ['T3', 'T3 (2)', '3', '3a']:
        x = 3
        Ts.append(x)
    elif t in ['T2', 'T2 (2)', 'T2a', 'T2b', '2']:
        x = 2
        Ts.append(x)
    elif t in ['T1', 'T1 (2)', 'T1b', 'T1a', '1']:
        x = 1
        Ts.append(x)
    else:
        x = ''
        Ts.append(x)
df['T-Stage'] = Ts

# n-stage
Ns = []
for t in df['N-Stage']:
    if t in ['3', 'N3', 'N3a', 'N3b']:
        x = 3
        Ns.append(x)
    elif t in ['N2', 'N2a', 'N2b', 'N2c', '2']:
        x = 2
        Ns.append(x)
    elif t in ['N1', '1']:
        x = 1
        Ns.append(x)
    elif t in ['N0', '0']:
        x = 0
        Ns.append(x)
    else:
        x = ''
        Ns.append(x)
df['N-Stage'] = Ns


# sex: Female - 1; Male: 0
fs = []
for f in df['sex']:
    if f in ['F', 'Female']:
        x = 1
        fs.append(x)
    elif f in ['M', 'Male']:
        x = 0
        fs.append(x)
    else:
        x = ''
        fs.append(x)
df['Female'] = fs

# age>65
ages = []
for age in df['age']:
    if float(age) >= 65 :
        x = 1
        ages.append(x)
    else:
        x = 0
        ages.append(x)
df['Age>65'] = ages

# smoking>10py
smokes = []
for s in df['smoke']:
    if s == 'na':
        x = ''
        smokes.append(x)
    elif float(s) >= 10 :
        x = 1
        smokes.append(x)
    elif float(s) < 10:
        x = 0
        smokes.append(x)
    else:
        x = ''
        smokes.append(x)
df['Smoking>10py'] = smokes

print(df)
df.to_csv(csv_dir + '/tr_tot.csv', index=False)