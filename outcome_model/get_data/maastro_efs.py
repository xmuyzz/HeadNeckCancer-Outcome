import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk



proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/maastro'
df = pd.read_csv(proj_dir + '/maastro_label.csv')

events = []
locs = []
for i in range(df.shape[0]):
    if df['Local Failure'][i] == 1 or df['Regional Failure'][i] == 1 or df['DM'][i] == 1 or df['Status'][i] == 'Deceased':
        #print(i)
        event = 1
        events.append(event)
        locs.append(i)
    else:
        event = 0
        events.append(event)
#print(events)
#print(locs)


times = []
count = 0
names = []
for i in range(df.shape[0]):
    count += 1
    #if df['Date Local'][i] != 0:
    if df['Local Failure'][i] == 1:   
        print(count, 'local:', i, df['Date Local Failure'][i])
        times.append(df['Date Local Failure'][i])
        names.append(df['Local Failure'][i])
    else:
        if df['Regional Failure'][i] == 1:
            print(count, 'regional:', i, df['Date Regional Failure'][i])
            times.append(df['Date Regional Failure'][i])
            names.append(df['Regional Failure'][i])
        else:
            if df['DM'][i] == 1:
                print(count, 'distant:', i, df['Date DM'][i])
                times.append(df['Date DM'][i])
                names.append(df['DM'][i])
            else:
                print(count, 'death or live:', i, df['fu'][i])
                times.append(df['fu'][i])
                names.append(df['Status'][i])
#print(times)
# print(locs)
print('events:', len(events))
print('times:', len(times))
print('names:', len(names))

df1 = pd.DataFrame({'name': names, 'time': times, 'event': events})
# pd.set_option('display.max_columns', None)
print('df1:', df1)
# df1.to_csv(proj_dir + '/tmep.csv', index=False)

df['end_time'], df['event type'], df['efs_event'] = [times, names, events]
print(df)

efs_times = []
for start, end in zip(df['start time'], df['end_time']):
    #print(start)
    #print(end)
    if float(start.split('/')[2]) < 25:
        start_time = (float(start.split('/')[2]) + 100)*365 + float(start.split('/')[0])*30 + float(start.split('/')[1])
    else:
        start_time = float(start.split('/')[2])*365 + float(start.split('/')[0])*30 + float(start.split('/')[1])
    
    if float(end.split('/')[2]) < 25:
        end_time = (float(end.split('/')[2]) + 100)*365 + float(end.split('/')[0])*30 + float(end.split('/')[1])
    else:
        end_time = float(end.split('/')[2])*365 + float(end.split('/')[0])*30 + float(end.split('/')[1])       
    
    #print(start_time)
    #print(end_time)
    efs_time = end_time - start_time
    #print(efs_time)
    efs_times.append(efs_time)
#print(efs_times)
df['efs_time'] = efs_times

# add muscle and adipose area/density 
#-------------------------------------
df0 = pd.read_csv(proj_dir + '/maastro.csv')
df['Muscle_Area'] = df0['Auto_Muscle_Area'].to_list()
df['Adipose_Area'] = df0['Auto_Adipose_Area'].to_list()
df['Muscle_Density'] = df0['Auto_Muscle_Density'].to_list()
df['Adipose_Density'] = df0['Auto_Adipose_Density'].to_list()


# change clinical variables
#-----------------------------
sexs = []
for gender in df['gender']:
    if gender == 'Female':
        sex = 1
        sexs.append(sex)
    elif gender == 'Male':
        sex = 2
        sexs.append(sex)
    else:
        print('wrong input')
df['sex'] = sexs

# calculate BMI
bmis = []
for h, w in zip(df['height cm'], df['weight kg']):
    bmi = w / (h*h*0.0001)
    #print(bmi)
    bmis.append(bmi)
df['BMI'] = np.around(bmis, 2)

# adipose
ads = []
for adi in df['Adipose_Area']:
    if adi < df['Adipose_Area'].median():
        adipose = 0
        ads.append(adipose)
    else:
        adipose = 1
        ads.append(adipose)
df['Adipose01'] = ads 

# female = 1, male = 0
females = []
for gender in df['gender']:
    if gender in ['female', 'Female']:
        female = 1
        females.append(female)
    elif gender in ['male', 'Male']:
        female = 0
        females.append(female)
df['Female'] = females

# age >= 65
ages = []
for age in df['age']:
    if age >= 65:
        age65 = 1
        ages.append(age65)
    else:
        age65 = 0
        ages.append(age65)
df['Age>65'] = ages

# smoking >10py
smokes = []
for py in df['PY']:
    if py >= 10:
        smoke = 1
        smokes.append(smoke)
    else:
        smoke = 0
        smokes.append(smoke)
df['Smoking>10py'] = smokes

# ACE-27 score
aces = []
for ace in df['ACE-27']:
    if ace >= 2:
        ace_27 = 1
        aces.append(ace_27)
    else:
        ace_27 = 0
        aces.append(ace_27)
df['ACE01vs23'] = aces

# T stage
#------------
Ts = []
for T in df['cT (8th)']:
    if T in ['1', '2']:
        t = 0
        Ts.append(t)
    else:
        t = 1
        Ts.append(t)
df['T-Stage-01'] = Ts

Ts = []
for T in df['cT (8th)'].astype(str):
    if T in ['1']:
        t = 1
        Ts.append(t)
    if T in ['2']:
        t = 2
        Ts.append(t)
    if T in ['3']:
        t = 3
        Ts.append(t)
    elif T in ['4', '4a', '4b']:
        t = 4
        Ts.append(t)
print(len(Ts))
df['T-Stage-1234'] = Ts


# N stage: use 1, 2, 3, 4
#-------------------------
Ns = []
for N in df['cN (8th)']:
    if N in ['0', '1']:
        n = 0
        Ns.append(n)
    else:
        n = 1
        Ns.append(n)
df['N-Stage-01'] = Ns

# N stage: use 0 and 1
Ns = []
for N in df['cN (8th)']:
    if N in ['0']:
        n = 0
        Ns.append(n)
    elif N in ['1']:
        n = 1
        Ns.append(n)
    elif N in ['2']:
        n = 2
        Ns.append(n)
    else:
        n = 3
        Ns.append(n)
df['N-Stage-0123'] = Ns


# AJCC stage
#-------------
stages = []
for s in df['stage (TNM 8)']:
    if s in ['1']:
        print(s)
        stage = 1
        stages.append(stage)
    elif s in ['2']:
        print(s)
        stage = 2
        stages.append(stage)   
    elif s in ['3']:
        print(s)
        stage = 3
        stages.append(stage)   
    else:
        print(s)
        stage = 4
        stages.append(stage)
print(stages)
df['AJCC-Stage-01'] = stages

stages = []
for s in df['stage (TNM 8)']:
    if s in ['1', '2']:
        print(s)
        stage = 0
        stages.append(stage)
    else:
        print(s)
        stage = 1
        stages.append(stage)
print(stages)
df['AJCC-Stage-1234'] = stages

# fix nn_id
ids = []
for id in df['nn_id']:
    nn_id = id.split('_.nii.gz')[0] + '.nii.gz'
    #print(nn_id)
    ids.append(nn_id)
df['nn_id'] = ids
# calculate SMI and sarcopenia
#------------------------------
# SMIs = []
# for i in range(df.shape[0]):
#     #print(df['Auto_Muscle_Area'])
#     CSA = df['muscle_area'][i]
#     age = df['age'][i]
#     sex = df['sex'][i]
#     weight = df['weight kg'][i]
#     height = df['height cm'][i] / 100
#     SMI = (27.30 + (1.36 * CSA) - (0.67 * age) + (0.64 * weight) + (26.44 * sex)) / (height * height)
#     #print('SMI:', SMI)
#     SMIs.append(SMI)
# df['SMI'] = np.around(SMIs, 2)

# # determine sarcopenia
# sarcos = []
# for sex, smi in zip(df['sex'], df['SMI']):
#     if sex == 1:
#         if smi > 38.5:
#             sarcopenia = 1 
#             sarcos.append(sarcopenia)
#         else:
#             sarcopenia = 0
#             sarcos.append(sarcopenia)
#     elif sex == 2:
#         if smi > 52.4:
#             sarcopenia = 1 
#             sarcos.append(sarcopenia)
#         else:
#             sarcopenia = 0    
#             sarcos.append(sarcopenia)
# df['sarcopenia'] = sarcos


df.to_csv(proj_dir + '/maastro_efs.csv', index=False)


