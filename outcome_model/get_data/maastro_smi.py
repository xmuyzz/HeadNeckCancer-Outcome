import pandas as pd
import numpy as np
import os 


csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/maastro'
df = pd.read_csv(csv_dir + '/maastro_efs_raw.csv')
df = df.dropna(subset=['height cm', 'weight kg']).reset_index()

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

# calculate SMI
SMIs = []
for i in range(df.shape[0]):
    #print(df['Auto_Muscle_Area'])
    CSA = df['muscle_area'][i]
    age = df['age'][i]
    sex = df['sex'][i]
    weight = df['weight kg'][i]
    height = df['height cm'][i] / 100
    SMI = (27.30 + (1.36 * CSA) - (0.67 * age) + (0.64 * weight) + (26.44 * sex)) / (height * height)
    #print('SMI:', SMI)
    SMIs.append(SMI)
df['SMI'] = np.around(SMIs, 2)

# determine sarcopenia
sarcos = []
for sex, smi in zip(df['sex'], df['SMI']):
    if sex == 1:
        if smi > 38.5:
            sarcopenia = 1 
            sarcos.append(sarcopenia)
        else:
            sarcopenia = 0
            sarcos.append(sarcopenia)
    elif sex == 2:
        if smi > 52.4:
            sarcopenia = 1 
            sarcos.append(sarcopenia)
        else:
            sarcopenia = 0    
            sarcos.append(sarcopenia)
df['sarcopenia'] = sarcos

# adipose
ads = []
for adi in df['adipose_area']:
    if adi < df['adipose_area'].median():
        adipose = 0
        ads.append(adipose)
    else:
        adipose = 1
        ads.append(adipose)
df['adipose'] = ads 

# female = 1, male = 0
females = []
for gender in df['gender']:
    if gender == 'female':
        female = 1
        females.append(female)
    else:
        female = 0
        females.append(female)
df['female'] = females

# age >= 65
ages = []
for age in df['age']:
    if age >= 65:
        age65 = 1
        ages.append(age65)
    else:
        age65 = 0
        ages.append(age65)
df['age65'] = ages

# smoking >10py
smokes = []
for py in df['PY']:
    if py >= 10:
        smoke = 1
        smokes.append(smoke)
    else:
        smoke = 0
        smokes.append(smoke)
df['smoke>10py'] = smokes

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
Ts = []
for T in df['cT (8th)']:
    if T in ['1', '2']:
        t = 0
        Ts.append(t)
    else:
        t = 1
        Ts.append(t)
df['T-34'] = Ts

# N stage
Ns = []
for N in df['cN (8th)']:
    if N in ['0', '1']:
        n = 0
        Ns.append(n)
    else:
        n = 1
        Ns.append(n)
df['N-23'] = Ns

# AJCC stage
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
df['Stage-12'] = stages

df.to_csv(csv_dir + '/maastro_tot_labels.csv', index=False)
print('successfully saved masstro total label file in csv!!!')








