import pandas as pd


# maastro dataset
csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/maastro'
df = pd.read_csv(csv_dir + '/maastro_efs.csv')

# adipose
ads = []
for adi in df['adipose_area']:
    if adi < df['adipose_area'].median():
        adipose = 0
        ads.append(adipose)
    else:
        adipose = 1
        ads.append(adipose)
df['Adipose_Area'] = ads 

# adipose
ads = []
for adi in df['adipose_density']:
    if adi < df['adipose_density'].median():
        adipose = 0
        ads.append(adipose)
    else:
        adipose = 1
        ads.append(adipose)
df['Adipose_Density'] = ads 

df.to_csv(csv_dir + '/maastro_efs.csv', index=False)


# BWH dataset
csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/bwh'
df = pd.read_csv(csv_dir + '/bwh_efs.csv')

# adipose
ads = []
for adi in df['adipose_area']:
    if adi < df['adipose_area'].median():
        adipose = 0
        ads.append(adipose)
    else:
        adipose = 1
        ads.append(adipose)
df['Adipose_Area'] = ads 

# adipose
ads = []
for adi in df['adipose_density']:
    if adi < df['adipose_density'].median():
        adipose = 0
        ads.append(adipose)
    else:
        adipose = 1
        ads.append(adipose)
df['Adipose_Density'] = ads 

df.to_csv(csv_dir + '/bwh_efs.csv', index=False)