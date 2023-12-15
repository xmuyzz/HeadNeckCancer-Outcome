import pandas as pd

proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/maastro'
df = pd.read_csv(proj_dir + '/maastro_efs.csv')
df['Female'] = df['gender'].map({'Female': 1, 'Male': 0})
print(df['Female'])
df.to_csv(proj_dir + '/maastro_efs.csv', index=False)