import pandas as pd


csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file'
df = pd.read_csv(csv_dir + '/TCIA_Radcure_label.csv')
df['cancer_type'] = df['cancer_type'].fillna('Oropharynx')
df.to_csv(csv_dir + '/new_tot_label.csv', index=False)