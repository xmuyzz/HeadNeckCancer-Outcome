import pandas as pd


proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/maastro'
df = pd.read_csv(proj_dir + '/maastro_efs_raw.csv')
ids = []
for id in df['nn_id']:
    nn_id = id.split('_.nii.gz')[0] + '.nii.gz'
    print(nn_id)
    ids.append(nn_id)
df['nn_id'] = ids
df.to_csv(proj_dir + '/maastro_efs_raw.csv')

