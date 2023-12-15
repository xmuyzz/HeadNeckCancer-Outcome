import pandas as pd

csv_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/maastro'
df = pd.read_csv(csv_dir + '/maastro_label.csv')
FUs = []
for fu in df['Date Last FU']:
    FU = fu.split('/')[1] + '/' + fu.split('/')[0] + '/' + fu.split('/')[2]
    print(fu, FU)
    FUs.append(FU)
df['FU'] = FUs
df.to_csv(csv_dir + '/maastro_label.csv', index=False)
