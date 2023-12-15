import pandas as pd
import glob 
import numpy as np


#proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/test'
save_dir = '/home/xmuyzz/data/HNSCC/outcome/origin_atn122' 
#save_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/test'
tr = pd.read_csv(save_dir + '/tr_img_label_pn0.csv')
va = pd.read_csv(save_dir + '/va_img_label_pn0.csv')

img_paths = []
for i, id in enumerate(tr['seg_nn_id']):
    print(i, id)
    img_path = save_dir + '/tr_pn/' + id
    img_paths.append(img_path)
tr['img_dir'] = img_paths
tr.to_csv(save_dir + '/tr_img_label_pn.csv')
print('done')

img_paths = []
for i, id in enumerate(va['seg_nn_id']):
    print(i, id)
    img_path = save_dir + '/tr_pn/' + id
    img_paths.append(img_path)
va['img_dir'] = img_paths
va.to_csv(save_dir + '/va_img_label_pn.csv')
print('done')


