import os
import glob
import numpy as np
import pandas as pd



def get_gtv(data_dir, out_dir):

    paths = []
    segs = []
    IDs = []
    for path in sorted(glob.glob(data_dir + '/*/*.nrrd')):
        x = path.split('/')[-1].split('.')[0].split('_')
        seg = path.split('/')[-1].split('.')[0]
        ID = path.split('/')[-2].split('_')[1]
        if 'GTV' in x or 'GTV70' in x:
            paths.append(path)
            segs.append(seg)
            IDs.append(ID)
    print('patient number:', len(set(IDs)))
    df = pd.DataFrame({'ID': IDs, 'seg': segs, 'path': paths})
    print(df)
    df.to_csv(os.path.join(out_dir, 'dfci_seg.csv'), index=False)


if __name__ == '__main__':

    data_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/dfci_seg'
    out_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/dfci_out'

    get_gtv(data_dir, out_dir)
            
