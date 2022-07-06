import os
import pandas as pd
import numpy as np
#import zipfile
import shutil
import glob
import dicom2nifti
from glob import iglob
from pathlib import Path
import dicom2nifti.settings as settings



def main(data_dir):
   
    count = 0
    for root, subdirs, files in os.walk(data_dir):
        count += 1
        print(count)
        for data in files:
            path = os.path.join(root, data)
            #print(path)
            os.rename(path, path.replace(' ', '_'))
            #print(path)
    

if __name__ == '__main__':

    data_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/dfci_seg'
    #curation_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH/BCH_curated/Girard_Michael_A_4228140'

    main(data_dir)
