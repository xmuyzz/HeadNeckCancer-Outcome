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



def main(input_dir, output_dir):
    
    # Inconsistent slice incremement
    settings.disable_validate_slice_increment()
    settings.enable_resampling()
    settings.set_resample_spline_interpolation_order(1)
    settings.set_resample_padding(-1000)
    # single slice
    settings.disable_validate_slicecount()
    
    for folder in os.listdir(input_dir):
        print(folder)
        path = os.path.join(input_dir, folder)
        dicom2nifti.convert_directory(
            dicom_directory=path, 
            output_folder=output_dir, 
            compression=True, 
            reorient=True)



if __name__ == '__main__':

    input_dir = '/mnt/aertslab/USERS/Christian/For_Ben'
    output_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/dfci_data'

    main(input_dir, output_dir)




