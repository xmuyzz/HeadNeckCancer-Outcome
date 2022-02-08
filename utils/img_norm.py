import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk

def img_norm(data, norm_type):
    
    ## normalize CT signal
    data[data <= -1024] = -1024
    ### strip skull, skull UHI = ~700
    data[data > 700] = 0
    ### normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
    if norm_type == 'np_interp':
        norm_data = np.interp(data, [-200, 200], [0, 1])
    elif norm_type == 'np_clip':
        data = np.clip(data, a_min=-200, a_max=200)
        MAX, MIN = data.max(), data.min()
        norm_data = (data - MIN) / (MAX - MIN)

    return norm_data
