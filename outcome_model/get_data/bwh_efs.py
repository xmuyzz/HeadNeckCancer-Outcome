import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk



proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/bwh'
df = pd.read_csv(proj_dir + '/bwh_label.csv')

events = []
locs = []
for i in range(df.shape[0]):
    if df['Local/Regional Recurrence 1'][i] == 'Yes' or df['Distant Metastasis'][i] == 'Yes' or df['Dead'][i] == 1:
        #print(i)
        event = 1
        events.append(event)
        locs.append(i)
    else:
        event = 0
        events.append(event)
#print(events)
#print(locs)


times = []
count = 0
for i in range(df.shape[0]):
    count += 1
    #if df['Date Local'][i] != 0:
    if df['Local/Regional Recurrence 1'][i] == 1:   
        print(count, 'local:', i, df['Date of Recurrence Diagnosis'][i])
        times.append(df['Date of Recurrence Diagnosis'][i])
    else:
        if df['Distant Metastasis'][i] == 1:
            print(count, 'regional:', i, df['Date of DM Diagnosis'][i])
            times.append(df['Date of DM Diagnosis'][i])
        else:
            if df['Dead'][i] == 1:
                print(count, 'dead:', i, df['Date of Death'][i])
                times.append(df['Date of Death'][i])
            else:
                print(count, 'live:', i, df['Last Oncological Follow-Up'][i])
                times.append(df['Last Oncological Follow-Up'][i])
#print(times)
# print(locs)
print('events:', len(events))
print('times:', len(times))

df1 = pd.DataFrame({'time': times, 'event': events})
# pd.set_option('display.max_columns', None)
print('df1:', df1)
# df1.to_csv(proj_dir + '/tmep.csv', index=False)

df['end_time'], df['efs_event'] = [times, events]
print(df)

efs_times = []
for start, end in zip(df['Pre-treatment date vitals'], df['end_time']):
    #print(start)
    #print(end)
    if float(start.split('/')[2]) < 25:
        start_time = (float(start.split('/')[2]) + 100)*365 + float(start.split('/')[0])*30 + float(start.split('/')[1])
    else:
        start_time = float(start.split('/')[2])*365 + float(start.split('/')[0])*30 + float(start.split('/')[1])
    
    if float(end.split('/')[2]) < 25:
        end_time = (float(end.split('/')[2]) + 100)*365 + float(end.split('/')[0])*30 + float(end.split('/')[1])
    else:
        end_time = float(end.split('/')[2])*365 + float(end.split('/')[0])*30 + float(end.split('/')[1])       
    
    #print(start_time)
    #print(end_time)
    efs_time = end_time - start_time
    #print(efs_time)
    efs_times.append(efs_time)
#print(efs_times)
df['efs_time'] = efs_times
df.to_csv(proj_dir + '/bwh_efs.csv', index=False)


