import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk



proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome/data/csv_file/radcure'

df = pd.read_csv(proj_dir + '/radcure_label.csv')
#list = ['Yes', 'Possible', 'Persistent']
list = ['Yes', 'Possible']
events = []
locs = []
for i in range(df.shape[0]):
    if df['Local'][i] in list or df['Regional'][i] in list or df['Distant'][i] in list or df['Status'][i] == 'Dead':
        #print(i)
        event = 1
        events.append(event)
        locs.append(i)
    else:
        event = 0
        events.append(event)
#print(events)
#print(locs)

# df['Date Local'] = df['Date Local'].fillna(0)
# df['Date Regional'] = df['Date Regional'].fillna(0)
# df['Date Distant'] = df['Date Distant'].fillna(0)

times = []
count = 0
names = []
for i in locs:
    count += 1
    #if df['Date Local'][i] != 0:
    if df['Local'][i] in list:   
        print(count, 'local:', i, df['Date Local'][i])
        times.append(df['Date Local'][i])
        names.append(df['Local'][i])
    else:
        if df['Regional'][i] in list:
            print(count, 'regional:', i, df['Date Regional'][i])
            times.append(df['Date Regional'][i])
            names.append(df['Regional'][i])
        else:
            if df['Distant'][i] in list:
                print(count, 'distant:', i, df['Date Distant'][i])
                times.append(df['Date Distant'][i])
                names.append(df['Distant'][i])
            else:
                print(count, 'death:', i, df['Last FU'][i])
                times.append(df['Last FU'][i])
                names.append(df['Status'][i])
#print(times)
# print(locs)
print(len(events))
print(len(times))
print(len(locs))

df1 = pd.DataFrame({'loc': locs, 'name': names, 'time': times})
# pd.set_option('display.max_columns', None)
print('df1:', df1)
# df1.to_csv(proj_dir + '/tmep.csv', index=False)

locs = df1['loc'].to_list()
print(locs)
end_times = []
j = 0
for i in range(df.shape[0]):
    #print('i:', i)
    if i in locs:
        print(i, 'event')
        time = df1['time'][j]
        end_times.append(time)
        j += 1
    else:
        print(i, 'no event')
        time = df['Last FU'][i]
        end_times.append(time)
print(len(end_times))
#print('events:', events)

df['efs_event'], df['end_time'] = [events, end_times]
print(df)

efs_times = []
for start, end in zip(df['RT Start'], df['end_time']):
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
    
    #start_time = float(start.split('/')[0])*365 + float(start.split('/')[1])*30 + float(start.split('/')[2])
    #end_time = float(end.split('/')[0])*365 + float(end.split('/')[1])*30 + float(end.split('/')[2])
    #print(start_time)
    #print(end_time)
    efs_time = end_time - start_time
    #print(efs_time)
    efs_times.append(efs_time)
#print(efs_times)
df['efs_time'] = efs_times
df.to_csv(proj_dir + '/radcure_efs.csv', index=False)


