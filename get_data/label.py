import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk



def label(proj_dir, clinical_data_file, save_label):
    
    clinical_data_dir = os.path.join(proj_dir, 'data')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    
    if not os.path.exists(clinical_data_dir): os.mkdir(clinical_data_dir)
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)

    df = pd.read_csv(os.path.join(clinical_data_dir, clinical_data_file))
    # drop duplicates in MDACC cohort, keep first patient that has distant control info
    df.drop_duplicates(subset=['patientid'], keep='first', inplace=True)
    print(df['distantcontrol'].to_list())
    #print(df['death'].to_list())
    #print(df['vitalstatus1'].to_list())
    #print(df['vitalstatus2'].to_list())
    df['locoreginalcontrol'] = df['locoregionalcontrol'].map({'Yes': 1, 'No': 0})
    df['localcontrol'] = df['localcontrol'].map({'Yes': 1, 'No': 0})
    df['distantcontrol'] = df['distantcontrol'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
    df['regionalcontrol'] = df['regionalcontrol'].map({'Yes': 1, 'No': 0})
    df['vitalstatus1'] = df['vitalstatus1'].map({'Dead': 1, 'Alive': 0})
    df['vitalstatus2'] = df['vitalstatus2'].map({'Dead': 1, 'Alive': 0})
    #print(df['vitalstatus1'].to_list())
    #print(df['vitalstatus2'].to_list())
    
    # create patient id
    #------------------
    patids = []
    groupids = []
    for patientid in df['patientid']:
        id = patientid.split('/')[-1].split('-')[0].strip()
        if id == 'HN':
            groupid = patientid.split('/')[-1].split('-')[1]
            patid = patientid.split('/')[-1].split('-')[1] + \
                    patientid.split('/')[-1].split('-')[2]
        elif id == 'HNSCC':
            groupid = 'MDACC'
            patid = 'MDACC' + patientid.split('/')[-1].split('-')[2][1:]
        elif id == 'OPC':
            groupid = 'PMH'
            patid = 'PMH' + patientid.split('/')[-1].split('-')[1][2:]
        groupids.append(groupid)
        patids.append(patid)

    # local/regional control event and duration time
    #-----------------------------------------------
    # events
    lr_ctrs = []
    for lr_ctr, l_ctr, r_ctr in zip(df['locoregionalcontrol'], df['localcontrol'], 
                                    df['regionalcontrol']):
        if lr_ctr == 1 or l_ctr == 1 or r_ctr == 1:
            lr_ctr = 1
        else: 
            lr_ctr = 0
        lr_ctrs.append(lr_ctr)
    # duration time
    lr_durations = []
    #print('fu:', df['daystolastfu'])
    for duration, fu in zip(df['locoregionalcontrol_duration'], df['daystolastfu']):
        # check if duration is nan or none
        if duration is None or np.isnan(duration):
            lr_duration = fu
            lr_durations.append(lr_duration)
        elif duration:
            lr_duration = duration
            lr_durations.append(lr_duration)
    #print('lr_durations:', lr_durations)    
 
    # distant control event and duration time
    #-----------------------------------------
    ## events
    ds_ctrs = []
    for ds, x in zip(df['distantcontrol'], 
                     df['siteofrecurrence(distal/local/locoregional)']):
        if ds == 1:
            ds_ctr = 1
        elif ds == 0:
            ds_ctr = 0
        elif np.isnan(ds) == True:
            if x in ['Distant metastasis', 
                     'Locoregional and distant metastasis', 
                     'Regional recurrence and distant metasatsis',
                     'Local recurrence and distant metastasis',
                     'Regional and distant metastasis']:
                ds_ctr = 0
            else:
                ds_ctr = 1
        ds_ctrs.append(ds_ctr)
    print(ds_ctrs)
    
    ## duration time
    ds_durations = []
    for duration, fu, ds, x in zip(df['distantcontrol_duration'], 
                                   df['daystolastfu'],
                                   df['distantcontrol'],
                                   df['disease-freeinterval(months)']):
        if np.isnan(ds) == False:
            if duration is None or np.isnan(duration):
                ds_duration = fu
            elif duration:
                ds_duration = duration
        elif np.isnan(ds) == True:
            ds_duration = x * 30
        ds_durations.append(ds_duration)
    #print('ds_durations:', ds_durations)
    
    #df = pd.DataFrame({'ds_ctr': ds_ctrs, 'df_duration': ds_durations})

    # overall survival event and duration time
    #-----------------------------------------
    ## events
    deaths = []
    for death, vital1, vital2 in zip(df['death'], df['vitalstatus1'], df['vitalstatus2']):
        if np.isnan(death) == False:
            a = death
        elif np.isnan(death) == True:
            if np.isnan(vital1) == False:
                a = vital1
            elif np.isnan(vital1) == True and np.isnan(vital2) == False:
                a = vital2
        deaths.append(a)
    survivals = []
    for x in deaths:
        if x == 0:
            survival = 1
        elif x == 1:
            survival = 0
        survivals.append(survival)
    print('deaths:', deaths)
    print('survivals:', survivals)
    ## duration time
    sur_durations = df['daystolastfu'].to_list()

    # create df for label info
    #----------------------------
    print('group ID:', len(groupids))
    print('patient ID:', len(patids))
    print('local/regional control event:', len(lr_ctrs))
    print('lpcal/regional control duration:', len(lr_durations))
    print('distant control event:', len(ds_ctrs))
    print('distant control duration:', len(ds_durations))
    print('survival:', len(survivals))
    print('survival duration:', len(sur_durations))

    df = pd.DataFrame({
        'groupid': groupids,
        'patid': patids,
        'lr_ctr': lr_ctrs,
        'lr_duration': lr_durations,
        'ds_ctr': ds_ctrs,
        'ds_duration': ds_durations,
        'survival': survivals,
        'sur_duration': sur_durations
        })

    df.to_csv(os.path.join(pro_data_dir, save_label), index=False)
    print('successfully save label file in csv format!')

#    ## delete patients in the empty or missing pn seg list
#    ## change seg names to be consistent with label files
#    fnss = []
#    for segs in [exclude_pn, exclude_p]:
#        fns = []
#        for seg in segs:
#            ID = seg.split('-')[0]
#            if ID == 'HNSCC':
#                fn = 'MDACC' + '_' + seg.split('-')[2][1:]
#            elif ID == 'OPC':
#                fn = 'PMH' + '_' + seg.split('-')[1][2:]
#            elif ID == 'HN':
#                if seg.split('-')[1] == 'CHUM':
#                    fn = 'CHUM' + '_' + seg.split('-')[2]
#                elif seg.split('-')[1] == 'CHUM':
#                    fn = 'CHUS' + '_' + seg.split('-')[2]
#            fns.append(fn)
#        fnss.append(fns)
#    exclude_pn = fnss[0] 
#    exclude_p = fnss[1]
#
#    print('total label number:', df.shape[0])
#    print(df['patid'][0:10])
#    df = df[~df['patid'].isin(exclude_pn)]
#    print('pn label number:', df.shape[0])
#    df.to_csv(os.path.join(pro_data_dir, save_label_pn))
#    
#    ## p seg
#    df = df[~df['patid'].isin(exclude_p)]
#    print('pn label number:', df.shape[0])
#    df.to_csv(os.path.join(pro_data_dir, save_label_p))
#
#    print('successfully save label file in csv format!')




