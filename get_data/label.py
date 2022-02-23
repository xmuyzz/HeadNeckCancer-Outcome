import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk



def label(proj_dir, clinical_data_file, save_label):
    
    """
    Create dataframe to store labels;

    Args:
        proj_dir {path} -- project path;
        clinical_data_file {csv} -- clinical meta file;
        save_label {str} -- save label;

    Returns:
        dataframe;

    Raise errors:
        None

    """

    clinical_data_dir = os.path.join(proj_dir, 'data')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    
    if not os.path.exists(clinical_data_dir): os.mkdir(clinical_data_dir)
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)

    df = pd.read_csv(os.path.join(clinical_data_dir, clinical_data_file))
    # drop duplicates in MDACC cohort, keep first patient that has distant control info
    df.drop_duplicates(subset=['patientid'], keep='first', inplace=True)
    #print(df['distantcontrol'].to_list())
    #print(df['death'].to_list())
    #print(df['vitalstatus1'].to_list())
    #print(df['vitalstatus2'].to_list())
    
    """only keep oropharynx patients
    """
    print('total patient number:', df.shape[0])
    df = df.loc[df['diseasesite'].isin(['Oropharynx'])]
    print('total oropharynx patient number:', df.shape[0])
    print('cancer type:', df['diseasesite'])
    cancer_types = df['diseasesite'].to_list()

    """
    Local/regional recurence labeled as event "1", LR control labeled as no event "0";
    Distant recurence labeled as event "1", distant control labeled as no event "0";
    Death labeled as event "1", survive labeled as no event "0";
    """
    df['locoreginalcontrol'] = df['locoregionalcontrol'].map({'Yes': 0, 'No': 1})
    df['localcontrol'] = df['localcontrol'].map({'Yes': 0, 'No': 1})
    df['distantcontrol'] = df['distantcontrol'].map({'Yes': 0, 'No': 1, '1': 0, '0': 1})
    df['regionalcontrol'] = df['regionalcontrol'].map({'Yes': 0, 'No': 1})
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
    lr_events = []
    for lr_event, l_event, r_event in zip(df['locoregionalcontrol'], 
                                          df['localcontrol'], 
                                          df['regionalcontrol']):
        if lr_event == 1 or l_event == 1 or r_event == 1:
            lr_event = 1
        else: 
            lr_event = 0
        lr_events.append(lr_event)
    # duration time
    lr_times = []
    #print('fu:', df['daystolastfu'])
    for time, fu in zip(df['locoregionalcontrol_duration'], df['daystolastfu']):
        # check if duration is nan or none 
        if time is None or np.isnan(time):
            lr_time = fu
            lr_times.append(lr_time)
        else:
            lr_time = time
            lr_times.append(lr_time)
        #print('lr_durations:', lr_durations)    
 

    # distant control event and duration time
    #-----------------------------------------
    ## events
    ds_events = []
    for ds, x in zip(df['distantcontrol'], 
                     df['siteofrecurrence(distal/local/locoregional)']):
        if ds == 1:
            ds_event = 1
        elif ds == 0:
            ds_event = 0
        elif np.isnan(ds) == True:
            if x in ['Distant metastasis', 
                     'Locoregional and distant metastasis', 
                     'Regional recurrence and distant metasatsis',
                     'Local recurrence and distant metastasis',
                     'Regional and distant metastasis']:
                ds_event = 0
            else:
                ds_event = 1
        ds_events.append(ds_event)
    print(ds_events)
    
    ## duration time
    ds_times = []
    for duration, fu, ds, x in zip(df['distantcontrol_duration'], 
                                   df['daystolastfu'],
                                   df['distantcontrol'],
                                   df['disease-freeinterval(months)']):
        if np.isnan(ds) == False:
            if duration is None or np.isnan(duration):
                ds_time = fu
            elif duration:
                ds_time = duration
        elif np.isnan(ds) == True:
            ds_time = x * 30
        ds_times.append(ds_time)
    #print('ds_durations:', ds_durations)
    
    #df = pd.DataFrame({'ds_ctr': ds_ctrs, 'df_duration': ds_durations})

    # overall survival event and duration time
    #-----------------------------------------
    ## events
    death_events = []
    for death, vital1, vital2 in zip(df['death'], 
                                     df['vitalstatus1'], 
                                     df['vitalstatus2']):
        if np.isnan(death) == False:
            a = death
        elif np.isnan(death) == True:
            if np.isnan(vital1) == False:
                a = vital1
            elif np.isnan(vital1) == True and np.isnan(vital2) == False:
                a = vital2
        death_events.append(a)
#    survivals = []
#    for x in deaths:
#        if x == 0:
#            survival = 1
#        elif x == 1:
#            survival = 0
#        survivals.append(survival)
#    print('survivals:', survivals)
    print('death:', death_events)
    ## duration time
    death_times = df['daystolastfu'].to_list()

    # create df for label info
    #----------------------------
    print('group ID:', len(groupids))
    print('patient ID:', len(patids))
    print('local/regional event:', len(lr_events))
    print('lpcal/regional time:', len(lr_times))
    print('distant event:', len(ds_events))
    print('distant time:', len(ds_times))
    print('death_event:', len(death_events))
    print('death time:', len(death_times))

    df = pd.DataFrame({
        'group_id': groupids,
        'pat_id': patids,
        'cancer_type': cancer_types,
        'lr_event': lr_events,
        'lr_time': lr_times,
        'ds_event': ds_events,
        'ds_time': ds_times,
        'death_event': death_events,
        'death_time': death_times
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




