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
    output_dir = os.path.join(proj_dir, 'output')
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)

    df = pd.read_csv(os.path.join(pro_data_dir, clinical_data_file))
    # drop duplicates in MDACC cohort, keep first patient that has distant control info
    df.drop_duplicates(subset=['patientid'], keep='first', inplace=True)
    #print(df['distantcontrol'].to_list())
    #print(df['death'].to_list())
    #print(df['vitalstatus1'].to_list())
    #print(df['vitalstatus2'].to_list())
    
    # only keep oropharynx patients
    #--------------------------------
    print('total patient number:', df.shape[0])
    df = df.loc[df['diseasesite'].isin(['Oropharynx'])]
    print('total oropharynx patient number:', df.shape[0])
    print('cancer type:', df['diseasesite'])
    cancer_types = df['diseasesite'].to_list()

    
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
    """
    Local/regional recurence labeled as event "1", 
    LR control labeled as no event "0"
    """
    #print(df['locoregionalcontrol'].to_list())
    #print(df['locoregionalcontrol'].value_counts())
    # local/region control
    df['locoregionalcontrol'] = df['locoregionalcontrol'].astype(str)
    df['locoregionalcontrol'] = df['locoregionalcontrol'].replace({'1': 'Yes', '0': 'No'})
    df['locoregionalcontrol'] = df['locoregionalcontrol'].replace({'Yes': 0, 'No': 1})
    df['locoregionalcontrol'] = df['locoregionalcontrol'].astype(float)
    # local control
    df['localcontrol'] = df['localcontrol'].astype(str)
    df['localcontrol'] = df['localcontrol'].map({'Yes': 0, 'No': 1})
    df['localcontrol'] = df['localcontrol'].astype(float)
    # regional control
    df['regionalcontrol'] = df['regionalcontrol'].astype(str)
    df['regionalcontrol'] = df['regionalcontrol'].map({'Yes': 0, 'No': 1})
    df['regionalcontrol'] = df['regionalcontrol'].astype(float)
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
    print('lr events:', lr_events)
    # time
    df['daystolastfu'] = df['daystolastfu'].astype(float)
    df['locoregionalcontrol_duration'].fillna(value=100, inplace=True)
    df['locoregionalcontrol_duration'] = df['locoregionalcontrol_duration'].astype(float)
    df['localcontrol_duration'].fillna(value=100, inplace=True)
    df['localcontrol_duration'] = df['localcontrol_duration'].astype(float)
    df['regionalcontrol_duration'].fillna(value=100, inplace=True)
    df['regionalcontrol_duration'] = df['regionalcontrol_duration'].astype(float)
    lr_times = []
    for lr_t, l_t, r_t, fu_t in zip(df['locoregionalcontrol_duration'], 
                                    df['localcontrol_duration'],
                                    df['regionalcontrol_duration'],
                                    df['daystolastfu']):
        if lr_t != 100:
            lr_time = lr_t
        elif lr_t == 100:
            if l_t != 100:
                lr_time = l_t
            elif l_t == 100 and r_t != 100:
                lr_time = r_t
            elif l_t == 100 and r_t == 100:
                lr_time = fu_t
        lr_times.append(lr_time)
        lr_times = [int(x) for x in lr_times]
     
    # distant control event and duration time
    #---------------------------------------
    """
    Distant recurence labeled as event "1", 
    distant control labeled as no event "0";
    """
    df['distantcontrol'].fillna(value=100, inplace=True)
    df['distantcontrol'] = df['distantcontrol'].astype('str')
    df['distantcontrol'] = df['distantcontrol'].replace({'1': 'Yes', '0': 'No'})
    df['distantcontrol'] = df['distantcontrol'].replace({'Yes': 0, 'No': 1})
    df['distantcontrol'] = df['distantcontrol'].astype(float)
    df['siteofrecurrence(distal/local/locoregional)'] = df['siteofrecurrence(distal/local/locoregional)'].astype('str')
    ## events
    ds_events = []
    for ds, x in zip(df['distantcontrol'], 
                     df['siteofrecurrence(distal/local/locoregional)']):
        if ds == 1:
            ds_event = 1
        elif ds == 0:
            ds_event = 0
        elif ds == 100:
            if x in ['Distant metastasis', 
                     'Locoregional and distant metastasis', 
                     'Regional recurrence and distant metasatsis',
                     'Local recurrence and distant metastasis',
                     'Regional and distant metastasis']:
                ds_event = 1
            else:
                ds_event = 0
        ds_events.append(ds_event)
    print('ds events:', ds_events)
    
    ## duration time
    df['distantcontrol_duration'].fillna(value=100, inplace=True)
    df['disease-freeinterval(months)'].fillna(value=100, inplace=True)
    ds_times = []
    for duration, fu, ds, x in zip(df['distantcontrol_duration'], 
                                   df['daystolastfu'],
                                   df['distantcontrol'],
                                   df['disease-freeinterval(months)']):
        if ds != 100:
            if duration == 100:
                ds_time = fu
            elif duration != 100:
                ds_time = duration
        elif ds == 100:
            ds_time = x * 30
        ds_times.append(ds_time)
        ds_times = [int(x) for x in ds_times]
    #print('ds_durations:', ds_durations)
    
    # overall survival event and duration time
    #-----------------------------------------
    """
    Death labeled as event "1", survive labeled as no event "0"
    """
    # events
    df['death'].fillna(value=100, inplace=True)
    df['death'] = df['death'].astype(float)
    df['vitalstatus1'].fillna(value=100, inplace=True)
    df['vitalstatus2'].fillna(value=100, inplace=True)
    df['vitalstatus1'] = df['vitalstatus1'].replace({'Dead': 1, 'Alive': 0})
    df['vitalstatus2'] = df['vitalstatus2'].replace({'Dead': 1, 'Alive': 0})
    df['vitalstatus1'] = df['vitalstatus1'].astype(float)
    df['vitalstatus2'] = df['vitalstatus2'].astype(float)
    death_events = []
    for death, vital1, vital2 in zip(df['death'], 
                                     df['vitalstatus1'], 
                                     df['vitalstatus2']):
        if death != 100:
            death_event = death
        elif death == 100:
            if vital1 != 100 and vital2 == 100:
                death_event = vital1
            elif vital1 == 100 and vital2 != 100:
                death_event = vital2
            elif vital1 == 100 and vital2 == 100:
                print('missing death info')
        death_events.append(death_event)
        death_events = [int(x) for x in death_events]
    print('death_events:', death_events)
    
    # time
    death_times = []
    df['daystolastfu'] = df['daystolastfu'].astype(float)
    df['overallsurvival_duration'].fillna(value=100, inplace=True)
    df['overallsurvival_duration'] = df['overallsurvival_duration'].astype(float)
    for fu_t, os_t in zip(df['daystolastfu'], df['overallsurvival_duration']):
        if os_t != 100:
            t = os_t
        else:
            t = fu_t
        death_times.append(t)
        death_times = [int(x) for x in death_times]

    # HPV, smoking, stages, age, gender
    #-----------------------------------------
    # HPV status
    hpvs = []
    df['hpv'] = df.iloc[:, 8].astype(str) + df.iloc[:, 9].astype(str)
    for hpv in df['hpv']:
        if hpv in ['nannan', 'Unknownnan', 'Nnan', 'Not testednan', 'no tissuenan']:
            hpv = 'unknown'
        elif hpv in ['  positivenan', 'Pnan', '+nan', 'nanpositive', 'Positivenan', 
                     'Positive -Strongnan', 'Positive -focalnan']:
            hpv = 'positive'
        elif hpv in ['  Negativenan', 'Negativenan', '-nan', 'nannegative']:
            hpv = 'negative'
        hpvs.append(hpv)
    #print('hpv pos:', hpvs.count('positive'), hpvs.count('positive')/len(hpvs))
    #print('hpv negative:', hpvs.count('negative'), hpvs.count('negative')/len(hpvs))
    #print('hpv unknown:', hpvs.count('unknown'), hpvs.count('unknown')/len(hpvs))
    
    # smoking status
    # 0: non smoker; 1: former smoker; 2: current smoker;
    # convert nan to str first;
    df.iloc[:, 62].fillna(10, inplace=True)
    df.iloc[:, 63].fillna(10, inplace=True)
    df['smoke'] = df.iloc[:, 62].astype(float) + df.iloc[:, 63].astype(float)
    df.replace({10: 0, 11: 1, 12: 2}, inplace=True)
    smokes = df['smoke'].to_list()
    #print(smokes)

    # overall stage
    stages = []
    for stage in df['ajccstage']:
        if stage in ['I', 'Stade I']:
            stage = 'I'
        elif stage in ['II', 'Stade II', 'StageII']:
            stage = 'II'
        elif stage in ['III', 'Stade III', 'Stage III']:
            stage = 'III'
        elif stage in ['IVA', 'IV', 'IVB', 'Stade IVA', 'Stage IV', 'Stade IVB']:
            stage = 'IV'
        stages.append(stage)
    
    #print('stage I:', stages.count('I'), stages.count('I')/len(stages))
    #print('stage II:', stages.count('II'), stages.count('II')/len(stages))
    #print('stage III:', stages.count('III'), stages.count('III')/len(stages))
    #print('stage IV:', stages.count('IV'), stages.count('IV')/len(stages))

    # sex
    df['gender'].replace(['F'], 'Female', inplace=True)
    df['gender'].replace(['M'], 'Male', inplace=True)
    genders = df['gender'].to_list()

    # age
    ages = df['ageatdiag'].to_list()
    #print('age range:', np.max(ages), np.min(ages), np.median(ages))
    
    # therapeutic combination
    #print(df['therapeuticcombination'].value_counts())
    #print(df['therapeuticcombination'].value_counts(normalize=True))
    
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
        'death_time': death_times,
        'hpv': hpvs,
        'smoke': smokes,
        'stage': stages,
        'gender': genders,
        'age': ages,
        })
    
    # save labels to csv files
    df.to_csv(os.path.join(pro_data_dir, save_label), index=False)
    df.to_csv(os.path.join(output_dir, 'tot_label.csv'), index=False)
    print('successfully save label file in csv format!')
    
    df = df.loc[df['group_id'].isin(['PMH', 'MDACC'])]
    #print(df['stage'].value_counts())
    #print(df['stage'].value_counts(normalize=True))
    #print(df['hpv'].value_counts())
    #print(df['hpv'].value_counts(normalize=True))



