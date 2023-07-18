import os
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
from time import localtime, strftime
from sklearn.cluster import KMeans
from lifelines import CoxPHFitter
from opts import parse_opts


def coxph(opt, coxph_model, hpv, score_type):

    task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_type + '_' + \
            opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 
    save_dir = task_dir + '/' + opt.data_set
    data_dir = opt.proj_dir + '/data/' + opt.img_type
        
    # calculate probility score for each patient
    #--------------------------------------------
    surv = pd.read_csv(save_dir + '/surv.csv')
    #print('surv:', surv.T)
    if score_type == 'mean_surv':
        prob_scores = surv.mean(axis=0).to_list()
    elif score_type == 'median_surv':
        prob_scores = surv.median(axis=0).to_list()
    elif score_type == '3yr_surv':
        # choose 5th row if number of durations is 20
        prob_scores = surv.iloc[4, :].to_list()
    elif score_type == '5yr_surv':
        prob_scores = surv.iloc[7, :].to_list()
    elif score_type == 'os_surv':
        prob_scores = surv.iloc[19, :].to_list()

    fn = opt.data_set + '_img_label_' + opt.tumor_type + '.csv'
    df = pd.read_csv(data_dir + '/' + fn)
    times, events = surv_type + '_time', surv_type + '_event'
    df = df.dropna(subset=[times, events])
    df = df[[times, events, 'hpv', 'stage', 'gender', 'age', 'smoke']]
    df['hpv'] = df['hpv'].replace({'negative': 0, 'positive': 1, 'unknown': 2})
    df['stage'] = df['stage'].replace({'I': 0, 'II': 1, 'III': 2, 'IV': 3})
    df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1})

    # choose task: rfs, os, lr, dr
    if opt.task == 'rfs':
        # rfs = recurence free survival
        df = df.dropna(subset=['rfs_time', 'rfs_event'])
        times, events = 'rfs_time', 'rfs_event'
    elif opt.task == 'os':
        # os = overall survival
        df = df.dropna(subset=['os_time', 'os_event']).reset_index()
        times, events = 'os_time', 'os_event'
    elif opt.task == 'lr':
        # lr = local recurrence
        df = df.dropna(subset=['lr_time', 'lr_event'])
        times, events = 'lr_time', 'lr_event'
    elif opt.task == 'dr':
        # dr = distant recurrence
        df = df.dropna(subset=['dr_time', 'dr_event'])
        times, events = 'dr_time', 'dr_event'

    df0 = df[[times, events, 'hpv', 'smoke', 'stage', 'gender', 'age']] 
    #df0['score'] = prob_scores
    df0['stage'] = df0['stage'].replace({'I': 0, 'II': 1, 'III': 2, 'IV': 3})
    df0['gender'] = df0['gender'].replace({'Male': 0, 'Female': 1})
    df0['hpv'] = df0['hpv'].replace({'negative': 0, 'positive': 1, 'unknown': 2})
    #print(df0)

    df = surv.T
    #print(df)
    #print(df0)
    #df = pd.concat([df0, surv], axis=1, ignore_index=True, sort=False)
    df[times] = df0[times].to_list()
    df[events] = df0[events].to_list()
    df['score'] = prob_scores
    df['hpv'] = df0['hpv'].to_list()
    df['smoke'] = df0['smoke'].to_list()
    df['stage'] = df0['stage'].to_list()
    df['gender'] = df0['gender'].to_list()
    df['age'] = df0['age'].to_list()
    print('df:', df)

    # test coxph for hpv or no hpv
    if hpv == 'pos':
        df = df.loc[df['hpv'].isin([1])]
        df.drop('hpv', axis=1, inplace=True)
        print('patient n = ', df.shape[0])
    elif hpv == 'neg':
        df = df.loc[df['hpv'].isin([0])]
        df.drop('hpv', axis=1, inplace=True)
        print('patient n = ', df.shape[0])
    elif hpv == 'all':
        df = df
        #df.drop('hpv', axis=1, inplace=True)
        print('patient n = ', df.shape[0])
    elif hpv == 'hpv':
        df = df.loc[df['hpv'].isin([0, 1])]
        print('patient n = ', df.shape[0])

    if coxph_model == 'DL':
        #df = df[[times, events, 'score']]   
        df = df.iloc[:, :22]
        print(df)
    elif coxph_model == 'clinical':
        df = df[[times, events, 'hpv', 'smoke', 'stage', 'gender', 'age']]
    elif coxph_model == 'DL_clinical':
        df = df


    # fit data in to model
    cph = CoxPHFitter()
    #cph.fit(df, duration_col=times, event_col=events)
    cph.fit(df, duration_col=times, event_col=events)
    cph.print_summary()


if __name__ == '__main__':
    opt = parse_opts()
    score_type = 'os_surv'
    hpv = 'all'
    coxph_model = 'clinical'

    coxph(opt, coxph_model, hpv, score_type)



