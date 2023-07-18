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


def coxph(proj_dir, coxph_model, hpv, surv_fn, score_type):

    output_dir = os.path.join(proj_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')

    # calculate probility score for each patient
    surv = pd.read_csv(os.path.join(pro_data_dir, surv_fn))
    if score_type == 'mean':
        prob_scores = surv.mean(axis=0).to_list()
    elif score_type == 'median':
        prob_scores = surv.median(axis=0).to_list()
    elif score_type == '3yr_surv':
        # choose 5th row if number of durations is 20
        prob_scores = surv.iloc[4, :].to_list()
    elif score_type == '5yr_surv':
        prob_scores = surv.iloc[7, :].to_list()
    elif score_type == 'os_surv':
        prob_scores = surv.iloc[19, :].to_list()
    
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    df_val['score'] = prob_scores

    df0 = df_val[['death_time', 'death_event', 'hpv', 'smoke',
                 'stage', 'gender', 'age', 'score']] 
    df0['stage'] = df0['stage'].replace({'I': 0, 'II': 1, 'III': 2, 'IV': 3})
    df0['gender'] = df0['gender'].replace({'Male': 0, 'Female': 1})
    df0['hpv'] = df0['hpv'].replace({'negative': 0, 'positive': 1, 'unknown': 2})

    # test coxph for hpv or no hpv
    if coxph_model == 'clinical':
        df = df0[['death_time', 'death_event', 'hpv', 'smoke',
                 'stage', 'gender', 'age']]
        if hpv == 'pos':
            df = df.loc[df['hpv'].isin([1])]
            df.drop('hpv', axis=1, inplace=True)
            print('patient n = ', df.shape[0])
        elif hpv == 'neg':
            df = df.loc[df['hpv'].isin([0])]
            df.drop('hpv', axis=1, inplace=True)
            print('patient n = ', df.shape[0])
        else:
            df = df
            df.drop('hpv', axis=1, inplace=True)
            print('patient n = ', df.shape[0])
    elif coxph_model == 'DL':
        df = df0[['death_time', 'death_event', 'score']]
    elif coxph_model == 'DL_clinical':
        df = df0
        if hpv == 'pos':
            df = df.loc[df['hpv'].isin([1])]
            df.drop('hpv', axis=1, inplace=True)
            print('patient n = ', df.shape[0])
        elif hpv == 'neg':
            df = df.loc[df['hpv'].isin([0])]
            df.drop('hpv', axis=1, inplace=True)
            print('patient n = ', df.shape[0])
        else:
            df = df
            df.drop('hpv', axis=1, inplace=True)
            print('patient n = ', df.shape[0])
    
    # fit data in to model
    cph = CoxPHFitter()
    cph.fit(df, duration_col='death_time', event_col='death_event')
    cph.print_summary()


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    surv_fn = 'resnet101_20_0.0001_surv.csv'
    score_type = 'os_surv'
    hpv = 'pos'
    coxph_model = 'clinical'

    coxph(proj_dir, coxph_model, hpv, surv_fn, score_type)



