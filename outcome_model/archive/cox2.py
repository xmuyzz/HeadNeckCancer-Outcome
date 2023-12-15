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
from opts import parse_opts
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from sklearn.preprocessing import minmax_scale


def risk_strat(opt, dataset, score_type, surv_type, cluster_model, random_state):
    """
    Kaplan-Meier analysis for risk group stratification
    Args:
        proj_dir {path} -- project dir;
        out_dir {path} -- output dir;
        score_type {str} -- prob scores: mean, median, 3-year survival;
        hpv {str} -- hpv status: 'pos', 'neg';
    Returns:
        KM plot, median survial time, log-rank test;
    Raise errors:
        None;
    """

    task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_size + '_' + \
        opt.img_type + '_' + opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 
    save_dir = task_dir + '/' + dataset
    data_dir = opt.data_dir + '/' + opt.img_size + '_' + opt.img_type 
    
    fn = dataset + '_img_label_' + opt.tumor_type + '.csv'
    df = pd.read_csv(data_dir + '/' + fn)
    times, events = surv_type + '_time', surv_type + '_event'
    df = df.dropna(subset=[times, events])

    surv = pd.read_csv(save_dir + '/surv.csv')


    # choose variables 
    if cluster_model == 'clinical':
        df3 = df[['Female', 'Age>65', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123']]
    elif cluster_model == 'dl':
        df3 = surv.T
        #df3['HPV'] = df['HPV'].to_list()
    elif cluster_model == 'dl_clinical':
        df3 = surv.T
        df3.columns = ['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9', 'time10']
        df3 = df3.reset_index()
        df2 = df[['Age>65', 'Female', 'T-Stage-1234', 'N-Stage-0123', 'Smoking>10py']].reset_index()
        df3 = pd.concat([df3, df2], axis=1)
        print('df3:', df3)
    elif cluster_model == 'dl_clinical_muscle':
        df3 = surv.T
        df3.columns = ['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9', 'time10']
        df3 = df3.reset_index()
        df2 = df[['Age>65', 'Female', 'T-Stage-1234', 'N-Stage-0123', 'Smoking>10py', 'Muscle_Area', 'Muscle_Density']].reset_index()
        df3 = pd.concat([df3, df2], axis=1)
        print('df3:', df3)
    elif cluster_model == 'dl_clinical_adipose':
        df3 = surv.T
        df3.columns = ['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9', 'time10']
        df3 = df3.reset_index()
        df2 = df[['Age>65', 'Female', 'T-Stage-1234', 'N-Stage-0123', 'Smoking>10py', 'Adipose_Area', 'Adipose_Density']].reset_index()
        df3 = pd.concat([df3, df2], axis=1)
        print('df3:', df3)       
    elif cluster_model == 'dl_clinical_muscle_adipose':
        df3 = surv.T
        df3.columns = ['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9', 'time10']
        df3 = df3.reset_index()
        df2 = df[['HPV', 'Age>65', 'Female', 'T-Stage-1234', 'N-Stage-0123', 'Smoking>10py', 'Muscle_Area', 
            'Adipose_Area', 'Adipose_Density']].reset_index()
        df3 = pd.concat([df3, df2], axis=1)
        df3[:] = minmax_scale(df3)
        df3[times], df3[events] = [df[times], df[events]]
        df3 = df3.drop('index', axis=1)
        df3.to_csv(save_dir + '/df3.csv', index=False)
        print('df3:', df3)

    #df3 = pd.read_csv(save_dir + '/df3.csv')
    print('coxph df:')
    cph = CoxPHFitter()
    cph.fit(df3, duration_col=times, event_col=events)
    cph.print_summary()

    
    
if __name__ == '__main__':

    opt = parse_opts()

    dataset = 'ts'
    #dataset = 'tx_bwh'
    score_type = 'mean_surv'
    surv_type = 'efs'
    cluster_model = 'dl_clinical_muscle_adipose'
    random_state = 10
    #random_state = 1234

    risk_strat(opt, dataset, score_type, surv_type, cluster_model, random_state)






