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
from lifelines.utils import concordance_index
import statsmodels.api as sm
from sklearn.preprocessing import minmax_scale


def coxph_model(opt, score_type, hpv, surv_type, cox_variable, normalize):
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
    data_dir = opt.data_dir + '/' + opt.img_size + '_' + opt.img_type 
    
    clinical_list = ['Female', 'Age>65', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV']
    
    dfs = []
    for dataset in ['ts', 'tx_maastro', 'tx_bwh']:
        print('dataset:', dataset)
        save_dir = task_dir + '/' + dataset
        fn = dataset + '_img_label_' + opt.tumor_type + '.csv'
        df = pd.read_csv(data_dir + '/' + fn)
        times, events = surv_type + '_time', surv_type + '_event'

        # nan_values = df[pd.isnull(df).any(axis=1)]
        # print("Rows with NaN values:")
        # print(nan_values)

        # calculate probility score for each patient
        surv = pd.read_csv(save_dir + '/surv.csv')
        if score_type == 'mean_surv':
            dl_scores = surv.mean(axis=0).to_list()
        elif score_type == 'median_surv':
            dl_scores = surv.median(axis=0).to_list()
        elif score_type == '3yr_surv':
            # choose 5th row if number of durations is 20
            dl_scores = surv.iloc[2, :].to_list()
        elif score_type == '5yr_surv':
            dl_scores = surv.iloc[3, :].to_list()
        elif score_type == 'os_surv':
            dl_scores = surv.iloc[9, :].to_list()
        # add DL scores to df
        df['dl_score'] = dl_scores

        df = df.dropna(subset=[times, events, 'Muscle_Area', 'Muscle_Density', 'Adipose_Area', 'Adipose_Density', 
            'Female', 'Age>65', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV'])
        # # divide dl scores to 2 groups
        # dls = []
        # for s in df['dl_score']:
        #     if s < df['dl_score'].median():
        #         dl = 0
        #         dls.append(dl)
        #     else:
        #         dl = 1
        #         dls.append(dl)
        # df['dl_score'] = dls 

        # subgroup analysis based on HPV status
        if hpv == 'hpv+':
            df = df.loc[df['HPV'].isin([1])]
            #df0 = df0.loc[df0['hpv'].isin([1])]
            print('patient n = ', df.shape[0])
        elif hpv == 'hpv-':
            df = df.loc[df['HPV'].isin([0])]
            #df0 = df0.loc[df0['hpv'].isin([0])]
            print('patient n = ', df.shape[0])
        elif hpv == 'hpv':
            # patients with known HPV status
            df = df.loc[df['HPV'].isin([0, 1])]
            #df0 = df0.loc[df0['hpv'].isin([0, 1])]
            print('patient n = ', df.shape[0])
        elif hpv == 'all':
            print('patient n = ', df.shape[0])

        #--------------------------------------
        # CoxPH analysis for different models
        #--------------------------------------
        clinical_list = ['Female', 'Age>65', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV']
        if cox_variable == 'dl':
            df_cox = df[[times, events, 'dl_score']]
        elif cox_variable == 'clinical':
            df_cox = df[[times, events] + clinical_list].astype(float)
        elif cox_variable == 'clinical_dl':
            df_cox = df[[times, events, 'dl_score'] + clinical_list].astype(float)
        # elif cox_variable == 'clinical_sarcopenia':
        #     df_cox = df[[times, events, 'Sarcopenia'] + clinical_list]
        # elif cox_variable == 'clinical_sarcopenia_adipose':
        #     df_cox = df[[times, events, 'Sarcopenia', 'Adipose_Density', 'Adipose_Area'] + clinical_list]
        # elif cox_variable == 'clinical_dl_sarcopenia':
        #     df_cox = df[[times, events, 'dl_score', 'Sarcopenia'] + clinical_list]  
        # elif cox_variable == 'clinical_dl_sarcopenia_adipose':
        #     df_cox = df[[times, events, 'dl_score', 'Sarcopenia', 'Adipose_Density', 'Adipose_Area'] + clinical_list]   
        elif cox_variable == 'clinical_muscle':
            if normalize:
                df_cox = df[['Muscle_Area', 'Muscle_Density'] + clinical_list]   
                df_cox[:] = minmax_scale(df_cox)
                df_cox[times], df_cox[events] = [df[times], df[events]]
            else:
                df_cox = df[[times, events, 'Muscle_Area', 'Muscle_Density'] + clinical_list]
        elif cox_variable == 'clinical_muscle_adipose':
            if normalize:
                df_cox = df[['Muscle_Area', 'Muscle_Density', 'Adipose_Area', 'Adipose_Density'] + clinical_list]   
                df_cox[:] = minmax_scale(df_cox)
                df_cox[times], df_cox[events] = [df[times], df[events]]
            else:
                df_cox = df[[times, events, 'Muscle_Area', 'Muscle_Density', 'Adipose_Area', 'Adipose_Density'] + clinical_list]
        elif cox_variable == 'clinical_dl_muscle':
            if normalize:
                df_cox = df[['dl_score', 'Muscle_Area', 'Muscle_Density'] + clinical_list]   
                df_cox[:] = minmax_scale(df_cox)
                df_cox[times], df_cox[events] = [df[times], df[events]]
            else:
                df_cox = df[[times, events, 'dl_score', 'Muscle_Area', 'Muscle_Density'] + clinical_list]  
        elif cox_variable == 'clinical_dl_muscle_adipose':
            if normalize:
                df_cox = df[['dl_score', 'Muscle_Area', 'Muscle_Density', 'Adipose_Area', 'Adipose_Density'] + clinical_list]   
                nan_values = df_cox[pd.isnull(df_cox).any(axis=1)]
                print("Rows with NaN values:")
                print(nan_values)
                df_cox[:] = minmax_scale(df_cox)
                df_cox[times], df_cox[events] = [df[times], df[events]]
            else:
                df_cox = df[[times, events, 'dl_score', 'Muscle_Area', 'Muscle_Density', 'Adipose_Area', 'Adipose_Density'] + clinical_list]   
            print('df_cox:', df_cox)
        elif cox_variable == 'clinical_adipose':
            if normalize:
                df_cox = df[['Adipose_Area', 'Adipose_Density'] + clinical_list]   
                df_cox[:] = minmax_scale(df_cox)
                df_cox[times], df_cox[events] = [df[times], df[events]]
            else:
                df_cox = df[[times, events, 'Adipose_Area', 'Adipose_Density'] + clinical_list]   
            print('df_cox:', df_cox)       
        dfs.append(df_cox)

    df_tr = dfs[0]
    df_ts1 = dfs[1]
    df_ts2 = dfs[2]

    print('coxph df:')
    cph = CoxPHFitter()
    cph.fit(df_tr, duration_col=times, event_col=events)
    cph.print_summary()

    for df_ts in [df_ts1, df_ts2]:
        #print(cph.score(df_ts, scoring_method='concordance_index'))
        loglik = cph.score(df_ts, scoring_method='log_likelihood')
        #loglik = -149.42
        n = df_tr.shape[0]
        i = df.shape[1]

        surv = cph.predict_survival_function(df_ts)
        mean_surv = surv.mean(axis=0).to_list()
        print('surv_prob:', surv)
        #print('mean_surv:', mean_surv)

        # Calculate AIC and BIC
        aic = cph.AIC_partial_
        #bic = cph.BIC_partial_

        print(f"AIC: {aic}")
        #print(f"BIC: {bic}")

        AIC = sm.tools.eval_measures.aic(loglik, n, i)
        BIC = sm.tools.eval_measures.bic(loglik, n, i)
        print('sm AIC:', round(AIC, 2))
        print('sm BIC:', round(BIC, 2))

        print('log likelihood:', round(loglik, 2))
        AIC = round(2*df_tr.shape[0] - 2*loglik, 2)
        BIC = round(-2*loglik + df.shape[1]*(np.log(df_tr.shape[0]) - np.log(2*np.pi)), 2)

        print('AIC:', AIC)
        print('BIC:', BIC)

        # c-index on val/test set
        c_index = concordance_index(df_tr[times], -cph.predict_partial_hazard(df_tr), df_tr[events])
        print('tr c-index:', round(c_index, 2))

        c_index = concordance_index(df_ts[times], -cph.predict_partial_hazard(df_ts), df_ts[events])
        print('ts c-index:', round(c_index, 2))
    
    
if __name__ == '__main__':

    opt = parse_opts()
    score_type = 'mean_surv'
    surv_type = 'os'
    cox_variable = 'clinical_dl_muscle_adipose'
    normalize = True
    #cox_variable = 'clinical+dl+sarcopenia'
    for hpv in ['all']:    
        #for score_type in ['median_surv', 'mean_surv', '3yr_surv', '5yr_surv', 'os_surv']:
        coxph_model(opt, score_type, hpv, surv_type, cox_variable, normalize)


