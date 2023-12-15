import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import glob
import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from time import localtime, strftime
from sklearn.utils import resample
import scipy.stats as ss
from sklearn.preprocessing import minmax_scale
from opts import parse_opts


def mean_CI(data):
    """
    Calculate mean value and 95% CI
    """
    mean = np.mean(np.array(data))
    CI = ss.t.interval(
        confidence=0.95,
        df=len(data)-1,
        loc=np.mean(data),
        scale=ss.sem(data))
    lower = CI[0]
    upper = CI[1]
    return mean, lower, upper


def plot_roc(y_true, y_pred, save_dir):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_auc = np.around(roc_auc, 3)
    print('ROC AUC:', roc_auc)

    # plot roc
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.plot(fpr, tpr, color='red', linewidth=3, label='AUC %0.3f' % roc_auc)
    plt.xlim([-0.03, 1])
    plt.ylim([0, 1.03])
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=1.03, color='k', linewidth=4)
    ax.axvline(x=-0.03, color='k', linewidth=4)
    ax.axvline(x=1, color='k', linewidth=4)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    plt.xlabel('1 - Specificity', fontweight='bold', fontsize=16)
    plt.ylabel('Sensitivity', fontweight='bold', fontsize=16)
    plt.legend(loc='lower right', prop={'size': 16, 'weight': 'bold'})
    plt.grid(True)
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    save_path = save_dir + '/5yr_roc.png'
    plt.savefig(save_path, format='png', dpi=600)
    #plt.show()
    plt.close()
    print('saved roc curves!')


def get_CI(y_true, y_pred, n_bootstrap=1000):

    AUC = []
    THRE = []
    TNR = []
    TPR = []
    for j in range(n_bootstrap):
        #print("bootstrap iteration: " + str(j+1) + " out of " + str(n_bootstrap))
        index = range(len(y_pred))
        indices = resample(index, replace=True, n_samples=int(len(y_pred)))
        fpr, tpr, thre = roc_curve(np.array(y_true)[indices], np.array(y_pred)[indices])
        q = np.arange(len(tpr))
        roc = pd.DataFrame(
            {'fpr': pd.Series(fpr, index=q),
             'tpr': pd.Series(tpr, index=q),
             'tnr': pd.Series(1 - fpr, index=q),
             'tf': pd.Series(tpr - (1 - fpr), index=q),
             'thre': pd.Series(thre, index=q)})
        ### calculate optimal TPR, TNR under uden index
        roc_opt = roc.loc[(roc['tpr'] - roc['fpr']).idxmax(),:]
        AUC.append(roc_auc_score(np.array(y_true)[indices], np.array(y_pred)[indices]))
        TPR.append(roc_opt['tpr'])
        TNR.append(roc_opt['tnr'])
        THRE.append(roc_opt['thre'])
    ### calculate mean and 95% CI
    AUCs = np.around(mean_CI(AUC), 3)
    TPRs = np.around(mean_CI(TPR), 3)
    TNRs = np.around(mean_CI(TNR), 3)
    THREs = np.around(mean_CI(THRE), 3)
    #print(AUCs)
    ### save results into dataframe
    stat_roc = pd.DataFrame(
        [AUCs, TPRs, TNRs, THREs],
        columns=['mean', '95% CI -', '95% CI +'],
        index=['AUC', 'TPR', 'TNR', 'THRE'])
    print(stat_roc)

   

def get_roc(opt, dataset, score_type, hpv, surv_type, cox_variable, normalize):

    task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_size + '_' + \
        opt.img_type + '_' + opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 
    data_dir = opt.data_dir + '/' + opt.img_size + '_' + opt.img_type 
    save_dir = task_dir + '/' + dataset
    
    fn = dataset + '_img_label_' + opt.tumor_type + '.csv'
    df = pd.read_csv(data_dir + '/' + fn)
    times, events = surv_type + '_time', surv_type + '_event'
    df = df.dropna(subset=[times, events])

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
    clinical_list = ['Female', 'Age>65', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123']
    if cox_variable == 'dl':
        df_cox = df[[times, events, 'dl_score']]
    elif cox_variable == 'clinical':
        df_cox = df[[times, events] + clinical_list].astype(float)
    elif cox_variable == 'clinical_dl':
        df_cox = df[[times, events, 'dl_score'] + clinical_list].astype(float)  
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

    df = df_cox
    # calculate 5-year os AUC
    #--------------------------
    _events = []
    _times = []
    for time, event in zip(df[times], df[events]):
        if time < 1825 and event == 1:
            event = 1
            time = time
        else:
            event = 0
            if time < 1825:
                time = time
            elif time > 1825:
                time = 1825
        _events.append(event)
        _times.append(time)
    df['5yr_time'] = _times
    df['5yr_event'] = _events

    y_pred = [1 - s for s in df['dl_score']]
    y_true = df['5yr_event'].to_list()

    plot_roc(y_true, y_pred, save_dir)

    get_CI(y_true, y_pred)




if __name__ == '__main__':

    opt = parse_opts()
    dataset = 'ts'
    score_type = 'mean_surv'
    surv_type = 'efs'
    cox_variable = 'clinical_dl_muscle_adipose'
    normalize = True
    hpv = 'all'
    n_bs = 1000

    get_roc(opt, dataset, score_type, hpv, surv_type, cox_variable, normalize)




