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
from sklearn.utils import resample
from sklearn import metrics
from roc_utils import *



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


def roc_plot(y_true, y_pred, save_dir):

    # use roc_utils package
    rocs = compute_roc_bootstrap(X=y_pred, y=y_true, pos_label=True,
                                 n_bootstrap=1000,
                                 random_state=42,
                                 return_mean=False)
    plot_mean_roc(rocs, show_ci=True, show_ti=True, color='white', alpha=0)
    plt.legend(loc='lower right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    # save_path = save_dir + '/5yr_roc_.png'
    # plt.savefig(save_path, format='png', dpi=600)

    roc = compute_roc(X=y_pred, y=y_true, pos_label=True,
                      objective=['minopt', 
                                 'minoptsym', 
                                 'youden', 
                                 'cost',
                                 'concordance',
                                 #"lr+", "lr-",    # Possibly buggy
                                 #"dor", "chi2",   # Possibly buggy
                                 'acc', 
                                 'cohen'
                                ])
    for key, val in roc.opd.items():
        print('%-15s thr=% .3f, J=%7.3f' % (key+':', val.opt, val.opo) )
        
    plot_roc(roc, show_opt=True, alpha=1)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    # plt.xlim([-0.03, 1])
    # plt.ylim([0, 1.03])
    # ax.axhline(y=0, color='k', linewidth=2)
    # ax.axhline(y=1.03, color='k', linewidth=2)
    # ax.axvline(x=-0.03, color='k', linewidth=2)
    # ax.axvline(x=1, color='k', linewidth=4)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=10, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=10, fontweight='bold')
    plt.xlabel('FPR (False Positive Rate)', fontweight='bold', fontsize=11)
    plt.ylabel('TPR (True Positive Rate)', fontweight='bold', fontsize=11)
    #plt.title('5 Year Overall Survival', fontsize=12, fontweight='bold')
    #plt.legend(loc='lower right', prop={'size': 14, 'weight': 'bold'})
    plt.grid(True)
    #plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    save_path = save_dir + '/5yr_roc.png'
    plt.savefig(save_path, format='png', dpi=600)



def get_CI(y_true, y_pred, n_bootstrap=1000):

    aucs = []
    tnrs = []
    tprs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_pred), size=len(y_pred), replace=True)
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
        aucs.append(roc_auc_score(np.array(y_true)[indices], np.array(y_pred)[indices]))
        tprs.append(roc_opt['tpr'])
        tnrs.append(roc_opt['tnr'])
    ## calculate median and 95% CI
    auc_CI = np.percentile(aucs, [2.5, 97.5])
    tpr_CI = np.percentile(tprs, [2.5, 97.5])
    tnr_CI = np.percentile(tnrs, [2.5, 97.5])
    auc_med = np.median(aucs)
    tpr_med = np.median(tprs)
    tnr_med = np.median(tnrs)

    AUC = np.around((auc_med, auc_CI[0], auc_CI[1]), 3)
    TPR = np.around((tpr_med, auc_CI[0], tpr_CI[1]), 3)
    TNR = np.around((tnr_med, auc_CI[0], tnr_CI[1]), 3)
    # AUC = mean_CI(aucs)
    # TPR = mean_CI(tprs)
    # TNR = mean_CI(tnrs)

    ### save results into dataframe
    stat_roc = pd.DataFrame([AUC, TPR, TNR],
                            columns=['mean', '95% CI -', '95% CI +'],
                            index=['AUC', 'TPR', 'TNR'])
    print(stat_roc)


def get_roc(opt, dataset, score_type, hpv, surv_type, cox_variable):

    opt.task = 'Task053'
    opt.surv_type = 'os'

    task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_size + '_' + \
        opt.img_type + '_' + opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 
            
    data_dir = task_dir + '/' + dataset 
    save_dir = data_dir + '/' + cox_variable
    
    fn = dataset + '_img_label_' + opt.tumor_type + '.csv'
    df = pd.read_csv(data_dir + '/' + fn)

    times, events = surv_type + '_time', surv_type + '_event'
    df = df.dropna(subset=[times, events])

    # calculate probility score for each patient
    surv = pd.read_csv(save_dir + '/raw_surv_2.csv')
    print(surv)

    if score_type == 'mean_surv':
        dl_scores = surv.mean(axis=0).to_list()
    elif score_type == 'median_surv':
        dl_scores = surv.median(axis=0).to_list()
    elif score_type == '3yr_surv':
        # choose 5th row if number of durations is 20
        dl_scores = surv.iloc[2, :].to_list()
    elif score_type == '5yr_surv':
        dl_scores = surv.iloc[4, :].to_list()
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
    y_true = df[events].to_list()
    #y_true = df['5yr_event'].to_list()

    roc_plot(y_true, y_pred, save_dir)

    get_CI(y_true, y_pred)


if __name__ == '__main__':

    opt = parse_opts()
    dataset = 'tx_bwh'
    score_type = '5yr_surv'
    surv_type = 'os'
    cox_variable = 'tot'
    hpv = 'all'
    n_bs = 1000

    get_roc(opt, dataset, score_type, hpv, surv_type, cox_variable)





# def roc_plot(y_true, y_pred, save_dir):

#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     fpr, tpr, _ = roc_curve(y_true, y_pred)
#     roc_auc = auc(fpr, tpr)
#     roc_auc = np.around(roc_auc, 3)
#     print('ROC AUC:', roc_auc)

#     # plot roc
#     fig = plt.figure()
#     ax  = fig.add_subplot(1, 1, 1)
#     ax.set_aspect('equal')
#     plt.plot(fpr, tpr, color='red', linewidth=2, label='AUC %0.3f' % roc_auc)
#     plt.xlim([-0.03, 1])
#     plt.ylim([0, 1.03])
#     ax.axhline(y=0, color='k', linewidth=2)
#     ax.axhline(y=1.03, color='k', linewidth=2)
#     ax.axvline(x=-0.03, color='k', linewidth=2)
#     ax.axvline(x=1, color='k', linewidth=4)
#     plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
#     plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
#     plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
#     plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
#     plt.legend(loc='lower right', prop={'size': 14, 'weight': 'bold'})
#     plt.grid(True)
#     plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
#     save_path = save_dir + '/5yr_roc.png'
#     plt.savefig(save_path, format='png', dpi=600)
#     #plt.show()
#     plt.close()
#     print('saved roc curves!')