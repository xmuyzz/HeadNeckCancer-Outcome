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

def mean_CI(data):

    """
    Calculate mean value and 95% CI
    """

    mean = np.mean(np.array(data))
    CI = ss.t.interval(
        alpha=0.95,
        df=len(data)-1,
        loc=np.mean(data),
        scale=ss.sem(data)
        )
    lower = CI[0]
    upper = CI[1]

    return mean, lower, upper


def roc(proj_dir, surv_fn, hpv, n_bs):
    
    """
    Get AUC, ROC curve and 95% CI

    Args:
        proj_dir {path} -- project directory;
        surv_fn {str} -- survival scores file name;
        hpv {str} -- HPV status;
    Returns:
        AUC, 95% CI AUC and ROC curve;

    """

    output_dir = os.path.join(proj_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(pro_data_dir):
        os.mkdir(pro_data_dir)
    
    surv = pd.read_csv(os.path.join(pro_data_dir, surv_fn))
    prob_scores = surv.mean(axis=0).to_list()
    median_score = np.median(prob_scores)
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    df_val['score'] = prob_scores
    df = df_val[['death_time', 'death_event', 'hpv', 'stage', 'gender', 'age', 'score']]
    df['stage'] = df['stage'].map({'I': 1, 'II': 2, 'III': 3, 'IV': 4})
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['hpv'] = df['hpv'].map({'negative': 0, 'positive': 1, 'unknown': 2})

    # test on different hpv status
    #------------------------------
    print('HPV status:', hpv)
    if hpv == 'pos':
        df = df.loc[df['hpv'] == 1]
        print('patient n = ', df.shape[0])
    elif hpv == 'neg':
        df = df.loc[df['hpv'] == 0]
        print('patient n = ', df.shape[0])
    elif hpv == 'hpv':
        df = df.loc[df['hpv'].isin([0, 1])]
        print('patient n = ', df.shape[0])
    else:
        df = df_val

    # calculate 5-year os AUC
    #--------------------------
    events = []
    times = []
    for time, event in zip(df['death_time'], df['death_event']):
        if time < 1825 and event == 1:
            event = 1
            time = time
        else:
            event = 0
            if time < 1825:
                time = time
            elif time > 1825:
                time = 1825
        events.append(event)
        times.append(time)
    df['time'] = times
    df['event'] = events

    # get AUC 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    threshold = dict()
    death_score = [1 - s for s in df['score']]
    fpr, tpr, threshold = roc_curve(
        df['event'], 
        death_score
        )
    roc_auc = auc(fpr, tpr)
    roc_auc = np.around(roc_auc, 3)
    print('ROC AUC:', roc_auc)

    # plot roc
    #----------
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.plot(fpr, tpr, color='blue', linewidth=3, label='AUC %0.3f' % roc_auc)
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
    fn = 'roc' + '_' + str(strftime('%Y_%m_%d_%H_%M_%S', localtime())) + '.png'
    plt.savefig(os.path.join(output_dir, fn), format='png', dpi=600)
    #plt.show()
    plt.close()
    print('saved roc curves!')

    # get 95% CI
    #------------
    y_true = df['event'].to_numpy()
    y_pred = np.array(death_score)
    AUC  = []
    THRE = []
    TNR  = []
    TPR  = []
    for j in range(n_bs):
        #print("bootstrap iteration: " + str(j+1) + " out of " + str(n_bootstrap))
        index = range(len(y_pred))
        indices = resample(index, replace=True, n_samples=int(len(y_pred)))
        fpr, tpr, thre = roc_curve(y_true[indices], y_pred[indices])
        q = np.arange(len(tpr))
        roc = pd.DataFrame(
            {'fpr' : pd.Series(fpr, index=q),
             'tpr' : pd.Series(tpr, index=q),
             'tnr' : pd.Series(1 - fpr, index=q),
             'tf'  : pd.Series(tpr - (1 - fpr), index=q),
             'thre': pd.Series(thre, index=q)}
             )
        ### calculate optimal TPR, TNR under uden index
        roc_opt = roc.loc[(roc['tpr'] - roc['fpr']).idxmax(),:]
        AUC.append(roc_auc_score(y_true[indices], y_pred[indices]))
        TPR.append(roc_opt['tpr'])
        TNR.append(roc_opt['tnr'])
        THRE.append(roc_opt['thre'])
    ### calculate mean and 95% CI
    AUCs  = np.around(mean_CI(AUC), 3)
    TPRs  = np.around(mean_CI(TPR), 3)
    TNRs  = np.around(mean_CI(TNR), 3)
    THREs = np.around(mean_CI(THRE), 3)
    #print(AUCs)
    ### save results into dataframe
    stat_roc = pd.DataFrame(
        [AUCs, TPRs, TNRs, THREs],
        columns=['mean', '95% CI -', '95% CI +'],
        index=['AUC', 'TPR', 'TNR', 'THRE']
        )
    print(stat_roc)



if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    surv_fn = 'resnet101_20_0.0001_surv.csv'
    hpv = 'neg'
    n_bs = 1000


    roc(
        proj_dir=proj_dir, 
        surv_fn=surv_fn, 
        hpv=hpv, 
        n_bs=n_bs
        )




