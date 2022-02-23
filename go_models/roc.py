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



def roc(proj_dir, out_dir):

    output_dir = os.path.join(out_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(pro_data_dir):
        os.mkdir(pro_data_dir)

    surv = pd.read_csv(os.path.join(pro_data_dir, 'surv.csv'))
    prob_scores = surv.mean(axis=0).to_list()
    median_score = np.median(prob_scores)
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    df_val['score'] = prob_scores
    #df1 = df_val[df_val['score'] >= median_score]
    #df2 = df_val[df_val['score'] < median_score]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    threshold = dict()

    ### calculate auc
    death_score = [1 - s for s in df_val['score']]
    fpr, tpr, threshold = roc_curve(
        df_val['death_event'], 
        death_score
        )
    roc_auc = auc(fpr, tpr)
    roc_auc = np.around(roc_auc, 3)
    print('ROC AUC:', roc_auc)

    """plot roc
    """
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


if __name__ == '__main__':

    proj_dir = '/mnt/HDD_6TB/HN_Outcome'
    out_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'

    roc(proj_dir, out_dir)



