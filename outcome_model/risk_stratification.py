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


def risk_strat(opt, dataset, score_type, hpv, n_group, surv_type, cluster_model, random_state):
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

    #---------------------------------------------
    # combine DL scores with clinical info
    #---------------------------------------------
    surv = pd.read_csv(save_dir + '/surv.csv')

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

    #----------------------------------------
    # K-mean clustering to find 3 risk groups      
    #----------------------------------------
    print('clustering patients into 3 risk groups')
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
        df2 = df[['Age>65', 'Female', 'T-Stage-1234', 'N-Stage-0123', 'Smoking>10py', 'Muscle_Area', 
            'Muscle_Density', 'Adipose_Area', 'Adipose_Density']].reset_index()
        df3 = pd.concat([df3, df2], axis=1)
        print('df3:', df3)
    # if hpv != 'hpv' or cluster_model == 'dl':
    #     #df.drop('hpv', axis=1, inplace=True)
    #     df3.drop('HPV', axis=1, inplace=True)
    
    x = StandardScaler().fit_transform(df3.values)
    k_means = KMeans(
        n_clusters=n_group, 
        copy_x=True, 
        init='k-means++', 
        n_init='auto', 
        max_iter=500,
        random_state=random_state, 
        tol=0.0001, 
        verbose=0)
    k_means.fit(x)
    y = k_means.predict(x)
    #print(groups)
    df['group'] = y
    #print(df)

    # plot clustering maps
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='viridis')
    centers = k_means.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.savefig(save_dir + '/clustering.png', format='png', dpi=300)
    plt.close() 
    print('save clustering figure')

    #----------------------------------------------
    # multivariate log-rank test for 3 risk groups
    #----------------------------------------------
    results = multivariate_logrank_test(df[times], df['group'], df[events])
    #results.print_summary()
    p_value = np.around(results.p_value, 8)
    print('log-rank test p-value:', p_value)
    #print(results.test_statistic)

    #----------------------------------------------
    # 5-year survival rates for 3 risk groups
    #----------------------------------------------
    dfs = []
    for i in range(n_group):
        df3 = df.loc[df['group'] == i]
        print('df:', i, df3.shape[0])
        dfs.append(df3)
        #print('df0:', dfs)

    # 5-year OS for subgroups
    for i in range(n_group):
        ls_event = []
        for time, event in zip(dfs[i][times], dfs[i][events]):
            if time <= 1825 and event == 1:
                event = 1
                ls_event.append(event)
        #print('dfs[i]:', dfs[i])
        os_5yr = round(1 - len(ls_event)/dfs[i].shape[0], 3)
        print('5-year survial rate:', i, os_5yr)

    #--------------------------------------------------------------
    # Kaplan-Meier plots for 3 risk groups 
    #--------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    labels = ['I', 'II', 'III', 'IV']
    #labels = ['Low risk', 'High risk', 'Intermediate risk']
    #dfs = [df0, df1, df2]
    for df, label in zip(dfs, labels):
        kmf = KaplanMeierFitter()
        #print(df[times])
        #print(df[events])
        #print('df:', df)
        kmf.fit(df[times], df[events], label=label)
        #kmf.fit(df['rfs_time'], df['rfs_event'], label=label)
        ax = kmf.plot_survival_function(ax=ax, show_censors=True, ci_show=True) #,censor_style={"marker": "o", "ms": 60})
        #add_at_risk_counts(kmf, ax=ax)
        median_surv = kmf.median_survival_time_
        median_surv_CI = median_survival_times(kmf.confidence_interval_)
        print('median survival time:', median_surv)
        #print('median survival time 95% CI:\n', median_surv_CI)
    
    plt.xlabel('Time (days)', fontweight='bold', fontsize=12)
    plt.ylabel('Survival probability', fontweight='bold', fontsize=12)
    plt.xlim([0, 5000])
    plt.ylim([0, 1.05])
    #ax.patch.set_facecolor('gray')
    ax.axhline(y=0, color='k', linewidth=2)
    ax.axhline(y=1.05, color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axvline(x=5000, color='k', linewidth=2)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000], fontsize=12, fontweight='bold')
    #plt.xticks([0, 500, 1000, 1500, 2000], fontsize=12, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12, fontweight='bold')
    plt.legend(loc='lower left', prop={'size': 12, 'weight': 'bold'})
    #plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', mode="expand", 
    #           borderaxespad=0, ncol=3, prop={'size': 12, 'weight': 'bold'})
    #plt.grid(True)
    plt.title('Log-Rank Test: p = %s' % p_value, fontsize=16, fontweight='bold')
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.savefig(save_dir + '/KM_' + score_type + '_' + hpv + '.jpg', format='png', dpi=300)
    #plt.savefig(save_dir + 'KM.jpg', format='png', dpi=300)
    plt.close()
    #print('saved Kaplan-Meier curve!')
    
    
if __name__ == '__main__':

    opt = parse_opts()

    dataset = 'ts'
    #dataset = 'tx_bwh'
    score_type = 'mean_surv'
    surv_type = 'efs'
    n_group = 3
    cluster_model = 'dl_clinical_muscle_adipose'
    random_state = 10
    #random_state = 1234
    #for hpv in ['all', 'pos', 'neg', 'no']:
    for hpv in ['all']:    
        #for score_type in ['median_surv', 'mean_surv', '3yr_surv', '5yr_surv', 'os_surv']:
        risk_strat(opt, dataset, score_type, hpv, n_group, surv_type, cluster_model, random_state)





    # # calculate probility score for each patient
    # #--------------------------------------------
    # surv = pd.read_csv(save_dir + '/surv.csv')
    # if score_type == 'mean_surv':
    #     prob_scores = surv.mean(axis=0).to_list()
    # elif score_type == 'median_surv':
    #     prob_scores = surv.median(axis=0).to_list()
    # elif score_type == '3yr_surv':
    #     # choose 5th row if number of durations is 20
    #     prob_scores = surv.iloc[4, :].to_list()
    # elif score_type == '5yr_surv':
    #     prob_scores = surv.iloc[7, :].to_list()
    # elif score_type == 'os_surv':
    #     prob_scores = surv.iloc[19, :].to_list()
