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


def kmf_risk_strat(opt, score_type, hpv):
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
    csv_dir = opt.proj_dir + '/data/csv_file'
    log_dir = opt.proj_dir + '/log/' + opt.task + '_' + opt.tumor_type + '_' + opt.cnn_name + str(opt.model_depth) + '/cindex_0.69'
    if opt.run_type == 'val':
        save_dir = log_dir + '/va'
    elif opt.run_type == 'test':
        save_dir = log_dir + '/ts'
    elif opt.run_type == 'external1':
        save_dir = log_dir + '/tx1'
    elif opt.run_type == 'external2':
        save_dir = log_dir + '/tx2'
        
    # calculate probility score for each patient
    #--------------------------------------------
    surv = pd.read_csv(save_dir + '/surv.csv')
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

    if opt.tumor_type == 'pn':
        tr_fn = 'tr_img_label_pn.csv'
        va_fn = 'va_img_label_pn.csv'
        ts_fn = 'ts_img_label_pn.csv'
    if opt.tumor_type == 'p':
        tr_fn = 'tr_img_label_p.csv'
        va_fn = 'va_img_label_p.csv'
        ts_fn = 'ts_img_label_p.csv'
    if opt.tumor_type == 'n':
        tr_fn = 'tr_img_label_n.csv'
        va_fn = 'va_img_label_n.csv'
        ts_fn = 'ts_img_label_n.csv'
    df_tr = pd.read_csv(csv_dir + '/' + tr_fn)
    df_va = pd.read_csv(csv_dir + '/' + va_fn)
    df_ts = pd.read_csv(csv_dir + '/' + ts_fn)

    if opt.run_type == 'val':
        df = df_va
    elif opt.run_type == 'test':
        df = df_ts

    # choose task: rfs, os, lr, dr
    if opt.task == 'rfs':
        # rfs = recurence free survival
        df = df.dropna(subset=['rfs_time', 'rfs_event'])
    elif opt.task == 'os':
        # os = overall survival
        df = df.dropna(subset=['os_time', 'os_event'])
    elif opt.task == 'lr':
        # lr = local recurrence
        df = df.dropna(subset=['lr_time', 'lr_event'])
    elif opt.task == 'dr':
        # dr = distant recurrence
        df = df.dropna(subset=['dr_time', 'dr_event'])

    # get median scores to stratify risk groups
    median_score = np.median(prob_scores)
    df = df[['time', 'event', 'hpv', 'stage', 'gender', 'age']]
    df['hpv'] = df['hpv'].replace({'negative': 0, 'positive': 1, 'unknown': 2})
    #pat_id = df_val['pat_id'].to_list()
    df['score'] = prob_scores
    #print('prob_scores:', prob_scores)

    # plot histogram for scores distribution
    #---------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.hist(prob_scores, bins=20)
    plt.savefig(save_dir + '/histogram.png', format='png', dpi=300)
    plt.close() 
    
    # KMeans to cluster scores to 3 groups
    #-------------------------------------
    k_means = KMeans(n_clusters=3, algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        random_state=0, tol=0.0001, verbose=0)
    prob_scores = np.array(prob_scores).reshape(-1, 1)
    k_means.fit(prob_scores)
    groups = k_means.predict(prob_scores)
    #print(groups)
    df['group'] = groups
    
    # check hpv status
    #------------------
    if hpv == 'pos':
        df = df.loc[df['hpv'].isin([1])]
        print('patient n = ', df.shape[0])
    elif hpv == 'neg':
        df = df.loc[df['hpv'].isin([0])]
        print('patient n = ', df.shape[0])
    else:
        print('patient n = ', df.shape[0])
    
    # multivariate log-rank test
    #---------------------------
    results = multivariate_logrank_test(df['time'], df['group'], df['event'])
    #results.print_summary()
    #p_value = np.around(results.p_value, 3)
    print('log-rank test p-value:', results.p_value)
    #print(results.test_statistic)
    
    # Kaplan-Meier curve
    #---------------------
    dfs = []
    for i in range(3):
        df = df.loc[df['group'] == i]
        print('df:', i, df.shape[0])
        dfs.append(df)
    
    # 5-year OS for subgroups
    for i in range(3):
        events = []
        for time, event in zip(dfs[i]['time'], dfs[i]['event']):
            if time <= 1825 and event == 1:
                event = 1
                events.append(event)
        os_5yr = 1 - np.around(len(events)/dfs[i].shape[0], 3)
        print('5-year survial rate:', i, os_5yr)

    # Kaplan-Meier plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    labels = ['I', 'II', 'III']
    #labels = ['Low risk', 'High risk', 'Intermediate risk']
    #dfs = [df0, df1, df2]
    for df, label in zip(dfs, labels):
        kmf = KaplanMeierFitter()
        kmf.fit(df['time'], df['event'], label=label)
        ax = kmf.plot_survival_function(ax=ax, show_censors=True, ci_show=True) #,censor_style={"marker": "o", "ms": 60})
        #add_at_risk_counts(kmf, ax=ax)
        median_surv = kmf.median_survival_time_
        median_surv_CI = median_survival_times(kmf.confidence_interval_)
        print('median survival time:', median_surv)
        #print('median survival time 95% CI:\n', median_surv_CI)
    
    plt.xlabel('Time (days)', fontweight='bold', fontsize=12)
    plt.ylabel('Survival probability', fontweight='bold', fontsize=12)
    plt.xlim([0, 5000])
    plt.ylim([0, 1])
    #ax.patch.set_facecolor('gray')
    ax.axhline(y=0, color='k', linewidth=2)
    ax.axhline(y=1, color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axvline(x=5000, color='k', linewidth=2)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000], fontsize=12, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', prop={'size': 12, 'weight': 'bold'})
    #plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', mode="expand", 
    #           borderaxespad=0, ncol=3, prop={'size': 12, 'weight': 'bold'})
    #plt.grid(True)
    #plt.title('Kaplan-Meier Survival Estimate', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.savefig(save_dir + '/kaplan_meier.png', format='png', dpi=300)
    plt.close()
    #print('saved Kaplan-Meier curve!')
    
    
if __name__ == '__main__':
    opt = parse_opts()
    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    surv_fn = 'resnet101_20_0.0001_surv.csv'
    score_type = 'os_surv'
    hpv = 'all'

    kmf_risk_strat(opt, score_type, hpv)



