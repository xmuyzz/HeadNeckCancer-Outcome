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


def kmf_risk_strat3(proj_dir, out_dir, score_type, cnn_name, epochs,
                    n_clusters=3):

    """
    Kaplan-Meier analysis for risk group stratification

    Args:
        proj_dir {path} -- project dir;
        out_dir {path} -- output dir;
        score_type {str} -- prob scores: mean, median, 3-year survival;

    Returns:
        KM plot, median survial time, log-rank test;
    
    Raise errors:
        None;

    """


    output_dir = os.path.join(out_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir): 
        os.mkdir(output_dir)
    if not os.path.exists(pro_data_dir): 
        os.mkdir(pro_data_dir)
    
    # calculate probility score for each patient
    #--------------------------------------------
    surv = pd.read_csv(os.path.join(pro_data_dir, 'surv.csv'))
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


    # get median scores to stratify risk groups
    median_score = np.median(prob_scores)
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    #pat_id = df_val['pat_id'].to_list()
    df_val['score'] = prob_scores
    #print('prob_scores:', prob_scores)
    
    # plot histogram for scores distribution
    #---------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.hist(prob_scores, bins=20)
    plt.savefig(os.path.join(output_dir, 'histogram.png'), format='png', dpi=300)
    plt.close() 
    
    # KMeans to cluster scores to 3 groups
    #-------------------------------------
    k_means = KMeans(
        n_clusters=n_clusters,
        algorithm='auto', 
        copy_x=True, 
        init='k-means++', 
        max_iter=300,
        random_state=0, 
        tol=0.0001, 
        verbose=0
        )
    prob_scores = np.array(prob_scores).reshape(-1, 1)
    k_means.fit(prob_scores)
    groups = k_means.predict(prob_scores)
    df_val['group'] = groups

    # multivariate log-rank test
    #---------------------------
    results = multivariate_logrank_test(
        df_val['death_time'],
        df_val['group'],
        df_val['death_event'],
        )
    #results.print_summary()

    #p_value = np.around(results.p_value, 3)
    print('log-rank test p-value:', results.p_value)
    #print(results.test_statistic)

    
    # Kaplan-Meier curve
    #---------------------
    dfs = []
    for i in range(n_clusters):
        df = df_val.loc[df_val['group'] == i]
        #print('df:', df.shape[0], df)
        dfs.append(df)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #labels = ['Medium risk', 'Low risk', 'High risk']
    labels = ['Group 1', 'Group 2', 'Group 3']
    #dfs = [df0, df1, df2]
    for df, label in zip(dfs, labels):
        kmf = KaplanMeierFitter()
        kmf.fit(
            df['death_time'],
            df['death_event'],
            label=label
            )
        ax = kmf.plot_survival_function(
            ax=ax,
            show_censors=True,
            ci_show=True,
            #censor_style={"marker": "o", "ms": 60}
            )
        #add_at_risk_counts(kmf, ax=ax)
        median_surv = kmf.median_survival_time_
        median_surv_CI = median_survival_times(kmf.confidence_interval_)
        print('median survival time:', median_surv)
        #print('median survival time 95% CI:\n', median_surv_CI)
    
    plt.xlabel('Time', fontweight='bold', fontsize=12)
    plt.ylabel('Proportion of studies (%)', fontweight='bold', fontsize=12)
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
    plt.grid(True)
    plt.title(score_type, fontsize=20)
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    fn = 'kmf_' + str(score_type) + '.png'
    plt.savefig(os.path.join(output_dir, fn), format='png', dpi=300)
    plt.close()
    
    #print('saved Kaplan-Meier curve!')

    




