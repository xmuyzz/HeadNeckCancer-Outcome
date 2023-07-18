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



def kmf_risk_strat4(proj_dir, out_dir, score_type, cnn_name, epochs):

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
    elif score_type == '3year_survival':
        # choose 5th row if number of durations is 20
        prob_scores = surv.iloc[4, :].to_list()
    elif score_type == '5year_survival':
        prob_scores = surv.iloc[7, :].to_list()


    # get median scores to stratify risk groups
    median_score = np.median(prob_scores)
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    #pat_id = df_val['pat_id'].to_list()
    df_val['score'] = prob_scores
    #print(np.percentile(prob_scores, 25))
    #print(np.percentile(prob_scores, 50))
    #df = pd.DataFrame({'pat_id': pat_id, 'score': prob_scores})
    df0 = df_val[df_val['score'] <= np.percentile(prob_scores, 25)]
    df1 = df_val[(df_val['score'] > np.percentile(prob_scores, 25)) & \
                 (df_val['score'] <= np.percentile(prob_scores, 50))]
    df2 = df_val[(df_val['score'] > np.percentile(prob_scores, 50)) & \
                 (df_val['score'] <= np.percentile(prob_scores, 75))]
    df3 = df_val[df_val['score'] > np.percentile(prob_scores, 75)]
    #print('df1:', df1.shape[0])
    #print('df2:', df2.shape[0])
    

    # multivariate log-rank test
    #---------------------------
    # add group ID to dataframe
    ids = [0, 1, 2, 3]
    dfs = [df0, df1, df2, df3]
    for i, df in zip(ids, dfs):
        group = [i] * df.shape[0]
        #print(group)
        df['group'] = group
        #print(df)
    df = pd.concat([df0, df1, df2, df3], axis=0)
    results = multivariate_logrank_test(
        df['death_time'],
        df['group'],
        df['death_event'],
        )
    #results.print_summary()

    p_value = np.around(results.p_value, 3)
    print('log-rank test p-value:', results.p_value)
    #print(results.test_statistic)

    
    # Kaplan-Meier curve
    #---------------------
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    for df in [df0, df1, df2, df3]:
        kmf = KaplanMeierFitter()
        kmf.fit(
            df['death_time'],
            df['death_event'],
            label='Low risk'
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
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    fn = 'kmf4' + str(cnn_name) + '_' + str(epochs) + '.png'
    plt.savefig(os.path.join(output_dir, fn), format='png', dpi=300)
    plt.close()
    
    #print('saved Kaplan-Meier curve!')

    




