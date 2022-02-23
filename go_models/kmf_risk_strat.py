import os
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test
from go_models.km_plot_mul import km_plot_mul
#from km_plot_mul import km_plot_mul
from time import localtime, strftime



def kmf_risk_strat(proj_dir, out_dir, score_type, cnn_name, epochs):

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
    
    """calculate probility score for each patient
    """
    surv = pd.read_csv(os.path.join(pro_data_dir, 'surv.csv'))
    if score_type == 'mean':
        prob_scores = surv.mean(axis=0).to_list()
    elif score_type == 'median':
        prob_scores = surv.median(axis=0).to_list()
    elif score_type == '3year_survival':
        # choose 4th row if number of durations is 20
        prob_scores == surv.iloc[4, :].to_list()

    """get median scores to stratify risk groups
    """
    median_score = np.median(prob_scores)
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    #pat_id = df_val['pat_id'].to_list()
    df_val['score'] = prob_scores
    #df = pd.DataFrame({'pat_id': pat_id, 'score': prob_scores})
    df1 = df_val[df_val['score'] >= median_score]
    df2 = df_val[df_val['score'] < median_score]
    
    """log-rank test
    """
    results = logrank_test(
        df1['death_time'],
        df2['death_time'],
        event_observed_A=df1['death_time'],
        event_observed_B=df2['death_time']
        )
    #results.print_summary()
    p_value = np.around(results.p_value, 3)
    print('log-rank test p-value:', p_value)
    #print(results.test_statistic)

    """Kaplan-Meier curve
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    kmf1 = KaplanMeierFitter()
    kmf1.fit(
        df1['death_time'],
        df1['death_event'],
        label='Low risk'
        )
    ax = kmf1.plot_survival_function(
        ax=ax,
        show_censors=True,
        ci_show=True,
        #censor_style={"marker": "o", "ms": 60}
        )

    kmf2 = KaplanMeierFitter()
    kmf2.fit(
        df2['death_time'],
        df2['death_event'],
        label='High risk'
        )
    ax = kmf2.plot_survival_function(
        ax=ax,
        show_censors=True,
        ci_show=True,
        #censor_style={"marker": "o", "ms": 60}
        )
    
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
    add_at_risk_counts(kmf1, ax=ax)
    add_at_risk_counts(kmf2, ax=ax)
    fn = 'kmf' + str(cnn_name) + '_' + str(epochs) + '.png'
    plt.savefig(os.path.join(output_dir, fn), format='png', dpi=300)
    plt.close()
    
    #print('saved Kaplan-Meier curve!')

    
    """median survival time
    """
    median_surv1 = kmf1.median_survival_time_
    median_surv_CI1 = median_survival_times(kmf1.confidence_interval_)
    print('median survival time:', median_surv1)
    #print('median survival time 95% CI:\n', median_surv_CI1)
    median_surv2 = kmf2.median_survival_time_
    median_surv_CI2 = median_survival_times(kmf2.confidence_interval_)
    print('median survival time:', median_surv2)
    #print('median survival time 95% CI:\n', median_surv_CI2)




if __name__ == '__main__':

    data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated'
    proj_dir = '/mnt/HDD_6TB/HN_Outcome'
    out_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'

    kmf_risk_strat(proj_dir, out_dir)



