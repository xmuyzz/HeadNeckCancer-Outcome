import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycox.evaluation import EvalSurv
from pycox.utils import kaplan_meier
from go_models.km_plot_mul import km_plot_mul



def risk_strat(proj_dir, out_dir):

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
    #pat_id = df_val['pat_id'].to_list()
    df_val['score'] = prob_scores
    #df = pd.DataFrame({'pat_id': pat_id, 'score': prob_scores})
    df1 = df_val[df_val['score'] >= median_score]
    df2 = df_val[df_val['score'] < median_score]
    
    # Kaplan-Meier curve
    df_kms = []
    for df in [df1, df2]:
        km = kaplan_meier(
            durations=df['death_time'].to_numpy(),
            events=df['death_event'].to_numpy(),
            start_duration=0
            )
        df_km = pd.DataFrame({'km_index': km.index, 'km_value': km.values})
        df_kms.append(df_km)

    #print('Kaplan-Meier curve:\n', surv_km.round(3))
    #surv_km.to_csv(os.path.join(pro_data_dir, 'surv_km.csv'), index=False)
    
    # plot Kaplan-Meier curve
    km_plot_mul(
        out_dir=out_dir,
        df_kms=df_kms,
        fn='risk_km.png'
        )
    print('saved Kaplan-Meier curve!')
