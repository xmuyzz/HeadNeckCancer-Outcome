import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from pycox.models import PCHazard
from pycox.models import LogisticHazard
from pycox.models import DeepHitSingle
from pycox.utils import kaplan_meier
from go_models.surv_plot import surv_plot
from go_models.surv_plot_mul import surv_plot_mul
from go_models.roc import roc
from go_models.kmf_risk_strat import kmf_risk_strat



def evaluate2(proj_dir, cox_model, load_model, model_fn, data_loader):

    """
    Model evaluation
    
    Args:
        durations {np.array[n]} -- Event times (or censoring times.)
        events {np.array[n]} -- Event indicators (0 is censoring).
        surv {np.array[n_times, n]} -- Survival function (each row is a duraratoin, and each col
            is an individual).
        surv_idx {np.array[n_test]} -- Mapping of survival_func s.t. 'surv_idx[i]' gives index in
            'surv' corresponding to the event time of individual 'i'.
    
    Keyword Args:
        method {str} -- Type of c-index 'antolini' or 'adj_antolini' (default {'adj_antolini'}).
    
    Returns:
        float -- Time dependent concordance index.
    
    """
    

    output_dir = os.path.join(proj_dir, 'output/tune')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)

    # prediction
    #--------------
    if load_model == 'model':
        cox_model.load_net(os.path.join(pro_data_dir, model_fn))
    elif load_model == 'weights':
        cox_model.load_model_weights(os.path.join(pro_data_dir, weights_fn))

    # predict on val dataset and save prediction df
    surv = cox_model.predict_surv_df(data_loader)
    surv_fn = model_fn.split('_model')[0] + '_' + 'surv.csv'
    surv.to_csv(os.path.join(pro_data_dir, surv_fn), index=False)

    # plot individual survival pred curve
    #-------------------------------------
    fn = surv_fn.split('.')[0] + '.png'
    surv_plot_mul(
        proj_dir=proj_dir,
        n_curves=100,
        fn=fn
        )

    # Concordance index
    #-------------------    
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    durations = df_val['death_time'].to_numpy()
    events = df_val['death_event'].to_numpy()
    ev = EvalSurv(
        surv=surv, 
        durations=durations,  
        events=events,
        censor_surv='km'
        )
    c_index = ev.concordance_td()
    print('concordance index:', round(c_index, 3))

    # Brier score
    #--------------
    """We can plot the the IPCW Brier score for a given set of times. 
    """
    time_grid = np.linspace(durations.min(), durations.max(), 100)
    _ = ev.brier_score(time_grid).plot()
    #time_grid = np.linspace(0, sim_test[0].max())
    
    """ 
    The two time-dependent scores above can be integrated over time to 
    produce a single score Graf et al. 1999. 
    In practice this is done by numerical integration over a defined time_grid.
    """
    brier_score = ev.integrated_brier_score(time_grid)
    print('brier_score:', round(brier_score, 3))

    # Negative binomial log-likelihood
    #-----------------------------------
    ev.nbll(time_grid).plot()
    plt.ylabel('NBLL')
    _ = plt.xlabel('Time')
    # Integrated scores
    nbll_score = ev.integrated_nbll(time_grid)
    print('nbll_score:', round(nbll_score, 3))


