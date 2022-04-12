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
from statistics.surv_plot import surv_plot
from statistics.surv_plot_mul import surv_plot_mul
from statistics.roc import roc
from statistics.kmf_risk_strat import kmf_risk_strat



def test(pro_data_dir, output_dir, cox_model, load_model, dl_val, score_type,
         cnn_name, epochs, lr):

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
    

    if load_model == 'model':
        model_fn = str(cnn_name) + '_' + str(epochs) + '_' + \
                   str(lr) + '_' + 'model.pt'
        cox_model.load_net(os.path.join(pro_data_dir, model_fn))
    elif load_model == 'weights':
        weights_fn = str(cnn_name) + '_' + str(epochs) + '_' + \
                     str(lr) + '_' + 'weights.pt'
        cox_model.load_model_weights(os.path.join(pro_data_dir, weights_fn))
    # predict on val dataset and save prediction df
    surv = cox_model.predict_surv_df(dl_val)

    #fn_surv = str(cnn_name) + '_' + str(epochs) + '_' + \
    #          str(lr) + '_' + 'surv.csv'
    #surv.to_csv(os.path.join(pro_data_dir, fn_surv), index=False)

    # load duration index to plot survival curves
    duration_index = np.load(os.path.join(pro_data_dir, 'duration_index.npy'))

    # check duration and events for each patient
    #df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    #df = df_val[['death_time', 'death_event']]
    #df.columns = ['time', 'event']
    #print('patient survival info:\n', df[0:10]) 

    # plot individual survival pred curve
    #-------------------------------------
    #surv = pd.read_csv(os.path.join(pro_data_dir, 'surv.csv'))
    #print('survival prediction:\n', surv.round(3))
    # plot multiple survival predictions
    fn = str(cnn_name) + '_' + str(epochs) + '_' + \
         str(lr) + '_' + 'surv.png'
    surv_plot_mul(
        proj_dir=proj_dir,
        n_curves=200,
        fn=fn)

    # Concordance index
    #-------------------    
    """
    Time dependent concordance index from:
    Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A time-dependent discrimination
    index for survival data. Statistics in Medicine 24:3927–3944.
    
    @Arguments:
        durations {np.array[n]} -- Event times (or censoring times.)
        events {np.array[n]} -- Event indicators (0 is censoring).
        surv {np.array[n_times, n]} -- Survival function (each row is a duraratoin, and each col
            is an individual).
        surv_idx {np.array[n_test]} -- Mapping of survival_func s.t. 'surv_idx[i]' gives index in
            'surv' corresponding to the event time of individual 'i'.
    @Keyword Arguments:
        method {str} -- Type of c-index 'antolini' or 'adj_antolini' (default {'adj_antolini'}).
    @Returns:
        float -- Time dependent concordance index.
    """

    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    durations = df_val['death_time'].to_numpy()
    events = df_val['death_event'].to_numpy()
    ev = EvalSurv(
        surv=surv, 
        durations=durations,  
        events=events,
        censor_surv='km')
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

    # risk stratification
    #---------------------------
    score_types = ['median', '3yr_surv', '5yr_surv', 'os_surv']
    for score_type in score_types:
        print(score_type)
        kmf_risk_strat(
            proj_dir,
            score_type=score_type,
            cnn_name=cnn_name,
            epochs=epochs,
            lr=lr)

    # calculate ROC
    #-------------
    roc(proj_dir)




