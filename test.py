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



def test(run_type, model_dir, log_dir, pro_data_dir, cox_model, dl_val, dl_test,
         cnn_name, model_depth):

    """
    Model evaluation
    
    Args:
        run_type {str} -- val or test.
    
    Returns:
        C-index {float} -- Time dependent concordance index.
        Surv {pd.df} -- survival prediction.
    
    """

    # load model
    fn = str(cnn_name) + str(model_depth) + '_c_indexs.npy'
    c_indexs = np.load(os.path.join(log_dir, fn))
    print(c_indexs)
    if eval_model == 'best_model':
        c_index = np.amax(c_indexs)
        fn = cnn_name + str(model_depth) + '_' + str(c_index) + '_final_model.pt'
    elif eval_model == 'final_model':
        c_index = c_indexs[-1]
        fn = cnn_name + str(model_depth) + '_' + str(c_index) + '_model.pt'
    cox_model.load_net(os.path.join(model_dir, fn))
    
    if run_type == 'val':
        dl = dl_val
        df_csv = 'df_pn_masked_val0.csv'
    elif run_type == 'test':
        dl = dl_test
        df_csv = 'df_pn_masked_test.csv'

    # surv prediction
    surv = cox_model.predict_surv_df(dl)
    fn_surv = run_type + cnn_name + str(model_depth) + str(c_index) + 'surv.csv'
    surv.to_csv(os.path.join(pro_data_dir, fn_surv), index=False)

    # c-index
    df = pd.read_csv(os.path.join(pro_data_dir, df_csv))
    durations = df['death_time'].to_numpy()
    events = df['death_event'].to_numpy()
    ev = EvalSurv(
        surv=surv,
        durations=durations,
        events=events,
        censor_surv='km')
    c_index = round(ev.concordance_td(), 3)
    print('c-index:', c_index)

    # Brier score
    #--------------
    """plot IPCW Brier score for a given set of times. 
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



