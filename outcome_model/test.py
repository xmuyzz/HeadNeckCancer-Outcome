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
from get_cnn_model import get_cnn_model
from get_cox_model import get_cox_model
from logger import test_logger


def test(surv_type, data_set, task_dir, eval_model, cox_model, df, dl, cox):
    """
    Model evaluation
    Args:
        run_type {str} -- val or test.
    Returns:
        C-index {float} -- Time dependent concordance index.
        Surv {pd.df} -- survival prediction.
    """
    # load model
    if eval_model == 'best_loss_model':
        fn = 'weights_best_loss.pt'
    elif eval_model == 'best_cindex_model':
        fn = 'weights_best_cindex.pt'
    elif eval_model == 'target_cindex_model':
        fn = 'weights_target_cindex.pt'
    elif eval_model == 'final_model':
        fn = 'weights_final.pt'
    cox_model.load_model_weights(task_dir + '/models/' + fn)
    print('successfully loaded model!')

    save_dir = task_dir + '/' + data_set
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # surv prediction

    raw_surv = cox_model.predict_surv_df(dl)
    raw_surv.to_csv(save_dir + '/raw_surv_1.csv', index=False)
    print(raw_surv.shape)
    if cox == 'PCHazard':
        cox_model.sub = 10
        surv = cox_model.predict_surv_df(dl)
    elif cox in ['LogisticHazard', 'MTLR', 'PMF', 'DeepHit']:
        surv = cox_model.interpolate(10).predict_surv_df(dl)
    """
    It is, therefore, often beneficial to interpolate the survival estimates. Linear interpolation (constant density interpolation) 
    can be performed with the interpolate method. We also need to choose how many points we want to replace each grid point with. 
    Her we will use 10."""
    # surv = cox_model.interpolate(10).predict_surv_df(dl)
    print(surv.shape)
    surv.to_csv(save_dir + '/full_surv_1.csv', index=False)

    # # individual Hazard Ratio
    # HR = cox_model.predict_hazard(dl)
    # mean_HR = round(np.mean(HR), 3)
    # median_HR = round(np.median(HR), 3)
    # print('Mean HR:', mean_HR)
    # print('Median HR:', median_HR)

    # c-index
    #surv_type = 'rfs'
    df = df.dropna(subset=[surv_type + '_event', surv_type + '_time'])
    durations = df[surv_type + '_time'].to_numpy()
    events = df[surv_type + '_event'].to_numpy()
    #df = df.dropna(subset=['rfs_event', 'rfs_time'])
    #durations = df['rfs_time'].to_numpy()
    #events = df['rfs_event'].to_numpy()
    raw_ev = EvalSurv(surv=raw_surv, durations=durations, events=events, censor_surv='km')
    raw_c_index = round(raw_ev.concordance_td(), 3)
    print('data set:', data_set)
    print('raw c-index:', raw_c_index)

    ev = EvalSurv(surv=surv, durations=durations, events=events, censor_surv='km')
    c_index = round(ev.concordance_td(), 3)
    print('data set:', data_set)
    print('c-index:', c_index)

    # Brier score
    # plot IPCW Brier score for a given set of times.
    time_grid = np.linspace(durations.min(), durations.max(), 100)
    _ = ev.brier_score(time_grid).plot()
    #time_grid = np.linspace(0, sim_test[0].max())
    """The two time-dependent scores above can be integrated over time to 
    produce a single score Graf et al. 1999. 
    In practice this is done by numerical integration over a defined time_grid.
    """
    brier_score = ev.integrated_brier_score(time_grid)
    brier_score = round(brier_score, 3)
    print('brier_score:', brier_score)

    # Negative binomial log-likelihood
    ev.nbll(time_grid).plot()
    plt.ylabel('NBLL')
    _ = plt.xlabel('Time')
    # Integrated scores
    nbll_score = ev.integrated_nbll(time_grid)
    nbll_score = round(nbll_score, 3)
    print('nbll_score:', nbll_score)

    # test logger
    #test_logger(save_dir, c_index, mean_HR, median_HR, brier_score, nbll_score, eval_model)
    test_logger(save_dir, c_index, brier_score, nbll_score, eval_model)
