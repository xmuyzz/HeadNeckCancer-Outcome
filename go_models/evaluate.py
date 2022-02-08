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
from go_models._surv_plot import _surv_plot
from go_models.km_plot import km_plot




def evaluate(proj_dir, lab_drive_dir, cox_model, load_model, dl_val):

    """
    We can use the EvalSurv class for evaluation the concordance, brier score and binomial 
    log-likelihood. Setting censor_surv='km' means that we estimate the censoring distribution 
    by Kaplan-Meier on the test set.
    """

    output_dir = os.path.join(lab_drive_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)

    # prediction
    #--------------
    """ load_model: load entire model or weights
    """
    if load_model == 'model':
        cox_model.load_net(os.path.join(pro_data_dir, 'model.pt'))
    elif load_model == 'weights':
        cox_model.load_model_weights(os.path.join(pro_data_dir, 'weights.pt'))
    # predict on val dataset and save prediction df
    surv = cox_model.predict_surv_df(dl_val)
    surv.to_csv(os.path.join(pro_data_dir, 'surv.csv'), index=False)

    # load duration index to plot survival curves
    duration_index = np.load(os.path.join(pro_data_dir, 'duration_index.npy'))

    # check duration and events for each patient
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    df = df_val[['sur_duration', 'survival']]
    df.columns = ['duration', 'events']
    print('patient survival info:\n', df[0:10]) 

    # plot individual survival pred curve
    #-------------------------------------
    surv = pd.read_csv(os.path.join(pro_data_dir, 'surv.csv'))
    print('survival prediction:\n', surv.round(3))
    # plot single survival curve
    surv_plot(
        lab_drive_dir=lab_drive_dir, 
        x=duration_index,
        y=surv.iloc[:, 0],
        fn='surv.png'
        )
    # plot multiple survival predictions
    _surv_plot(
        lab_drive_dir=lab_drive_dir,
        proj_dir=proj_dir,
        n_curves=100,
        fn='surv_multi.png'
        )

    # compute the average survival predictions
    #-------------------------------------------
    surv_mean = surv.mean(axis=1)
    print('survival pred mean:\n', surv_mean.round(3))
    # plot survival curve
    surv_plot(
        lab_drive_dir=lab_drive_dir,
        x=duration_index,
        y=surv_mean,
        fn='surv_mean.png'
        )

    # Kaplan-Meier curve
    #--------------------
    km = kaplan_meier(
        durations=df['duration'].values,  
        events=df['events'].values,
        start_duration=0
        )
    # convert pd series to pd dataframe
    surv_km = pd.DataFrame({'x': km.index, 'y': km.values})
    print('Kaplan-Meier curve:\n', surv_km.round(3))
    surv_km.to_csv(os.path.join(pro_data_dir, 'surv_km.csv'), index=False)
    # plot Kaplan-Meier curve
    km_plot(
        lab_drive_dir,
        x=surv_km['x'],
        y=surv_km['y'],
        fn='km_curve.png'
        )
    print('saved Kaplan-Meier curve!')

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
    durations = df_val['sur_duration'].values
    events = df_val['survival'].values
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




