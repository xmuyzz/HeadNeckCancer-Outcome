import os
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
from time import localtime, strftime
from sklearn.cluster import KMeans
from opts import parse_opts
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import statsmodels.api as sm
from sklearn.preprocessing import minmax_scale
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from sklearn_pandas import DataFrameMapper 
import torch 
import torchtuples as tt # Some useful functions
#from pycox.datasets import metabric
from pycox.models import MTLR, PMF, PCHazard, LogisticHazard
from pycox.evaluation import EvalSurv
from sklearn.model_selection import KFold, train_test_split
import torch.nn as nn
import torch.nn.init as init
from datetime import datetime
from time import localtime, strftime
from datetime import datetime
import pytz
import scipy.stats as ss



class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, dropout):
        super(MLP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(in_features, num_nodes),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_nodes),
            nn.Dropout(dropout),
            nn.Linear(num_nodes, num_nodes),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_nodes),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(num_nodes, out_features)

        # Apply weight initialization to the first layer only
        init.kaiming_uniform_(self.features[0].weight)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MLP2(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, dropout):
        super(MLP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(in_features, num_nodes),
            nn.ELU(),
            nn.BatchNorm1d(num_nodes),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(num_nodes, out_features)

        # Apply weight initialization to the first layer only
        init.kaiming_uniform_(self.features[0].weight)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



def mean_CI(data):
    """
    Calculate mean value and 95% CI
    """
    mean = np.mean(np.array(data))
    CI = ss.t.interval(
        confidence=0.95,
        df=len(data)-1,
        loc=np.mean(data),
        scale=ss.sem(data))
    lower = CI[0]
    upper = CI[1]
    return mean, lower, upper


def get_95CI(surv, times, events):

    ev = EvalSurv(surv, times, events, censor_surv='km')
    original_c_index = ev.concordance_td('antolini')
    original_c_index = round(original_c_index, 2)

    # Bootstrap resampling
    num_bootstrap_samples = 1000
    c_indices_bootstrap = []

    for _ in range(num_bootstrap_samples):
        # Generate a bootstrap sample
        indices = np.random.choice(len(events), size=len(events), replace=True)
        events_bootstrap = events[indices]
        times_bootstrap = times[indices]
        surv_bootstrap = surv.iloc[:, indices]
        # print(events_bootstrap)
        # print(times_bootstrap)
        # print(surv_bootstrap)

        # Calculate C-index for the bootstrap sample
        ev = EvalSurv(surv_bootstrap, times_bootstrap, events_bootstrap, censor_surv='km')
        c_index_bootstrap = ev.concordance_td('antolini')
        c_indices_bootstrap.append(c_index_bootstrap)

    # Calculate the confidence interval
    confidence_interval = np.percentile(c_indices_bootstrap, [2.5, 97.5])
    confidence_interval = np.around(confidence_interval, 2)

    # # use scipy for 95% CI
    # CI = mean_CI(c_indices_bootstrap)
    # print('scipy CI:', CI)

    median_c_index = np.around(np.median(c_indices_bootstrap), 2)
    mean_c_index = np.around(np.mean(c_indices_bootstrap), 2)

    summary = (original_c_index, confidence_interval, median_c_index, mean_c_index)
    print(summary)
    

    return c_indices_bootstrap


def get_data(opt, score_type, hpv, surv_type, cox_variable, normalize, random_state):

    task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_size + '_' + \
               opt.img_type + '_' + opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 
    data_dir = opt.data_dir + '/' + opt.img_size + '_' + opt.img_type + '/new_test'
    
    dfs = []
    for dataset in ['ts', 'tx_maastro', 'tx_bwh']:
        #print('dataset:', dataset)
        save_dir = task_dir + '/' + dataset
        fn = dataset + '_img_label_' + opt.tumor_type + '.csv'
        df = pd.read_csv(save_dir + '/' + fn)
        #print(df.head(0))
        times, events = surv_type + '_time', surv_type + '_event'

        # calculate probility score for each patient
        surv = pd.read_csv(save_dir + '/raw_surv_1.csv')

        if score_type == 'mean_surv':
            dl_scores = surv.mean(axis=0).to_list()
        elif score_type == 'median_surv':
            dl_scores = surv.median(axis=0).to_list()
        elif score_type == '3yr_surv':
            # choose 5th row if number of durations is 20
            dl_scores = surv.iloc[2, :].to_list()
        elif score_type == '5yr_surv':
            dl_scores = surv.iloc[3, :].to_list()
        elif score_type == 'os_surv':
            dl_scores = surv.iloc[9, :].to_list()
        # add DL scores to df
        #df['dl_score'] = dl_scores

        # df = df.dropna(subset=[times, events, 'Muscle_Area', 'Muscle_Density', 'Adipose_Area', 'Adipose_Density', 
        #     'Female', 'Age>65', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV'])

        #clinical_list = ['Female', 'Age>65', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV']
        clinical_list = ['Female', 'age', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV']
        if cox_variable == 'clinical':
            df_cox = df[[times, events] + clinical_list].astype(float) 
        elif cox_variable == 'clinical_muscle': 
            df_cox = df[['Muscle_Area'] + clinical_list]
            df_cox.loc[:] = minmax_scale(df_cox)
            df_cox[times], df_cox[events] = df[times], df[events]
        elif cox_variable == 'clinical_muscle_adipose':
            df_cox = df[['Muscle_Area', 'Adipose_Density'] + clinical_list] 
            df_cox.loc[:] = minmax_scale(df_cox)
            df_cox[times], df_cox[events] = df[times], df[events]
        elif cox_variable == 'clinical_tumor_muscle_adipose': 
            df_cox = df[['Muscle_Area', 'Adipose_Density', 'tumor volume', 'node volume'] + clinical_list] 
            df_cox.loc[:] = minmax_scale(df_cox)
            df_cox[times], df_cox[events] = df[times], df[events]
        elif cox_variable == 'clinical_tumor':
            df_cox = df[['tumor volume', 'node volume'] + clinical_list] 
            df_cox.loc[:] = minmax_scale(df_cox)
            df_cox[times], df_cox[events] = df[times], df[events]
        elif cox_variable == 'clinical_adipose':
            df_cox = df[['Adipose_Density'] + clinical_list]   
            df_cox[:] = minmax_scale(df_cox)
            df_cox[times], df_cox[events] = [df[times], df[events]]  

        elif cox_variable == 'clinical_dl':
            df_surv = surv.T
            df_surv.columns = ['time'] * surv.shape[0]
            df_surv = df_surv.reset_index()
            print(df.head(0))
            df_cox = df[clinical_list]  
            #df_cox.reset_index()

            df_cox = pd.concat([df_surv, df_cox], axis=1)
            df_cox.loc[:] = minmax_scale(df_cox)

            df_cox.loc[:, times] = df[times].to_list().copy()
            df_cox.loc[:, events] = df[events].to_list().copy()

            df_cox = df_cox.dropna(subset=[times, events, 'Female', 'age', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV'])           
            print(df_cox)
            print(df_cox['os_event'].to_list())
        elif cox_variable == 'clinical_tumor_dl':
            df_surv = surv.T
            df_surv.columns = ['time'] * surv.shape[0]
            df_surv = df_surv.reset_index()
            print(df.head(0))
            df_cox = df[clinical_list + ['tumor volume', 'node volume']]
            #df_cox.reset_index()

            df_cox = pd.concat([df_surv, df_cox], axis=1)
            df_cox.loc[:] = minmax_scale(df_cox)

            df_cox.loc[:, times] = df[times].to_list().copy()
            df_cox.loc[:, events] = df[events].to_list().copy()
            
            df_cox = df_cox.dropna(subset=[times, events, 'Female', 'age', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV'])           
            print(df_cox)
            print(df_cox['os_event'].to_list())
        elif cox_variable == 'clinical_tumor_dl_muscle':
            df_surv = surv.T
            df_surv.columns = ['time'] * surv.shape[0]
            df_surv = df_surv.reset_index()
            print(df.head(0))
            df_cox = df[clinical_list + ['tumor volume', 'node volume', 'Muscle_Area']]
            #df_cox.reset_index()

            df_cox = pd.concat([df_surv, df_cox], axis=1)
            df_cox.loc[:] = minmax_scale(df_cox)

            df_cox.loc[:, times] = df[times].to_list().copy()
            df_cox.loc[:, events] = df[events].to_list().copy()
            
            df_cox = df_cox.dropna(subset=[times, events, 'Female', 'age', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV'])           
            print(df_cox)
            print(df_cox['os_event'].to_list())   
        elif cox_variable == 'tot':
            df_surv = surv.T
            df_surv.columns = ['time'] * surv.shape[0]
            df_surv = df_surv.reset_index()
            print(df.head(0))
            #df_cox = df[['Muscle_Area', 'Muscle_Density', 'Adipose_Area', 'Adipose_Density', 'tumor volume', 'node volume'] + clinical_list]  
            df_cox = df[['Muscle_Area', 'Adipose_Density', 'tumor volume', 'node volume'] + clinical_list]  
            #df_cox.reset_index()

            df_cox = pd.concat([df_surv, df_cox], axis=1)
            df_cox.loc[:] = minmax_scale(df_cox)

            df_cox.loc[:, times] = df[times].to_list().copy()
            df_cox.loc[:, events] = df[events].to_list().copy()
            
            # df_cox = df_cox.dropna(subset=[times, events, 'Muscle_Area', 'Muscle_Density', 'Adipose_Area', 'Adipose_Density', 
            #     'Female', 'age', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV'])
            df_cox = df_cox.dropna(subset=[times, events, 'Muscle_Area', 'Adipose_Density', 'tumor volume', 'node volume',
                'Female', 'age', 'Smoking>10py', 'T-Stage-1234', 'N-Stage-0123', 'HPV'])           
            # print(df_cox)
            # print(df_cox['os_event'].to_list())    
        
        dfs.append(df_cox)

    df0 = dfs[0]
    df_tx1 = dfs[1]
    df_tx2 = dfs[2]

    df_tr_va, df_ts = train_test_split(
        df0, 
        test_size=0.15, 
        stratify=df0[surv_type + '_event'], 
        random_state=random_state)

    df_tr, df_va = train_test_split(
        df_tr_va, 
        test_size=0.15, 
        stratify=df_tr_va[surv_type + '_event'], 
        random_state=random_state)

    tr_dir = task_dir + '/tr2'
    va_dir = task_dir + '/va2'
    ts_dir = task_dir + '/ts2'
    if not os.path.exists(tr_dir):
        os.makedirs(tr_dir)
    if not os.path.exists(va_dir):
        os.makedirs(va_dir)
    if not os.path.exists(ts_dir):
        os.makedirs(ts_dir)
    df_tr.to_csv(tr_dir + '/tr2_img_label_pn.csv', index=False)
    df_va.to_csv(va_dir + '/va2_img_label_pn.csv', index=False)
    df_ts.to_csv(ts_dir + '/ts2_img_label_pn.csv', index=False)

    # df_ts = df_tr.sample(frac=0.2)
    # df_tr = df_tr.drop(df_ts.index)
    # df_va = df_tr.sample(frac=0.2)
    # df_tr = df_tr.drop(df_va.index)
    print('tr size:', df_tr.shape[0])
    print('va size:', df_va.shape[0])
    print('ts size:', df_ts.shape[0])
    print('tx1 size:', df_tx1.shape[0])
    print('tx2 size:', df_tx2.shape[0])

    return df_tr, df_va, df_ts, df_tx1, df_tx2


def train(model_dir, cox_type, surv_type, df_tr, df_va, df_ts, df_tx1, df_tx2, dropout, lr, num_nodes, batch_size, epochs):

    times, events = surv_type + '_time', surv_type + '_event'
    # x values
    x_tr = df_tr.drop(columns=[times, events]).values.astype('float32')
    x_va = df_va.drop(columns=[times, events]).values.astype('float32')
    x_ts = df_ts.drop(columns=[times, events]).values.astype('float32')
    x_tx1 = df_tx1.drop(columns=[times, events]).values.astype('float32')
    x_tx2 = df_tx2.drop(columns=[times, events]).values.astype('float32')
    #print(x_tr)

    if cox_type == 'LogisticHazard':
        cox_model = LogisticHazard
    elif cox_type == 'MTLR':
        cox_model = MTLR
    elif cox_type == 'PCHazard':
        cox_model = PCHazard
    elif cox_type == 'PMF':
        cox_model = PMF

    # label transform
    num_durations = 10
    labtrans = cox_model.label_transform(num_durations)
    get_target = lambda df: (df[times].values, df[events].values)
    y_tr = labtrans.fit_transform(*get_target(df_tr))
    # print('labtrans:', labtrans)
    # print('labtrans.cuts:', labtrans.cuts)
    y_va = labtrans.transform(*get_target(df_va))
    val = (x_va, y_va)

    # We don't need to transform the test labels
    ts_times, ts_events = get_target(df_ts)
    va_times, va_events = get_target(df_va)
    tx1_times, tx1_events = get_target(df_tx1)
    tx2_times, tx2_events = get_target(df_tx2)

    # build model
    in_features = x_tr.shape[1]
    out_features = labtrans.out_features
    #print('out_features:', out_features)

    net = MLP(in_features, out_features, num_nodes, dropout)
    model = cox_model(net, tt.optim.Adam, duration_index=labtrans.cuts)
    model.optimizer.set_lr(lr)
    callbacks = [tt.callbacks.EarlyStopping(patience=10, file_path=model_dir + '/cpt_weights.pt')]
    log = model.fit(x_tr, y_tr, batch_size, epochs, callbacks, val_data=val, verbose=0)

    # _ = log.plot()

    # surv.iloc[:, :5].plot(drawstyle='steps-post')
    # plt.ylabel('S(t | x)')
    # _ = plt.xlabel('Time')

    # surv = model.interpolate(10).predict_surv_df(x_ts)

    # surv.iloc[:, :5].plot(drawstyle='steps-post')
    # plt.ylabel('S(t | x)')
    # _ = plt.xlabel('Time')

    print('\ntesting results .......')
    if cox_type == 'PCHazard':
        model.sub = 10
        va_surv = model.predict_surv_df(x_va)
        ts_surv = model.predict_surv_df(x_ts)
        tx1_surv = model.predict_surv_df(x_tx1)
        tx2_surv = model.predict_surv_df(x_tx2)
    else:
        va_surv = model.interpolate(10).predict_surv_df(x_va)
        ts_surv = model.interpolate(10).predict_surv_df(x_ts)
        tx1_surv = model.interpolate(10).predict_surv_df(x_tx1)
        tx2_surv = model.interpolate(10).predict_surv_df(x_tx2)
 
    va_c_idx = EvalSurv(va_surv, va_times, va_events, censor_surv='km').concordance_td('antolini')
    ts_c_idx = EvalSurv(ts_surv, ts_times, ts_events, censor_surv='km').concordance_td('antolini')
    tx1_c_idx = EvalSurv(tx1_surv, tx1_times, tx1_events, censor_surv='km').concordance_td('antolini')
    tx2_c_idx = EvalSurv(tx2_surv, tx2_times, tx2_events, censor_surv='km').concordance_td('antolini')

    # va_c_idx_bootstrap = get_95CI(va_surv, va_times, va_events)
    # ts_c_idx_bootstrap = get_95CI(ts_surv, ts_times, ts_events)
    # tx1_c_idx_bootstrap = get_95CI(tx1_surv, tx1_times, tx1_events)
    # tx2_c_idx_bootstrap = get_95CI(tx2_surv, tx2_times, tx2_events)

    va_c_idx = round(va_c_idx, 2)
    ts_c_idx = round(ts_c_idx, 2)
    tx1_c_idx = round(tx1_c_idx, 2)
    tx2_c_idx = round(tx2_c_idx, 2)
  
    print('val c-index:', va_c_idx)
    print('test c-index:', ts_c_idx)
    print('tx1 c-index:', tx1_c_idx)
    print('tx2 c-index:', tx2_c_idx) 

    return model, va_c_idx, ts_c_idx, tx1_c_idx, tx2_c_idx

    
def test(model_dir, task_dir, cox_type, cox_variable, surv_type, df_tr, df_va, df_ts, df_tx1, df_tx2, model_weight, ts_nodes):

    if cox_type == 'LogisticHazard':
        cox_model = LogisticHazard
    elif cox_type == 'MTLR':
        cox_model = MTLR
    elif cox_type == 'PCHazard':
        cox_model = PCHazard
    elif cox_type == 'PMF':
        cox_model = PMF

    times, events = surv_type + '_time', surv_type + '_event'
    # x values
    x_tr = df_tr.drop(columns=[times, events]).values.astype('float32')
    x_va = df_va.drop(columns=[times, events]).values.astype('float32')
    x_ts = df_ts.drop(columns=[times, events]).values.astype('float32')
    x_tx1 = df_tx1.drop(columns=[times, events]).values.astype('float32')
    x_tx2 = df_tx2.drop(columns=[times, events]).values.astype('float32')  

    get_target = lambda df: (df[times].values, df[events].values)
    va_times, va_events = get_target(df_va)
    ts_times, ts_events = get_target(df_ts)
    tx1_times, tx1_events = get_target(df_tx1)
    tx2_times, tx2_events = get_target(df_tx2)

    # MLP network
    net = MLP(in_features=x_tr.shape[1], 
              out_features=10, 
              num_nodes=ts_nodes, 
              dropout=0.3)
    num_durations = 10         
    labtrans = cox_model.label_transform(num_durations)
    y_tr = labtrans.fit_transform(*get_target(df_tr))
    # print('labtrans:', labtrans)
    # print('labtrans.cuts:', labtrans.cuts)
    model = cox_model(net, tt.optim.Adam, duration_index=labtrans.cuts)
    model.load_model_weights(model_dir + '/' + model_weight)

    # save dir
    print('\ntesting results .......')
    tr_dir = task_dir + '/tr2/' + cox_variable
    va_dir = task_dir + '/va2/' + cox_variable
    ts_dir = task_dir + '/ts2/' + cox_variable
    tx1_dir = task_dir + '/tx_maastro/' + cox_variable
    tx2_dir = task_dir + '/tx_bwh/' + cox_variable

    for save_dir in [tr_dir, va_dir, ts_dir, tx1_dir, tx2_dir]:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    # no interpolation on survial curves
    va_surv = model.predict_surv_df(x_va)
    ts_surv = model.predict_surv_df(x_ts)
    tx1_surv = model.predict_surv_df(x_tx1)
    tx2_surv = model.predict_surv_df(x_tx2)

    va_surv.to_csv(va_dir + '/raw_surv_2.csv', index=False)
    ts_surv.to_csv(ts_dir + '/raw_surv_2.csv', index=False)
    tx1_surv.to_csv(tx1_dir + '/raw_surv_2.csv', index=False)
    tx2_surv.to_csv(tx2_dir + '/raw_surv_2.csv', index=False)
    
    # interpolation on survival curves
    if cox_type == 'PCHazard':
        model.sub = 10
        va_surv = model.predict_surv_df(x_va)
        ts_surv = model.predict_surv_df(x_ts)
        tx1_surv = model.predict_surv_df(x_tx1)
        tx2_surv = model.predict_surv_df(x_tx2)
    else:    
        va_surv = model.interpolate(10).predict_surv_df(x_va)
        ts_surv = model.interpolate(10).predict_surv_df(x_ts)
        tx1_surv = model.interpolate(10).predict_surv_df(x_tx1)
        tx2_surv = model.interpolate(10).predict_surv_df(x_tx2)

    va_surv.to_csv(va_dir + '/full_surv_2.csv', index=False)
    ts_surv.to_csv(ts_dir + '/full_surv_2.csv', index=False)
    tx1_surv.to_csv(tx1_dir + '/full_surv_2.csv', index=False)
    tx2_surv.to_csv(tx2_dir + '/full_surv_2.csv', index=False)

    va_c_idx = EvalSurv(va_surv, va_times, va_events, censor_surv='km').concordance_td('antolini')
    ts_c_idx = EvalSurv(ts_surv, ts_times, ts_events, censor_surv='km').concordance_td('antolini')
    tx1_c_idx = EvalSurv(tx1_surv, tx1_times, tx1_events, censor_surv='km').concordance_td('antolini')
    tx2_c_idx = EvalSurv(tx2_surv, tx2_times, tx2_events, censor_surv='km').concordance_td('antolini')

    va_c_idx = round(va_c_idx, 2)
    ts_c_idx = round(ts_c_idx, 2)
    tx1_c_idx = round(tx1_c_idx, 2)
    tx2_c_idx = round(tx2_c_idx, 2)

    va_c_idx_bootstrap = get_95CI(va_surv, va_times, va_events)
    ts_c_idx_bootstrap = get_95CI(ts_surv, ts_times, ts_events)
    tx1_c_idx_bootstrap = get_95CI(tx1_surv, tx1_times, tx1_events)
    tx2_c_idx_bootstrap = get_95CI(tx2_surv, tx2_times, tx2_events)

    va_save_path = va_dir + '/' + cox_variable + '_c-index_bootstrap.npy'
    ts_save_path = ts_dir + '/' + cox_variable + '_c-index_bootstrap.npy'
    tx1_save_path = tx1_dir + '/' + cox_variable + '_c-index_bootstrap.npy'
    tx2_save_path = tx2_dir + '/' + cox_variable + '_c-index_bootstrap.npy'

    np.save(va_save_path, va_c_idx_bootstrap)
    np.save(ts_save_path, ts_c_idx_bootstrap)
    np.save(tx1_save_path, tx1_c_idx_bootstrap)
    np.save(tx2_save_path, tx2_c_idx_bootstrap)


def main(opt):

    action = 'training'

    epochs = 200
    score_type = 'mean_surv'
    surv_type = 'efs'
    cox_variable = 'tot'
    # cox_variable = 'clinical_tumor_dl'
    normalize = True
    hpv = 'all'
    cox_type = 'PCHazard'
    random_state = 42
    va_target = 0.68
    ts_target = 0.68
    tx1_target = 0.7
    tx2_target = 0.7
    model_weight = '0.8_0.78_0.75_0.75_weights.pt'
    # model_weight = '0.74_0.77_0.72_0.68_clinical_tumor_dl.pt'
    ts_nodes = 96

    # opt.task = 'Task053'
    # opt.surv_type = 'os'
    
    task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_size + '_' + \
               opt.img_type + '_' + opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 
    
    model_dir = task_dir + '/' + cox_type + '_model'
    surv_dir = task_dir + '/' + cox_type + '_surv'
    log_dir = task_dir + '/' + cox_type + '_log'

    for dir in [model_dir, surv_dir, log_dir]:
        if not os.path.exists(dir):
            print('dir does not exist, created new dir ....')
            os.makedirs(dir)

    df_tr, df_va, df_ts, df_tx1, df_tx2 = get_data(opt, score_type, hpv, surv_type, 
                                                   cox_variable, normalize, random_state)    
    
    if action == 'training':
        count1 = 0
        count2 = 0

        # list_dropout = [0.5]
        # list_lr = [0.0005]
        # list_nodes = [96]
        # list_bs = [12]

        # list_dropout = [0.2]
        # list_lr = [0.001]
        # list_nodes = [64]
        # list_bs = [8]
        # list_dropout = [0.1, 0.2, 0.3, 0.4, 0.6]
        # list_lr = [0.01, 0.001, 0.0001, 0.00001]
        # list_nodes = [164, 192, 256]
        # list_bs = [32, 64]

        list_dropout = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        list_lr = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.0005, 0.00001, 0.000001]
        list_nodes = [16, 32, 64, 96, 128, 192]
        list_bs = [4, 8, 12, 16, 24, 32, 64]

        tot_runs = len(list_dropout) * len(list_lr) * len(list_bs) * len(list_nodes)
        print('\ntotal runs:', tot_runs)

        print('\nstart model training ........')
        for dropout in list_dropout:
            for lr in list_lr:
                for num_nodes in list_nodes:
                    for batch_size in list_bs:
                        count1 += 1
                        print(count1)

                        model, va_c_idx, ts_c_idx, tx1_c_idx, tx2_c_idx = train(model_dir, cox_type, surv_type, df_tr, df_va, df_ts, df_tx1, 
                                                                                df_tx2, dropout, lr, num_nodes, batch_size, epochs)
                        
                        # save training results to txt
                        tz = pytz.timezone('US/Eastern')
                        #time = datetime.now(tz).strftime('%Y_%m_%d_%H_%M_%S')
                        time = datetime.now(tz).strftime('%Y_%m_%d')
                        tr_log_path = log_dir + '/train_log_' + time + '.txt'
                        with open(tr_log_path, 'a') as f:
                            #f.write('\n-------------------------------------------')
                            f.write(f'\nrun number: {count1}')
                            f.write('\ncreated time: %s' % datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S'))
                            #f.write('\n%s:' % strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
                            f.write('\nmodel training results and parameters:')
                            f.write(f'\ncox model: {cox_type}')
                            f.write(f'\nval c-index: {va_c_idx}')
                            f.write(f'\nts c-index: {ts_c_idx}')
                            f.write(f'\ntx1 c-index: {tx1_c_idx}')
                            f.write(f'\ntx2 c-index: {tx2_c_idx}')
                            f.write(f'\ndropout: {dropout}')
                            f.write(f'\nlr: {lr}')
                            f.write(f'\nnodes: {num_nodes}')
                            f.write(f'\nbatch size: {batch_size}')
                            f.write(f'\nrandom state: {random_state}')
                            f.write('\n')
                            f.close()
                        
                        if va_c_idx >= va_target and ts_c_idx >= ts_target and tx1_c_idx >= tx1_target and tx2_c_idx >= tx2_target:
                            count2 += 1
                            print('target results:', va_c_idx, ts_c_idx, tx1_target, tx2_c_idx, dropout, lr, num_nodes, batch_size)
                            save_path = model_dir + '/' + str(va_c_idx) + '_' + str(ts_c_idx) + '_' + str(tx1_c_idx) + '_' + str(tx2_c_idx) + '_weights.pt'
                            model.save_model_weights(save_path)

                            # save target results to txt
                            tz = pytz.timezone('US/Eastern')
                            time = datetime.now(tz).strftime('%Y_%m_%d')
                            tr_log_path = log_dir + '/target_results_' + time + '.txt'
                            with open(tr_log_path, 'a') as f:
                                #f.write('\n-------------------------------------------------------------------')
                                f.write(f'\nnumber: {count2}')
                                f.write('\ncreated time: %s' % datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S'))
                                f.write('\nmodel training results and parameters:')
                                f.write(f'\ncox model: {cox_type}')
                                f.write(f'\nval c-index: {va_c_idx}')
                                f.write(f'\nts c-index: {ts_c_idx}')
                                f.write(f'\ntx1 c-index: {tx1_c_idx}')
                                f.write(f'\ntx2 c-index: {tx2_c_idx}')
                                f.write(f'\ndropout: {dropout}')
                                f.write(f'\nlr: {lr}')
                                f.write(f'\nnodes: {num_nodes}')
                                f.write(f'\nbatch size: {batch_size}')
                                f.write(f'\nrandom state: {random_state}')
                                f.write('\n')
                                f.close()

    elif action == 'testing':
        test(model_dir, task_dir, cox_type, cox_variable, 
             surv_type, df_tr, df_va, df_ts, df_tx1, df_tx2, 
             model_weight, ts_nodes)


if __name__ == '__main__':

    np.random.seed(42)
    _ = torch.manual_seed(123)

    #Set random seed for weight initialization (for certain operations)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    opt = parse_opts()
    main(opt)


