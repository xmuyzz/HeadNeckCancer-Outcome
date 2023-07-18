import os
import numpy as np
import pandas as pd
from datetime import datetime
from time import localtime, strftime
from datetime import datetime
import pytz


def train_logger(log_dir, surv_type, img_type, cnn_name, model_depth, cox, epoch, batch_size, lr, df_va):
    n_va = df_va.shape[0]
    n_tr = n_va*8
    tz = pytz.timezone('US/Eastern')
    time = datetime.now(tz).strftime('%Y_%m_%d_%H_%M_%S')
    tr_log_path = log_dir + '/training_log_' + time + '.txt'
    with open(tr_log_path, 'w') as f:
        #f.write('\n-------------------------------------------------------------------')
        f.write('\ncreated time: %s' % datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S'))
        #f.write('\n%s:' % strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        f.write('\nmodel and training parameters:')
        f.write('\survival type: %s' % surv_type)
        f.write('\input img type: %s' % img_type)
        f.write('\ncnn_model: %s%s' % (cnn_name, model_depth))
        f.write('\ncox model: %s' % cox)
        f.write('\ntrain size: %s' % n_tr)
        f.write('\nval size: %s' % n_va)
        f.write('\ninitial lr: %s' % lr)
        f.write('\nepoch: %s' % epoch)
        f.write('\nbatch size: %s' % batch_size)
        f.write('\n')
        f.write('\ntraining start ......')
        f.write('\n')
        f.close()
    return tr_log_path
    #print('successfully save train logs.')


def callback_logger(tr_log_path, tr_loss, va_loss, lr, c_index, best_c_index, epoch, epoch_time):
    tz = pytz.timezone('US/Eastern') 
    with open(tr_log_path, 'a') as f:
        #f.write('\n-------------------------------------------------------------------')
        #f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
        f.write('\n%s:' % datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'))
        f.write('\nepoch: %s' % (epoch))
        f.write('\n%s: train loss: %s' % (datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'), tr_loss))
        f.write('\n%s: val loss: %s' % (datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'), va_loss))
        f.write('\n%s: learning rate: %s' % (datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'), lr))
        f.write('\n%s: best c-index: %s' % (datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'), best_c_index))
        f.write('\n%s: current c-index: %s' % (datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'), c_index))
        f.write('\n%s: This epoch took %s s' % (datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f'), epoch_time))
        f.write('\n')
        f.close()
    #print('successfully save train logs.')


def test_logger(save_dir, c_index, mean_HR, median_HR, brier_score, nbll_score, eval_model): 
        
    tz = pytz.timezone('US/Eastern')
    time = datetime.now(tz).strftime('%Y_%m_%d_%H_%M_%S')
    #log_path = save_dir + '/log_' + time + '.txt'
    log_path = save_dir + '/test_log.txt'
    with open(log_path, 'a') as f:
        f.write('\n------------------------------------------')
        f.write('\ncreated time: %s' % datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S'))
        f.write('\neval model: %s' % eval_model)
        f.write('\nc-index: %s' % c_index)
        f.write('\nmean HR: %s' % mean_HR)
        f.write('\nmedian HR: %s' % median_HR)
        f.write('\nbrier-score: %s' % brier_score)
        f.write('\nnbll-score: %s' % nbll_score)
        f.write('\n')
        f.close()
    print('successfully save logs.')
    