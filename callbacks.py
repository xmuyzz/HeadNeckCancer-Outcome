import torch
import numpy as np
import torchtuples as tt
from pytorch_lightning.callbacks import Callback
from pycox.evaluation import EvalSurv


class Concordance(tt.cb.MonitorMetrics):

    """
    Callback function with concordance-index
    """

    def __init__(self, save_dir, run, df_tune, dl_tune_cb, per_epoch=1, verbose=True):
        super().__init__(per_epoch)
        self.save_dir = save_dir
        self.verbose = verbose
        self.dl_tune_cb = dl_tune_cb
        self.df_tune = df_tune
        self.best_c_index = 0.7
        self.c_indices = []
        self.run = run

    def on_epoch_end(self, logs={}):
        # get c-index
        time = self.df_tune['death_time'].to_numpy()
        event = self.df_tune['death_event'].to_numpy()
        super().on_epoch_end()
        if self.epoch % self.per_epoch == 0:
            surv = self.model.predict_surv_df(self.dl_tune_cb)
            ev = EvalSurv(surv, time, event)
            c_index = ev.concordance_td()
            c_index = np.around(c_index, 3)
            self.append_score('c-index', c_index)
            self.c_indices.append(c_index)
            if self.verbose:
                print('c-index:', c_index)
        med_c_index = np.median(self.c_indices)
        #print('tune median c-index: ', med_c_index)
        # save model
        if med_c_index > self.best_c_index:
            self.model.save_model_weights(self.save_dir + '/{}_c-index.pt'.format(self.run))
            self.best_cindex = med_c_index
            print("best c index model saved")
        # save logs (overwrite)
        np.save(self.save_dir + '/{}_c-index.npy'.format(self.run), med_c_index)

    def get_last_score(self):
        #return self.scores['c_index']['score'][-1]
        return self.c_indices[-1]

    def on_train_end(self):
        self.model.save_model_weights(self.save_dir + '/{}_final.pt'.format(self.run))


class LRScheduler(tt.cb.Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self):
        self.scheduler.step()
        stop_signal = False
        return stop_signal






class callback(Callback):

    def __init__(self, model, save_dir, val_data, run):
        self.model_to_save = model
        self.save_dir = save_dir
        self.val_data = val_data
        self.losses = []
        self.val_losses = []
        self.best_val_loss = 1000
        self.cindices = []
        self.best_cindex = 0

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)

        for i in range(0, len(self.val_data[0])):
            img = self.val_data[0][i]
            time = self.val_data[1][0][i]
            event = self.val_data[1][1][i]
            surv = self.model.predict_surv_df(img)
            ev = EvalSurv(surv, time, event)
            cindex = ev.concordance_td()
            self.cindices.append(cindex)
        med_cindex = median(self.c_indices)
        print("validation median c index: ", med_cindex)
        
        # save model
        if val_loss < self.best_val_loss:
            self.model_to_save.save(self.save_dir + '/{}.h5'.format(self.run))
            self.best_val_loss = val_loss
            print("best loss model saved.")
        elif med_cindex > self.best_cindex:
            self.model_to_save.save(self.save_dir + '/{}_dsc.h5'.format(self.run))
            self.best_cindex = med_cindex
            print("best c index model saved")

        # save logs (overwrite)
        np.save(self.save_dir + '/{}_loss.npy'.format(self.run), self.losses)
        np.save(self.save_dir + '/{}_val_loss.npy'.format(self.run), self.val_losses)
        np.save(self.save_dir + '/{}_cindices.npy'.format(self.run), med_cindex)

    def on_train_end(self, logs):
        self.model_to_save.save(self.save_dir + '/{}_final.h5'.format(self.run))

