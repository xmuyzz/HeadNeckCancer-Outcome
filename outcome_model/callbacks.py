import torch
import os
import numpy as np
import torchtuples as tt
#from pytorch_lightning.callbacks import Callback
from pycox.evaluation import EvalSurv
from matplotlib.pylab import plt
from logger import callback_logger
from time import localtime, strftime
import timeit
from time import time


class concordance_callback(tt.cb.MonitorMetrics):
    """
    Callback function with concordance-index
    Args:
        save_dir {path} -- save dir;
        df_tune {pd.df} -- tune data df;
        dl_tune_cb {torch dataloader} -- tuning dataloader;
        target_c_index {float} -- target c-index;
        cnn_name {str} -- cnn model name;
        model_depth {int} -- cnn model depth;
        lr {float} -- learning rate;
    """
    def __init__(self, task_dir, tr_log_path, dl_bl, dl_cb, df_va, cnn_name, cox, model_depth, lr_scheduler, optimizer, 
                 target_c_index, target_loss, gauss_prob, rot_prob, flip_prob):
        super().__init__(per_epoch=1)
        self.task_dir = task_dir
        self.tr_log_path = tr_log_path
        self.dl_bl = dl_bl
        self.dl_cb = dl_cb
        self.df_va= df_va
        self.cnn_name = cnn_name
        self.cox = cox
        self.model_depth = model_depth
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.target_c_index = target_c_index
        self.target_va_loss = target_loss
        self.lrs = []
        self.c_indexs = []
        self.gauss_prob = gauss_prob
        self.rot_prob = rot_prob
        self.flip_prob = flip_prob
        self.model_dir = self.task_dir + '/models'
        self.log_dir = self.task_dir + '/logs'
        self.metric_dir = self.task_dir + '/metrics'

    # def on_batch_end(self, log={}):
    #     loss = self.log.get('loss')
    #     self.tr_losses.append(self.log.get('loss'))
    #     print('loss:', loss)

    def on_fit_start(self):
        self.times = []

    def on_epoch_start(self):
        self.start_time = time()

    def on_epoch_end(self):
        #self.epoch_time = self.times.append(time.time() - self.start)
        #print('this epoch takes %s seconds.' % self.epoch_time)
        tr_losses = self.model.train_metrics.scores['loss']['score']
        va_losses = self.model.val_metrics.scores['loss']['score']
        tr_losses = np.round(tr_losses, 3)
        va_losses = np.round(va_losses, 3)
        tr_loss = tr_losses[-1]
        va_loss = va_losses[-1]
        # lr
        lr = self.optimizer.param_groups[0]['lr']
        #lr = self.lr_scheduler.get_last_lr()[0]
        #lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
        print('lr:', lr)
        self.lrs.append(lr)

        # get c-index
        time = self.df_va['time'].to_numpy()
        event = self.df_va['event'].to_numpy()
        super().on_epoch_end()
        #c_indexs = []
        if self.epoch % self.per_epoch == 0:
            if self.cox == 'CoxPH':
                baseline_data = tt.tuplefy([data for data in self.dl_bl]).cat()
                _ = self.model.compute_baseline_hazards(*baseline_data)
            if self.cox == 'PCHazard':
                self.model.sub = 10
                surv = self.model.predict_surv_df(self.dl_cb)
                print(surv)
                print(time)
                print(event)
                ev = EvalSurv(surv, time, event, censor_surv='km')
                c_index = ev.concordance_td('antolini')
                c_index = np.around(c_index, 3)
                print('current c-index:', c_index)
                self.append_score('c-index', c_index)
                self.c_indexs.append(c_index)
            elif self.cox in ['LogisticHazard', 'MTLR', 'PMF', 'DeepHit']:
                surv = self.model.interpolate(10).predict_surv_df(self.dl_cb)

            ev = EvalSurv(surv, time, event, censor_surv='km')
            c_index = ev.concordance_td('antolini')
            c_index = np.around(c_index, 3)
            print('current c-index:', c_index)
            self.append_score('c-index', c_index)
            self.c_indexs.append(c_index)
            #self.lrs.append(self.lr)
        best_c_index = np.amax(self.c_indexs)
        best_va_loss = np.amin(va_losses)
        print('best c_index:', best_c_index)
        
        # save best models
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if c_index >= best_c_index:
            self.model.save_model_weights(self.model_dir + '/weights_best_cindex.pt')
            print('best c-index model weights saved!')
        if c_index > self.target_c_index:
            self.model.save_model_weights(self.model_dir + '/weights_target_cindex.pt')
            print('target c-index model weights saved!')
        if va_loss <= best_va_loss:
            self.model.save_model_weights(self.model_dir + '/weights_best_loss.pt')
            print('best loss model weights saved!')

        # save logs (overwrite)
        np.save(self.metric_dir + '/tr_losses.npy', tr_losses)
        np.save(self.metric_dir + '/va_losses.npy', va_losses)
        np.save(self.metric_dir + '/c_indexs.npy', self.c_indexs)
        np.save(self.metric_dir + '/lrs.npy', self.lrs)

        # save train log and upodate every epoch
        plot_loss = True
        if plot_loss:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            #ax.set_aspect('equal')
            epoch = len(tr_losses) - 1
            plt.plot(tr_losses, color='red', linewidth=2, label='tr_loss')
            plt.plot(va_losses, color='green', linewidth=2, label='va_loss')
            plt.plot(self.c_indexs, color='blue', linewidth=2, label='c-index')
            plt.xlim([0, epoch+1])
            plt.ylim([0, 1])
            if epoch < 20:
                interval = 1
            elif epoch >= 20 and epoch < 50:
                interval = 5
            elif epoch >= 50:
                interval = 10
            x = np.arange(0, epoch+1, interval, dtype=int).tolist()
            plt.xticks(x, fontsize=12, fontweight='bold')
            #plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8], fontsize=12, fontweight='bold')
            plt.yticks([0, 1, 2, 3, 4, 5], fontsize=12, fontweight='bold')
            plt.xlabel('Epoch', fontweight='bold', fontsize=12)
            plt.ylabel('C-Index/Loss', fontweight='bold', fontsize=12)
            plt.legend(loc='upper right', prop={'size': 12, 'weight': 'bold'})
            plt.grid(True)
            plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
            log_path = self.log_dir + '/loss_' + str(self.gauss_prob) + '_' + \
                str(self.rot_prob) + '_' + str(self.flip_prob) + '.jpg'
            plt.savefig(log_path, format='jpg', dpi=200)
            plt.close()
        
        plot_cindex = True
        if plot_cindex:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            #ax.set_aspect('equal')
            epoch = len(tr_losses) - 1
            plt.plot(self.c_indexs, color='blue', linewidth=2, label='c-index')
            plt.xlim([0, epoch+1])
            plt.ylim([0, 1])
            if epoch < 20:
                interval = 1
            elif epoch >= 20 and epoch < 50:
                interval = 5
            elif epoch >= 50:
                interval = 10
            x = np.arange(0, epoch+1, interval, dtype=int).tolist()
            plt.xticks(x, fontsize=12, fontweight='bold')
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12, fontweight='bold')
            plt.xlabel('Epoch', fontweight='bold', fontsize=12)
            plt.ylabel('C-Index', fontweight='bold', fontsize=12)
            plt.legend(loc='upper right', prop={'size': 12, 'weight': 'bold'})
            plt.grid(True)
            plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
            log_path = self.log_dir + '/c-indx_' + str(self.gauss_prob) + '_' + str(self.rot_prob) + \
                '_' + str(self.flip_prob) + '.jpg'
            plt.savefig(log_path, format='jpg', dpi=200)
            plt.close()

        # calcualte time for each epoch
        import time
        epoch_time = round(time.time() - self.start_time, 6)
        print('epoch time: %s s' %round(epoch_time, 1))
        print('\n')

        # save training loggers
        save_logger = True
        if save_logger:
            callback_logger(tr_log_path=self.tr_log_path, 
                           tr_loss=tr_loss, 
                           va_loss=va_loss, 
                           lr=lr, 
                           c_index=c_index, 
                           best_c_index=best_c_index, 
                           epoch=epoch, 
                           epoch_time=epoch_time)

    def get_last_score(self):
        return self.c_indexs[-1]

    def get_scores(self):
        return self.c_indexs

    def on_fit_end(self):
        super().on_epoch_end()
        self.model.save_net(self.model_dir + '/weights_final.pt')


class LRScheduler(tt.cb.Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler
    def on_epoch_end(self):
        self.scheduler.step()
        stop_signal = False
        return stop_signal


# class callback(tt.cb.Callback):
#     def __init__(self, log_dir, model):
#         self.log_dir = log_dir
#         self.model = model
#         self.tr_losses = []
#         self.va_losses = []

#     def on_epoch_end(self, epoch, logs=None):
#         va_loss = logs.get('val_loss')
#         self.va_losses.append(va_loss)
#         # save logs (overwrite)
#         np.save(self.save_dir + '/tr_loss.npy', self.tr_losses)
#         np.save(self.save_dir + '/va_loss.npy', self.va_losses)
#         print('tr loss:', self.tr_losses)
#         print('va loss:', self.va_losses)


# class callback(Callback):
#     def __init__(self, model, save_dir, val_data, run):
#         self.model_to_save = model
#         self.save_dir = save_dir
#         self.val_data = val_data
#         self.losses = []
#         self.val_losses = []
#         self.best_val_loss = 1000
#         self.cindices = []
#         self.best_cindex = 0

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))

#     def on_epoch_end(self, epoch, logs=None):
#         val_loss = logs.get('val_loss')
#         self.val_losses.append(val_loss)

#         for i in range(0, len(self.val_data[0])):
#             img = self.val_data[0][i]
#             time = self.val_data[1][0][i]
#             event = self.val_data[1][1][i]
#             surv = self.model.predict_surv_df(img)
#             ev = EvalSurv(surv, time, event)
#             cindex = ev.concordance_td()
#             self.cindices.append(cindex)
#         med_cindex = median(self.c_indices)
#         print("validation median c index: ", med_cindex)
#         # save model
#         if val_loss < self.best_val_loss:
#             self.model_to_save.save(self.save_dir + '/{}.h5'.format(self.run))
#             self.best_val_loss = val_loss
#             print("best loss model saved.")
#         elif med_cindex > self.best_cindex:
#             self.model_to_save.save(self.save_dir + '/{}_dsc.h5'.format(self.run))
#             self.best_cindex = med_cindex
#             print("best c index model saved")
#         # save logs (overwrite)
#         np.save(self.save_dir + '/{}_loss.npy'.format(self.run), self.losses)
#         np.save(self.save_dir + '/{}_val_loss.npy'.format(self.run), self.val_losses)
#         np.save(self.save_dir + '/{}_cindices.npy'.format(self.run), med_cindex)

#     def on_train_end(self, logs):
#         self.model_to_save.save(self.save_dir + '/{}_final.h5'.format(self.run))

