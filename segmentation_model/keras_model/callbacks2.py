import keras
import numpy as np
from statistics import median
from keras import backend as K
from metrics import dice


class model_callbacks(keras.callbacks.Callback):

    def __init__(self, model, log_dir, val_data, n_labels):
        self.model_to_save = model
        self.log_dir = log_dir
        self.val_data = val_data
        self.n_labels = n_labels
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = 1000
        self.val_dices = []
        self.best_val_dice = 0

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        tr_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        #self.tr_losses.append(tr_loss)
        self.val_losses.append(val_loss)
        lr = float(K.get_value(self.model.optimizer.lr))
        # lr = self.model.optimizer.lr
        self.learning_rates.append(lr)
        val_dices = []
        #for j in range(2):
        for i in range(0, len(self.val_data[0])):
            image = self.val_data[0][i]
            label = self.val_data[1][i]
            label = label.reshape(1, *label.shape)
            pred = self.model_to_save.predict(image.reshape(1, *image.shape))
            #print('pred:', pred.shape)
            pred = np.squeeze(pred)
            #print('pred:', pred.shape)
            pred[pred<0.5] = 0
            pred[pred>=0.5] = 1
            #label = np.where(label == j+1, 1, 0)
            #pred = np.where(pred == j+1, 1, 0)
            #print('label:', label.shape)
            #print('pred:', pred.shape)
            val_dice = dice(np.squeeze(label), pred)
            self.val_dices.append(val_dice)
            val_dices.append(val_dice)
        epoch_med_dice = median(val_dices)
        val_med_dice = median(self.val_dices)
        print('epcoh:', epoch)
        print('train loss:', round(tr_loss, 3), 'val loss:', round(val_loss, 3))
        print('median dice:', epoch_med_dice)
        print('val dices:', val_dices)
        #med_dices.append(med_dice)
            #print('class %d dice score: %d:' % (j+1, med_dice))
            #print(j+1, med_dice)
        #print('label 1 dice:', med_dices[0])
        #print('label 2 dice:', med_dices[1])
        # save model
        if val_loss < self.best_val_loss:
            self.model_to_save.save(self.log_dir + '/best_loss_model.h5')
            self.best_val_loss = val_loss
            print('best loss model saved.')
        elif val_med_dice > self.best_val_dice:
            self.model_to_save.save(self.log_dir + '/best_dice_model.h5')
            self.best_val_dice = val_med_dice
            print('best dice score model saved.')

        # save logs (overwrite)
        np.save(self.log_dir + '/train_loss.npy', self.losses)
        np.save(self.log_dir + '/val_loss.npy', self.val_losses)
        np.save(self.log_dir + '/lr.npy', self.learning_rates)
        np.save(self.log_dir + '/dice_scores.npy', val_med_dice)
        
    def on_train_end(self, logs):
        self.model_to_save.save(self.log_dir + '/final_model.h5')






