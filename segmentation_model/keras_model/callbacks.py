import keras
import numpy as np
from statistics import median
from keras import backend as K
from metrics import dice

class model_callbacks(keras.callbacks.Callback):

    def __init__(self, model, RUN, dir_name, val_data):
        self.model_to_save = model
        self.run = RUN
        self.dir_name = dir_name
        self.val_data = val_data
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = 1000
        self.val_dice_scores = []
        self.best_val_dice = 0

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)
        lr = float(K.get_value(self.model.optimizer.lr))
        # lr = self.model.optimizer.lr
        self.learning_rates.append(lr)
        
        for i in range(0,len(self.val_data[0])):
            image = self.val_data[0][i]
            label = self.val_data[1][i]
            label = label.reshape(1,*label.shape)
            label_predict = self.model_to_save.predict(image.reshape(1,*image.shape))
            label_predict = np.squeeze(label_predict)
            label_predict[label_predict<0.5]=0
            label_predict[label_predict>=0.5]=1
            
            dice_score = dice(np.squeeze(label), label_predict)
            self.val_dice_scores.append(dice_score)
        med_dice = median(self.val_dice_scores)
        print("validation average dice score: ", med_dice)
        # save model
        if val_loss < self.best_val_loss:
            self.model_to_save.save(self.dir_name + '/{}.h5'.format(self.run))
            self.best_val_loss = val_loss
            print("best loss model saved.")
        elif med_dice > self.best_val_dice:
            self.model_to_save.save(self.dir_name + '/{}_dsc.h5'.format(self.run))
            self.best_val_dice = med_dice
            print("best dice score model saved")

        # save logs (overwrite)
        np.save(self.dir_name + '/{}_loss.npy'.format(self.run), self.losses)
        np.save(self.dir_name + '/{}_val_loss.npy'.format(self.run), self.val_losses)
        np.save(self.dir_name + '/{}_lr.npy'.format(self.run), self.learning_rates)
        np.save(self.dir_name + '/{}_dice_scores.npy'.format(self.run), med_dice)
        
    def on_train_end(self, logs):
        self.model_to_save.save(self.dir_name + '/{}_final.h5'.format(self.run))
