
import torch
import torchtuples as tt


class Concordance(tt.cb.MonitorMetrics):

    """
    Callback function with concordance index;
    """

    def __init__(self, x, durations, events, per_epoch=1, verbose=True):
        super().__init__(per_epoch)
        self.x = x
        self.durations = durations
        self.events = events
        self.verbose = verbose

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.epoch % self.per_epoch == 0:
            surv = self.model.interpolate(10).predict_surv_df(self.x)
            ev = EvalSurv(surv, self.durations, self.events)
            concordance = ev.concordance_td()
            self.append_score('concordance', concordance)
            if self.verbose:
                print('concordance:', concordance)

    def get_last_score(self):
        return self.scores['concordance']['score'][-1]

concordance = Concordance(x_test, durations_test, events_test)
early_stopping =  tt.callbacks.EarlyStopping(get_score=concordance.get_last_score,
                                             minimize=False)
callbacks = [concordance, early_stopping]
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
