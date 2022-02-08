import numpy as np
import torch
from go_models.data_loader import DataLoader
from go_models.train import train
from go_models.evaluate import evaluate
from go_models.get_cnn_model import get_cnn_model
from go_models.get_cox_model import get_cox_model


if __name__ == '__main__':

    data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated'
    proj_dir = '/mnt/HDD_6TB/HN_Outcome'
    aimlab_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    cnn_name = 'resnet'
    model_depth = 34  # [10, 18, 34, 50, 101, 152, 200]
    n_classes = 20
    in_channels = 3
    batch_size = 4
    epochs = 1
    lr = 0.0001
    num_durations = 20
    verbose = True
    _cox_model = 'LogisticHazard'
    #cox_model = 'PCHazard'
    load_model = 'model'
    evaluate_only = True
    
    np.random.seed(1234)
    _ = torch.manual_seed(1234)


    dl_train, dl_tune, dl_val= DataLoader(
        proj_dir=proj_dir,
        batch_size=batch_size,
        _cox_model=_cox_model,
        num_durations=num_durations
        )
    
    cnn_model = get_cnn_model(
        cnn_name=cnn_name, 
        model_depth=model_depth, 
        n_classes=n_classes, 
        in_channels=in_channels
        )

    cox_model = get_cox_model(
        proj_dir=proj_dir,
        cnn_model=cnn_model,
        _cox_model=_cox_model,
        lr=lr
        )

    if not evaluate_only:
        train(
            proj_dir=proj_dir,
            lab_drive_dir=lab_drive_dir,
            cox_model=cox_model,
            epochs=epochs,
            verbose=verbose,
            dl_train=dl_train,
            dl_tune=dl_tune,
            dl_val=dl_val
            )

    evaluate(
        proj_dir=proj_dir,
        lab_drive_dir=lab_drive_dir,
        cox_model=cox_model,
        load_model=load_model,
        dl_val=dl_val
        )

