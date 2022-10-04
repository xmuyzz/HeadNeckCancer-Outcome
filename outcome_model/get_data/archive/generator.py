import numpy as np
import os
import glob
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

def data_generator():
    
    ## test dataset
    df_test = pd.read_csv(os.path.join(pro_data_dir, 'df_test.csv'))

    ## k-fold cross validation
    df_development = pd.read_csv(os.path.join(pro_data_dir, 'df_development.csv'))
    df_test = pd.read_csv(os.path.join(pro_data_dir, 'df_test.csv'))
    kf = KFolod(n_splits=5)
    df_train, df_val = kf.split(df_development)

    data_loader = DataLoader(
        training_dataset, 
        batch_size=sets.batch_size, 
        shuffle=True, 
        num_workers=sets.num_workers, 
        pin_memory=sets.pin_memory
        )



import torch
from my_classes import Dataset


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

# Datasets
partition = # IDs
labels = # Labels

# Generators
training_set = Dataset(partition['train'], labels)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        [...]

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            [...]
