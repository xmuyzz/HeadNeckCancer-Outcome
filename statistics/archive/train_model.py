import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from PIL import Image
import torch
import torchtuples as tt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from pycox.models import PCHazard
from pycox.models import LogisticHazard
from pycox.utils import kaplan_meier

from models import ResNet





## load data from a list of numpy array
my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)

tensor_x = torch.Tensor(my_x) # transform to torch tensor
tensor_y = torch.Tensor(my_y)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset) # create your dataloader


## load data from numpy
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

# Let's create 10 RGB images of size 128x128 and 10 labels {0, 1}
data = list(np.random.randint(0, 255, size=(10, 3, 128, 128)))
targets = list(np.random.randint(2, size=(10)))

transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
dataset = MyDataset(data, targets, transform=transform)
dataloader = DataLoader(dataset, batch_size=5)



def train_model(pro_data_dir, ):

    # for reproducability
    np.random.seed(1234)
    _ = torch.manual_seed(1234)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
        )
    mnist_train = datasets.MNIST('.', train=True, download=True,
                             transform=transform)
    mnist_test = datasets.MNIST('.', train=False, transform=transform)
    #_ = plt.imshow(mnist_train[0][0][0].numpy(), cmap='gray')
    
    ## simulate data
    def sim_event_times(mnist, max_time=700):
        digits = mnist.targets.numpy()
        betas = 365 * np.exp(-0.6 * digits) / np.log(1.2)
        event_times = np.random.exponential(betas)
        censored = event_times > max_time
        event_times[censored] = max_time
        return tt.tuplefy(event_times, ~censored)

    sim_train = sim_event_times(mnist_train)
    sim_test = sim_event_times(mnist_test)
    ## visulize data
    for i in range(10):
        idx = mnist_train.targets.numpy() == i
        kaplan_meier(*sim_train.iloc[idx]).rename(i).plot()
    _ = plt.legend()

    ## Our simulated event times are drawn in continuous time, so to apply the LogisticHazard method, 
    ## we need to discretize the observations. This can be done with the label_transform attribute, 
    ## and we here use an equidistant grid with 20 grid points.
    labtrans = LogisticHazard.label_transform(20)
    target_train = labtrans.fit_transform(*sim_train)
    target_test = labtrans.transform(*sim_test)
    ## The disretization grid is
    print(labtrans.cuts)
    # and the dicrete targets are
    print(target_train)

    #-----------------
    # DataLoader 1
    #-----------------
    class MnistSimDatasetSingle(Dataset):
    """
    Simulatied data from MNIST. Read a single entry at a time.
    """
    def __init__(self, mnist_dataset, time, event):
        self.mnist_dataset = mnist_dataset
        self.time, self.event = tt.tuplefy(time, event).to_tensor()

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        if type(index) is not int:
            raise ValueError(f"Need `index` to be `int`. Got {type(index)}.")
        img = self.mnist_dataset[index][0]
        return img, (self.time[index], self.event[index])

    dataset_train = MnistSimDatasetSingle(mnist_train, *target_train)
    dataset_test = MnistSimDatasetSingle(mnist_test, *target_test)
    ## Our dataset gives a nested tuple (img, (idx_duration, event)), meaning the default 
    ## collate in PyTorch does not work. We therefore use tuplefy to stack the tensors instead
    def collate_fn(batch):
        """Stacks the entries of a nested tuple"""
        return tt.tuplefy(batch).stack()
    batch_size = 128
    dl_train = DataLoader(dataset_train, batch_size, shuffle=True, collate_fn=collate_fn)
    dl_test = DataLoader(dataset_test, batch_size, shuffle=False, collate_fn=collate_fn)
    ## If we now investigate a batch, we see that we have the same tuple structure 
    ## (img, (idx_durations, events)) but in a batch of size 128.
    batch = next(iter(dl_train))
    print(batch.shapes())
    print(batch.dtypes())
    
    #-----------------
    # data loader 2
    #-----------------
    ## When working with torchtuples it is typically simpler to read a batch at a times. 
    ## This means that we do not need a collate_fn, and all the logic is in the Dataset. 
    class MnistSimDatasetBatch(Dataset):
        def __init__(self, mnist_dataset, time, event):
            self.mnist_dataset = mnist_dataset
            self.time, self.event = tt.tuplefy(time, event).to_tensor()

        def __len__(self):
            return len(self.time)

        def __getitem__(self, index):
            if not hasattr(index, '__iter__'):
                index = [index]
            img = [self.mnist_dataset[i][0] for i in index]
            img = torch.stack(img)
            return tt.tuplefy(img, (self.time[index], self.event[index]))

    dataset_train = MnistSimDatasetBatch(mnist_train, *target_train)
    dataset_test = MnistSimDatasetBatch(mnist_test, *target_test)
    samp = dataset_train[[0, 1, 3]]
    print(samp.shapes())
    dl_train = tt.data.DataLoaderBatch(dataset_train, batch_size, shuffle=True)
    dl_test = tt.data.DataLoaderBatch(dataset_test, batch_size, shuffle=False)
    batch = next(iter(dl_train))
    print(batch.shapes())
    print(batch.dtypes())

    #---------------
    # CNN
    #---------------
    ## We will use a convolutional network with two convolutional layers, 
    ## global average pooling, and two dense layers. 

    class Net(nn.Module):
        def __init__(self, out_features):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 5, 1)
            self.max_pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 16, 5, 1)
            self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(16, 16)
            self.fc2 = nn.Linear(16, out_features)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.max_pool(x)
            x = F.relu(self.conv2(x))
            x = self.glob_avg_pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    net = Net(labtrans.out_features)
    print(net)

    #--------------------------
    # The Logistic-Hazard Model
    #--------------------------
    model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
    # To verify that the network works as expected we can use the batch from before
    pred = model.predict(batch[0])
    print(pred.shape)
    # train
    callbacks = [tt.cb.EarlyStopping(patience=5)]
    epochs = 50
    verbose = True
    log = model.fit_dataloader(dl_train, epochs, callbacks, verbose, val_dataloader=dl_test)
    _ = log.plot()
    # prediction
    # To predict, we need a data loader that only gives the images and not the targets. 
    # We therefore need to create a new Dataset for this purpose.
    # method 1
    class MnistSimInput(Dataset):
        def __init__(self, mnist_dataset):
            self.mnist_dataset = mnist_dataset

        def __len__(self):
            return len(self.mnist_dataset)

        def __getitem__(self, index):
            img = self.mnist_dataset[index][0]
            return img
    dataset_test_x = MnistSimInput(mnist_test)
    dl_test_x = DataLoader(dataset_test_x, batch_size, shuffle=False)
    print(next(iter(dl_test_x)).shape)
    
    # method 2
    # If you have used the batch method, we can use the method dataloader_input_only 
    # to create this Dataloader from dl_test.
    dl_test_x = tt.data.dataloader_input_only(dl_test)
    print(next(iter(dl_test_x)).shape)
    # We can obtain survival prediction in the regular manner, 
    # and one can include the interpolation if wanted.
    surv = model.predict_surv_df(dl_test_x)
    # We compute the average survival predictions for each digit in the test set
    for i in range(10):
        idx = mnist_test.targets.numpy() == i
        surv.loc[:, idx].mean(axis=1).rename(i).plot()
    _ = plt.legend()
    # and find that they are quite similar to the Kaplan-Meier estimates!
    for i in range(10):
        idx = mnist_test.targets.numpy() == i
        kaplan_meier(*sim_test.iloc[idx]).rename(i).plot()
    _ = plt.legend()

    # Concordance and Brier score
    surv = model.interpolate(10).predict_surv_df(dl_test_x)
    ev = EvalSurv(surv, *sim_test, 'km')
    print(ev.concordance_td())

    time_grid = np.linspace(0, sim_test[0].max())
    ev.integrated_brier_score(time_grid)



    




#    df_train = pd.read_csv(os.path.join(pro_data_dir, 'df_train0.csv'))
#    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_train0.csv'))
#
#    df_train = metabric.read_df()
#    df_test = df_train.sample(frac=0.2)
#    df_train = df_train.drop(df_test.index)
#    df_val = df_train.sample(frac=0.2)
#    df_train = df_train.drop(df_val.index)
#
#    num_durations = 10
#    labtrans = PCHazard.label_transform(num_durations)
#    get_target = lambda df: (df['duration'].values, df['event'].values)
#    y_train = labtrans.fit_transform(*get_target(df_train))
#    y_val = labtrans.transform(*get_target(df_val))
#
#    train = (x_train, y_train)
#    val = (x_val, y_val)
#
#    # We don't need to transform the test labels
#    durations_test, events_test = get_target(df_test)
#    print(type(labtrans))
#    print(y_train)
#
#    in_features = x_train.shape[1]
#    out_features = labtrans.out_features
#
#    net = torch.nn.Sequential(
#         torch.nn.Linear(in_features, 32),
#         torch.nn.ReLU(),
#         torch.nn.BatchNorm1d(32),
#         torch.nn.Dropout(0.1),
#         torch.nn.Linear(32, 32),
#         torch.nn.ReLU(),
#         torch.nn.BatchNorm1d(32),
#         torch.nn.Dropout(0.1),
#         torch.nn.Linear(32, out_features)
#        )
#    #cnn = resnet50(block, layers, sample_size, sample_duration, shortcut_type, num_classes)
#
#    model = PCHazard(net, tt.optim.Adam, duration_index=labtrans.cuts)
#
#    batch_size = 256
#    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=8)
#    _ = lr_finder.plot()
#    lr_finder.get_best_lr()
#    model.optimizer.set_lr(0.01)
#    epochs = 100
#    callbacks = [tt.callbacks.EarlyStopping()]
#    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
#    _ = log.plot()
#    surv = model.predict_surv_df(x_test)
#    surv.iloc[:, :5].plot(drawstyle='steps-post')
#    plt.ylabel('S(t | x)')
#    _ = plt.xlabel('Time')
#    model.sub = 10
#    surv = model.predict_surv_df(x_test)
#    surv.iloc[:, :5].plot(drawstyle='steps-post')
#    plt.ylabel('S(t | x)')
#    _ = plt.xlabel('Time')
#
#    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
#    ev.concordance_td('antolini')
#
#    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
#    ev.brier_score(time_grid).plot()
#    plt.ylabel('Brier score')
#    _ = plt.xlabel('Time')
#
#    ev.nbll(time_grid).plot()
#    plt.ylabel('NBLL')
#    _ = plt.xlabel('Time')
#
#    ev.integrated_brier_score(time_grid)
#
#    ev.integrated_nbll(time_grid)
