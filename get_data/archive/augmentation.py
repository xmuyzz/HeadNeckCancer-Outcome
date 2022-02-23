from monai.utils import first
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    Flipd,
    RandAffined,
    )

from monai.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import os
import torch
import nibabel as nib
from tqdm import tqd

def augmentation():

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
    # the path of ixi IXI-T1 dataset
    data_path = os.sep.join([".", "workspace", "data", "medical", "ixi", "IXI-T1"])
    images = [
        "IXI314-IOP-0889-T1.nii.gz",
        "IXI249-Guys-1072-T1.nii.gz",
        "IXI609-HH-2600-T1.nii.gz",
        "IXI173-HH-1590-T1.nii.gz",
        "IXI020-Guys-0700-T1.nii.gz",
        "IXI342-Guys-0909-T1.nii.gz",
        "IXI134-Guys-0780-T1.nii.gz",
        "IXI577-HH-2661-T1.nii.gz",
        "IXI066-Guys-0731-T1.nii.gz",
        "IXI130-HH-1528-T1.nii.gz",
        "IXI607-Guys-1097-T1.nii.gz",
        "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz",
        "IXI344-Guys-0905-T1.nii.gz",
        "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz",
        "IXI253-HH-1694-T1.nii.gz",
        "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz",
        "IXI585-Guys-1130-T1.nii.gz",
    ]

    images = [os.sep.join([data_path, f]) for f in images]
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    train_files = [{"img": img, "label": label} for img, label in zip(images[:10], labels[:10])]
    val_files = [{"img": img, "label": label} for img, label in zip(images[-10:], labels[-10:])]

    # Define transforms for image
    train_transforms = Compose([
        LoadImaged(keys=["img"]),
        AddChanneld(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(96, 96, 96)),
        RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
        EnsureTyped(keys=["img"]),
        ])

    val_transforms = Compose([
            LoadImaged(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            EnsureTyped(keys=["img"]),
            ])

    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["label"])

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    auc_metric = ROCAUCMetric()


def save_nifti(in_image, in_label, out, index = 0):

    # Convert the torch tensors into numpy array
    volume = np.array(in_image.detach().cpu()[0, :, :, :], dtype=np.float32)
    lab = np.array(in_label.detach().cpu()[0, :, :, :], dtype=np.float32)

    # Convert the numpy array into nifti file
    volume = nib.Nifti1Image(volume, np.eye(4))
    lab = nib.Nifti1Image(lab, np.eye(4))

    # Create the path to save the images and labels
    path_out_images = os.path.join(out, 'Images')
    path_out_labels = os.path.join(out, 'Labels')

    # Make directory if not existing
    if not os.path.exists(path_out_images):
        os.mkdir(path_out_images)
    if not os.path.exists(path_out_labels):
        os.mkdir(path_out_labels)

    path_data = os.path.join(out, 'Images')
    path_label = os.path.join(out, 'Labels')
    nib.save(volume, os.path.join(path_data, f'patient_generated_{index}.nii.gz'))
    nib.save(lab, os.path.join(path_label, f'patient_generated_{index}.nii.gz'))

    print(f'patient_generated_{index} is saved', end='\r')


data_dir = 'D:/3_Stage/ALL_THE_DATA/fixed_data_03_august/all_together'
train_images = sorted(glob.glob(os.path.join(data_dir, "TrainData", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "TrainLabels", "*.nii.gz")))

train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

original_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,),
        ToTensord(keys=["image", "label"]),
    ]
)

generat_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,),
        RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
        RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
        RandGaussianNoised(keys='image', prob=0.5),
        ToTensord(keys=["image", "label"]),
    ]
)

original_ds = Dataset(data=train_files, transform=original_transforms)
original_loader = DataLoader(original_ds, batch_size=1)
original_patient = first(original_loader)

generat_ds = Dataset(data=train_files, transform=generat_transforms)
generat_loader = DataLoader(generat_ds, batch_size=1)
generat_patient = first(generat_loader)

number_slice = 30
plt.figure("display", (12, 6))
plt.subplot(1, 2, 1)
plt.title(f"Original patient slice {number_slice}")
plt.imshow(original_patient["image"][0, 0, :, :, number_slice], cmap="gray")
plt.subplot(1, 2, 2)
plt.title(f"Generated patient slice {number_slice}")
plt.imshow(generat_patient["image"][0, 0, :, :, number_slice], cmap="gray")


plt.show()

output_path = 'D:/3_Stage/ALL_THE_DATA/generated_data'
number_runs = 10
for i in range(number_runs):
    name_folder = 'generated_data_' + str(i)
    os.mkdir(os.path.join(output_path, name_folder))
    output = os.path.join(output_path, name_folder)
    check_ds = Dataset(data=train_files, transform=generat_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    for index, patient in enumerate(check_loader):
        save_nifti(patient['image'], patient['label'], output, index)
    print(f'step {i} done')

    m
