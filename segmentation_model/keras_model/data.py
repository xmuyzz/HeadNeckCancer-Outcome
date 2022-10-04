import numpy as np
import pandas as pd
import SimpleITK as sitk

df = pd.read_csv("/mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTAION_DATA/data.csv", index_col=0)
print("data.csv shape :: ", df.shape)

###### 1
# maastro_train = df[(df["dataset"]=="maastro")][:375]
# harvart_rt_train = df[(df["dataset"]=="harvard-rt") & (df["topcoder_split"]=="train")][:206]
# maastro_tune = df[(df["dataset"]=="maastro")][375:]
# harvard_rt_tune = df[(df["dataset"]=="harvard-rt") & (df["topcoder_split"]=="train")][206:]
# test = df[(df["dataset"]=="harvard-rt") & (df["topcoder_split"]=="tune")]
# data_split = {
#     "train": pd.concat([maastro_train, harvart_rt_train]),
#     "tune" : pd.concat([maastro_tune, harvard_rt_tune]),
#     "test" : test,
# }


######## 2
train = df[(df["dataset"]=="harvard-rt")&(df["topcoder_split"]=="train")][:200]
tune = df[(df["dataset"]=="harvard-rt")&(df["topcoder_split"]=="train")][200:]
test = df[(df["dataset"]=="harvard-rt") & (df["topcoder_split"]=="tune")]
data_split = {
    "train": train,
    "tune" : tune,
    "test" : test
}


###### 3
# train = df[(df["dataset"]=="maastro")][:380]
# tune = df[(df["dataset"]=="maastro")][380:]
# test = df[(df["dataset"]=="harvard-rt") & (df["topcoder_split"]=="tune")]
#
# data_split = {
#     "train": train,
#     "tune" : tune,
#     "test" : test
# }

def print_shape(obj, mode):
    print ("{} image shape :: {} \n{} label shape :: {}".format(
        mode, obj["images"].shape,
        mode, obj["labels"].shape))

def get_arr(path_to_nrrd, mode, model_input_size):
    """
    Reads a nrrd file and spits out a numpy array.
    path_to_nrrd: path_to_nrrd
    mode: train or tune
    model_input_size: tuple of model input
    """
    sitk_image = sitk.ReadImage(path_to_nrrd)
    arr = sitk.GetArrayFromImage(sitk_image)
    if mode == "tune":
        arr = format_arr(arr, model_input_size) # If TRAIN set, the formatting will be done in the generator ( so that there can be random translated crops)
    return arr

def crop_arr(arr, model_input_size):
    start_z = arr.shape[0]//2 - model_input_size[0]//2
    start_y = arr.shape[1]//2 - model_input_size[1]//2
    start_x = arr.shape[2]//2 - model_input_size[2]//2
    #
    arr = arr[start_z:start_z+model_input_size[0],
              start_y:start_y+model_input_size[1],
              start_x:start_x+model_input_size[2]]
    return arr

def format_arr(arr, model_input_size):
    """
    Used for test mode. Crops and reshapes array.
    Also remaps image values.
    """
    arr = crop_arr(arr, model_input_size)
    arr = arr.reshape(1, *arr.shape)
    return arr

def assertions(arr_image, arr_label, dataset, patient_id):
    assert arr_image.shape == arr_label.shape, "image and label do not have the same shape."
    assert arr_label.min() == 0, "label min is not 0 @ {}_{}".format(dataset, patient_id)
    assert arr_label.max() == 1, "label max is not 1 @ {}_{}".format(dataset, patient_id)
    assert len(np.unique(arr_label))==2, "lenght of label unique vals is not 2 @ {}_{}".format(dataset, patient_id)

def generate_train_tune_data(data_split, mode, model_input_size, task):
    """
    Used for training and tuning only.
    data_split: dictionary of train, tune, and test split.
    mode: train, tune, or test
    """
    if task == "LOCALIZATION":
        VERSION = "interpolated_resized_rescaled"
    elif task == "SEGMENTATION":
        VERSION = "interpolated_roi_gt"
    images = []
    labels = []
    for idx, patient in data_split[mode].iterrows():
        dataset = patient["dataset"]
        patient_id = patient["patient_id"]
        # get arr
        arr_image = get_arr(patient["image_"+VERSION], mode, model_input_size)
        arr_image = np.interp(arr_image,[-1024,3071],[0,1]) # From CT range to convert to between 0,1
        arr_label = get_arr(patient["label_"+VERSION], mode, model_input_size) # This should already be between 0 and 1
        # assertions
        assertions(arr_image, arr_label, dataset, patient_id) # Sanity check
        # append to list
        images.append(arr_image) # generate a list of numpy arrays
        labels.append(arr_label)
        print ("{}_{}_{}".format(idx, dataset, patient_id))
    print("-------------")
    return {
            "images": np.array(images),
            "labels": np.array(labels)
           }

def generate_test_data(data_split, model_input_size, task, MULTI_PREDICTION):
    """
    Used for testing only. The image sitk object info is needed during test time. To avoid reading the image nrrd twice, it is read here.
    ###Very similar to the above function in "tune" mode###
    """
    if task == "LOCALIZATION":
        VERSION = "interpolated_resized_rescaled"
    elif task == "SEGMENTATION":
        VERSION = "interpolated_roi_gt"
    test = []
    for idx, patient in data_split["test"].iterrows():
        dataset = patient["dataset"]
        patient_id = patient["patient_id"]
        # get image
        image_sitk_obj = sitk.ReadImage(patient["image_"+VERSION])
        arr_image = sitk.GetArrayFromImage(image_sitk_obj)
        arr_image_interp = np.interp(arr_image,[-1024,3071],[0,1])
        if not MULTI_PREDICTION:
            arr_image_interp = format_arr(arr_image_interp, model_input_size)
        # get label
        label_sitk_obj = sitk.ReadImage(patient["label_"+VERSION])
        arr_label = sitk.GetArrayFromImage(label_sitk_obj)
        # assertions
        assertions(arr_image, arr_label, dataset, patient_id)
        # append to list
        test.append(
         {"patient_id": patient_id,
          "dataset": dataset,
          "image_sitk_obj": image_sitk_obj,
          "image": arr_image_interp,
          "label_sitk_obj": label_sitk_obj,
          # "label" not used because test files used array from label_sitk_obj
          }
        )
        print ("{}_{}_{}".format(idx, dataset, patient_id))
    return test

def get_data(mode, model_input_size, task, MULTI_PREDICTION=False):
    """
    ******* LOCALIZATION *********
    to call:
        model_input_size = (80, 96, 96)
        task = "LOCALIZATION"
        data_train_tune = get_data("train_tune", model_input_size, task)

    intended behaviour:
        train image shape :: (L, 84, 108, 108)
        train label shape :: (L, 84, 108, 108)
        tune image shape :: (L, 1, 80, 96, 96)
        tune label shape :: (L, 1, 80, 96, 96)
        test cases :: L
        test image shape :: (1, 80, 96, 96)
        or
        test image shape :: (84, 108, 108) if MULTI_PREDICTION=True

    ******* SEGMENTATION *********
    to call:
        model_input_size = (64, 160, 160)
        task = "SEGMENTATION"
        data_test = get_data("test", model_input_size, task)

    intended behaviour:
        train image shape :: (L, 76, 196, 196)
        train label shape :: (L, 76, 196, 196)
        tune image shape :: (L, 1, 64, 160, 160)
        tune label shape :: (L, 1, 64, 160, 160)
        test cases :: L
        test image shape :: (1, 64, 160, 160)
        or
        test image shape :: (76, 196, 196) if MULTI_PREDICTION=True

    ******* ACCESS *********

    for mode == "train_tune", objects are returned:
        data_train_tune["train"]["images"][patient, axial_slice, :, :]
        data_train_tune["train"]["labels"][patient, axial_slice, :, :]
        data_train_tune["tune"]["images"][patient, 0, axial_slice, :, :]
        data_train_tune["tune"]["labels"][patient, 0, axial_slice, :, :]
        data_test[patient]["image"][0, axial_slice, :, :]
        data_test[patient]["label"][0, axial_slice, :, :]

    for mode == "test", objects are returned:
        data[i]["patient_id"]
        data[i]["dataset"]
        data[i]["image_sitk_obj"] returns sitk object of image (original)
        data[i]["image"] returns array formatted (crop & reshape)
        or not cropped & reshaped if MULTI_PREDICTION==True
        data[i]["label_sitk_obj"] returns sitk object of label (original)
    """
    if mode=="train_tune":
        data = {
            "train": generate_train_tune_data(data_split, "train", model_input_size, task),
            "tune": generate_train_tune_data(data_split, "tune", model_input_size, task)
        }
        print_shape(data["train"], "train")
        print_shape(data["tune"], "tune")
    elif mode=="test":
        data = generate_test_data(data_split, model_input_size, task, MULTI_PREDICTION)
        print ("test cases :: {}\ntest image shape :: {}".format(len(data), data[0]["image"].shape))
    return data


# data = get_data("train_tune", (80, 96, 96), "LOCALIZATION")
# data = get_data("test", (80, 96, 96), "LOCALIZATION")
# data = get_data("train_tune", (64, 160, 160), "SEGMENTATION")
# data = get_data("test", (64, 160, 160), "SEGMENTATION")
