import os

nnUNet_raw_data_base= '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnUNet/nnUNet_raw_data_base'
nnUNet_preprocessed='/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnUNet/nnUNet_preprocessed'
RESULTS_FOLDER='/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnUNet/results'
os.environ["nnUNet_raw_data_base"] = str(nnUNet_raw_data_base)
os.environ["nnUNet_preprocessed"] = str(nnUNet_preprocessed)
os.environ["RESULTS_FOLDER"] = str(RESULTS_FOLDER)

#nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity
