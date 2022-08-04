from nnunet.dataset_conversion.utils import generate_dataset_json



if __name__ == '__main__':

    base = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data'
    task_name = 'Task501_PN'
    target_base = os.path.join(nnUNet_raw_data, task_name)
    target_imagesTr = os.path.join(target_base, 'imagesTr')
    target_imagesTs = os.path.join(target_base, 'imagesTs')
    target_labelsTr = os.path.join(target_base, 'labelsTr')
    target_labelsTs = os.path.join(target_base, 'labelsTs')
    # generating a dataset.json
    generate_dataset_json(
        output_file=os.path.join(target_base, 'dataset.json'), 
        imagesTr_dir=target_imagesTr, 
        imagesTs_dir=target_imagesTs, 
        modalities=('ct'),
        labels={0: 'background', 1: 'tumor'}, 
        dataset_name=task_name, 
        license='hands off!')

    """
    once this is completed, you can use the dataset like any other nnU-Net dataset. Note that since this is a 2D
    dataset there is no need to run preprocessing for 3D U-Nets. You should therefore run the 
    `nnUNet_plan_and_preprocess` command like this:
    
    > nnUNet_plan_and_preprocess -t 120 -pl3d None
    
    once that is completed, you can run the trainings as follows:
    > nnUNet_train 2d nnUNetTrainerV2 120 FOLD
    
    (where fold is again 0, 1, 2, 3 and 4 - 5-fold cross validation)
    
    there is no need to run nnUNet_find_best_configuration because there is only one model to choose from.
    Note that without running nnUNet_find_best_configuration, nnU-Net will not have determined a postprocessing
    for the whole cross-validation. Spoiler: it will determine not to run postprocessing anyways. If you are using
    a different 2D dataset, you can make nnU-Net determine the postprocessing by using the
    `nnUNet_determine_postprocessing` command
    """



