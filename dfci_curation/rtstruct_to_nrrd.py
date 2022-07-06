import subprocess
import os
import glob


def rtstruct_to_nrrd(dataset, patient_id, path_to_rtstruct, path_to_image, output_dir, prefix = ""):
    
    """
    Converts a single rtstruct file into a folder containing individual structure
    nrrd files. The folder will be named dataset_patient id.
    
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_rtstruct (str): Path to the rtstruct file.
        path_to_image (str): Path to the image (.nrrd) associated with this rtstruct file.This is needed to match the size and dimensions of the image.
        output_dir (str): Path to folder where the folder containing nrrds will be saved.
        prefix (str): If multiple rtstruct files belong to one patient, their contents can be saved in multiple folders using this prefix. If "", only one folder will be saved.
    Returns:
        None
    Raises:
        Exception if an error occurs.
    """

    if prefix == "":
        output_folder = os.path.join(output_dir, "{}_{}".format(dataset, patient_id))
    else:
        output_folder = os.path.join(output_dir, "{}_{}_{}".format(dataset, patient_id, prefix))
    cmd = ["plastimatch", "convert", "--input", path_to_rtstruct, "--output-prefix",
           output_folder, "--prefix-format", "nrrd", "--fixed", path_to_image]
    try:
        subprocess.call(cmd)
    except Exception as e:
        print ("dataset:{} patient_id:{} error:{}".format(dataset, patient_id, e))


if __name__ == '__main__':

    input_dir = '/mnt/aertslab/USERS/Christian/For_Ben'
    output_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/dfci_seg'
    data_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/dfci_data'

    count = 0
    for folder in os.listdir(input_dir):
        #print(folder)
        pat_id = str(folder)
        dcm_dir = os.path.join(input_dir, folder)
        for dcm in glob.glob(dcm_dir + '/*dcm'):
            dcm_type = dcm.split('/')[-1].split('.')[0]
            if dcm_type == 'RTSTRUCT':
                print(dcm)
                rtstruct_dir = dcm
                for ct_dir in sorted(glob.glob(data_dir + '/*nrrd')):
                    ID = ct_dir.split('/')[-1].split('_')[1]
                    if ID == pat_id:
                        count += 1
                        print(count)
                        img_dir = ct_dir
                        rtstruct_to_nrrd(
                            dataset='dfci', 
                            patient_id=pat_id, 
                            path_to_rtstruct=rtstruct_dir, 
                            path_to_image=img_dir, 
                            output_dir=output_dir, 
                            prefix="")




