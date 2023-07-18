import subprocess
import os
import glob


def rtstruct_to_nrrd(patient_id, path_to_rtstruct, path_to_image, output_dir):
    """
    Converts a single rtstruct file into a folder containing individual structure
    nrrd files. The folder will be named dataset_patient id.    
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_rtstruct (str): Path to the rtstruct file.
        path_to_image (str): Path to the image (.nrrd) associated with this rtstruct file.
                             This is needed to match the size and dimensions of the image.
        output_dir (str): Path to folder where the folder containing nrrds will be saved.
        prefix (str): If multiple rtstruct files belong to one patient, their contents can be saved in 
                      multiple folders using this prefix. If "", only one folder will be saved.
    Returns:
        None
    Raises:
        Exception if an error occurs.
    """
    output_folder = output_dir + '/' + patient_id
    cmd = ['plastimatch', 'convert', '--input', path_to_rtstruct, '--output-prefix',
           output_folder, '--prefix-format', 'nrrd', '--fixed', path_to_image]
    try:
        subprocess.call(cmd)
    except Exception as e:
        print ('patient_id:{} error:{}'.format(dataset, patient_id, e))


def main():

    #input_dir = '/mnt/kannlab_rfa/Ben/HN_Dicom_Export'
    #output_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/DFCI/new_curation/raw_segmentation2'
    #data_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/DFCI/new_curation/raw_img2'
    #input_dir = '/mnt/kannlab_rfa/Ben/NewerHNScans/OPX'
    #output_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH3/uncombined_seg'
    #data_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH3/raw_img'
    #input_dir = '/mnt/kannlab_rfa/Ben/HN_NonOPC_Dicom_Export'
    #output_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/NonOPC/uncombined_seg'
    #data_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/NonOPC/raw_img'
    input_dir = '/mnt/kannlab_rfa/Ben/NewerHNScans/OPX'
    output_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/OPC3/raw_gtv'
    data_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/OPC3/raw_img'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    for folder in os.listdir(input_dir):
        #print(folder)
        pat_id = str(folder)
        dcm_dir = os.path.join(input_dir, folder)
        for dcm in glob.glob(dcm_dir + '/*dcm'):
            dcm_type = dcm.split('/')[-1].split('.')[0]
            if dcm_type == 'RTSTRUCT':
                #print(dcm)
                rtstruct_dir = dcm
                for ct_dir in sorted(glob.glob(data_dir + '/*nrrd')):
                    print(ct_dir)
                    ID = ct_dir.split('/')[-1].split('.')[0]
                    print(ID)
                    if ID == pat_id:
                        count += 1
                        print(count, pat_id)
                        img_dir = ct_dir
                        rtstruct_to_nrrd(
                            patient_id=pat_id, 
                            path_to_rtstruct=rtstruct_dir, 
                            path_to_image=img_dir, 
                            output_dir=output_dir)

def main2():

    #input_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/HN_Dicom_Export'
    #output_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH2/uncombined_seg'
    #data_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/BWH2/raw_img'
    input_dir = '/mnt/kannlab_rfa/Ben/NewerHNScans/OPX'
    output_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/OPC3/raw_gtv'    
    data_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/OPC3/raw_img'

    #input_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/OPC2/dcm'
    #output_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/OPC2/gtv_seg'
    #data_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/Data/OPC2/raw_img'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pat_ids = []
    bad_data = []
    count = 0
    for root, dirs, files in os.walk(input_dir):
        if not dirs:
            names = root.split('/')[-1].split('_')
            if names[-1] == 'HN':
                dcm_dir = root
                pat_id = names[0] + '_' + names[1]
                count += 1
                print(count, pat_id)
                for dcm in glob.glob(dcm_dir + '/*dcm'):
                    dcm_type = dcm.split('/')[-1].split('.')[0]
                    if dcm_type == 'RTSTRUCT':
                        #print(dcm)
                        rtstruct_dir = dcm
                        for ct_dir in sorted(glob.glob(data_dir + '/*nrrd')):
                            print(ct_dir)
                            ID = ct_dir.split('/')[-1].split('.')[0]
                            print(ID)
                            if ID == pat_id:
                                count += 1
                                print(count, pat_id)
                                img_dir = ct_dir
                                try:
                                    rtstruct_to_nrrd(
                                        patient_id=pat_id, 
                                        path_to_rtstruct=rtstruct_dir, 
                                        path_to_image=img_dir, 
                                        output_dir=output_dir)
                                except Exception as e:
                                    print(e, pat_id)
                                    bad_data.append(pat_id)
    print(bad_data)


if __name__ == '__main__':

    main()




