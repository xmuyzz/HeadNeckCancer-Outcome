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
        print ('patient_id:{} error:{}'.format(patient_id, e))


def main():

    proj_dir = '/mnt/kannlab_rfa/Ben/Maastro_data/radcure/manifest/RADCURE'
    img_dir = '/mnt/kannlab_rfa/Ben/Maastro_data/nrrd'
    save_dir = '/mnt/kannlab_rfa/Ben/Maastro_data/seg'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pat_ids = []
    count = 0
    for folder in os.listdir(proj_dir):
        pat_id = str(folder)
        #if pat_id not in IDs:
        count += 1
        print('patient ID:', count, pat_id)
        if not folder.startswith('.'):
            seg_dir = proj_dir + '/' + folder + '/*/*'
            for seg_dir in glob.glob(seg_dir):
                folder1 = seg_dir.split('/')[-1]
                x = folder1.split('-')[0]
                if x == '1.000000':
                    print('seg folder:', folder1)
                    print('seg_dir:', seg_dir)
                    img_path = img_dir + '/' + pat_id + '.nrrd'
                    try:
                        rtstruct_to_nrrd(
                            patient_id=pat_id, 
                            path_to_rtstruct=seg_dir, 
                            path_to_image=img_path, 
                            output_dir=save_dir)
                    except Exception as e:
                        print(pat_id, e)
                        pat_ids.append(pat_id)
    print('problematic dcm data:', pat_ids)


if __name__ == '__main__':

    main()
