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

    cmd = ['plastimatch', 
           'convert', 
           '--input', path_to_rtstruct, 
           '--output-prefix', output_folder, 
           '--prefix-format', 'nrrd', 
           '--fixed', path_to_image]
    
    subprocess.call(cmd)


def main():

    proj_dir = '/mnt/kannlab_rfa/Ben/Radcure_data'
    img_dir = proj_dir + '/img'
    seg_dir = proj_dir + '/seg'

    count = 0
    bad_data = []
    for data_dir in glob.glob(proj_dir + '/radcure/manifest/RADCURE/*'):
        count += 1
        #print(count, data_dir)
        id = data_dir.split('/')[-1]
        print(count, id)
        for rt_dir in glob.glob(data_dir + '/*/*'):
            key = rt_dir.split('/')[-1].split('-')[0]
            if key == '1.000000':
                #print(key, rt_dir)
                img_path = img_dir + '/' + id + '.nrrd'
                try:
                    rtstruct_to_nrrd(
                        patient_id=id, 
                        path_to_rtstruct=rt_dir, 
                        path_to_image=img_path, 
                        output_dir=seg_dir)
                except Exception as e:
                    print(id, e)
                    bad_data.append(id)
    print('successfully convert rtstruct data in to GTV files')
    print('bad data:', bad_data)
                

if __name__ == '__main__':

    main()
