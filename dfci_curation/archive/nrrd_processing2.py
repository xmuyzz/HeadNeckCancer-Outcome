### Initiate this file after nrrds have been generated for labels and images in run_bk file 
## Benjamin Kann
### Order of operations: combine label nrrds, interpolate image, interpolate combined label, crop roi

import sys
import os
import pydicom
import matplotlib
from matplotlib.path import Path
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.pyplot import close
#sys.path.append('/data-utils')
from dcm_to_nrrd import dcm_to_nrrd
from rtstruct_to_nrrd import rtstruct_to_nrrd
from combine_structures import combine_structures
from interpolate import interpolate
from crop_roi import crop_roi, crop_top, crop_top_image_only
from nrrd_reg import nrrd_reg_rigid
import SimpleITK as sitk
#sys.path.append('/data-utils')
#sys.path.append('/Users/BHKann/git-code/hn-petct-net/data-utils')
#sys.path.append('/home/bhkann/git-repositories/hn-petct-net/data-utils')




def combine_mask():
    """
    COMBINING MASKS 
    """

    database = "MDACC" ## [CHUM, CHUS, HGJ, HMR, MDACC, PMH]
    PATH0 = path_input + "/curated/" + database + "_files/0_image_raw_" + database + "/"
    # path to image folders Replace with CHUS, HGJ, HRM as needed
    PATH1 = path_input + "/curated/" + database + "_files/1_label_raw_" + database + "_named/"
    # path to label folders
    image_type = "CT"
    #data_type = "combined_masks_p" ### '_n or _p or _pn'
    for i in ['_p','_n','_pn']:
        data_type = "combined_masks" + i
        for file in sorted(os.listdir(PATH0)): #LOOP Goes through nrrd raw images etc
            mask_arr = []
            if not file.startswith('.') and '_' + image_type in file:
                patient_id = file.split('_')[1]
                modality_date = file.split('_')[2]
                print("patient ID: ", patient_id, " modality: ", modality_date)
                path_image = os.path.join(PATH0,file)            
                print("image path: ",path_image)
                elif database == 'CHUM' and data_type == 'combined_masks_pn':
                    #try:
                    for folder in os.listdir(PATH1):
                        for struct in os.listdir(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/"): 
                            if struct.startswith("GTVp") or struct.startswith("GTVn"):
                                mask_arr.append(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/" + struct)
                        output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                        print(mask_arr)
                        combined_mask = combine_structures(
                            dataset='DFCI', 
                            patient_id=patient_id, 
                            data_type=data_type, 
                            mask_arr=mask_arr, 
                            path_to_reference_image_nrrd=path_image, 
                            binary=2, 
                            return_type="sitk_object", 
                            output_dir=output_dir
                            )
                    #except: 
                    #    print("combination failed.")    

                elif database == 'CHUM' and data_type == 'combined_masks_p':
                    try:
                        for struct in os.listdir(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/"): 
                            if struct.startswith("GTVp"):
                                mask_arr.append(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/" + struct)
                        path_to_reference_image_nrrd =  path_image
                        binary = 2 
                        return_type = "sitk_object"
                        output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                        print(mask_arr)
                        combined_mask = combine_structures(
                            dataset='DFCI',
                            patient_id=patient_id,
                            data_type=data_type,
                            mask_arr=mask_arr,
                            path_to_reference_image_nrrd=path_image,
                            binary=2,
                            return_type="sitk_object",
                            output_dir=output_dir
                            )
                    except: 
                        print("combination failed.")        

                elif database == 'CHUM' and data_type == 'combined_masks_n':
                    try:
                        for struct in os.listdir(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/"): 
                            if struct.startswith("GTVn"):
                                mask_arr.append(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/" + struct)
                        path_to_reference_image_nrrd =  path_image 
                        binary = 2 
                        return_type = "sitk_object"
                        output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                        print(mask_arr)
                        combined_mask = combine_structures(
                            dataset='DFCI',
                            patient_id=patient_id,
                            data_type=data_type,
                            mask_arr=mask_arr,
                            path_to_reference_image_nrrd=path_image,
                            binary=2,
                            return_type="sitk_object",
                            output_dir=output_dir
                            )
                    except: 
                        print("combination failed.")
            


def interpolate_img():
    """
    image interpolation
    """

    for img_dir in sorted(glob.glob(DFCI_img_dir + '/*nrrd')):
            pat_id = img_dir.split('_')[1]
            print("patient ID: ", pat_id)
            output_dir = path_input + "/curated/" + database + "_files/interpolated"
            if database == "MDACC":
                for folder in os.listdir(PATH1):
                    label_id = folder.split('_')[1]
                    date_id = 'CT-' + '-'.join(folder.split('-')[2:6]) + '-'
                    if label_id == patient_id and modality_date == date_id:
                        interpolated_nrrd = interpolate(
                            dataset=dataset, 
                            patient_id=patient_id, 
                            data_type='ct', 
                            path_to_nrrd=path_image, 
                            interpolation_type='linear', #"linear" for image, nearest neighbor for label
                            spacing=(1, 1, 3), 
                            return_type='numpy_array', 
                            output_dir=output_dir)
                    else:
                        interpolated_nrrd = interpolate(
                            dataset=dataset,
                            patient_id=patient_id,
                            data_type='ct',
                            path_to_nrrd=path_image,
                            interpolation_type='linear', #"linear" for image, nearest neighbor for label
                            spacing=(1, 1, 3),
                            return_type='numpy_array',
                            output_dir=output_dir) 
            try:
                # 6b. interpolate to a common voxel spacing - label pn
                data_type = "label"
                # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
                path_to_nrrd = path_input + "/curated/" + database + "_files/combinedlabels/" + dataset + "_" + patient_id + "_combined_masks_" + "pn" + "_interpolated_raw_raw_xx.nrrd"
                output_dir = path_input + "/curated/" + database + "_files/interpolated"
                interpolated_nrrd = interpolate(
                    dataset=dataset, 
                    patient_id=patient_id, 
                    data_type='ct', 
                    path_to_nrrd=path_to_nrrd, 
                    interpolation_type='nearest_neighbor', 
                    spacing=(1, 1, 3), 
                    return_type='numpy_array', 
                    output_dir=output_dir)
            except:
                print("could not interpolate pn")
            
            try:
                # 6c. interpolate to a common voxel spacing - label p
                data_type = "label_p"
                # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
                path_to_nrrd = path_input + "/curated/" + database + "_files/combinedlabels/" + dataset + "_" + patient_id + "_combined_masks_" + "p" + "_interpolated_raw_raw_xx.nrrd"
                output_dir = path_input + "/curated/" + database + "_files/interpolated"
                interpolated_nrrd = interpolate(
                    dataset=dataset,
                    patient_id=patient_id,
                    data_type='ct',
                    path_to_nrrd=path_to_nrrd,
                    interpolation_type='nearest_neighbor',
                    spacing=(1, 1, 3),
                    return_type='numpy_array',
                    output_dir=output_dir)
            except:
                print("could not interpolate p")
            
            try:
                # 6d. interpolate to a common voxel spacing - label n
                data_type = "label_n"
                # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
                path_to_nrrd = path_input + "/curated/" + database + "_files/combinedlabels/" + dataset + "_" + patient_id + "_combined_masks_" + "n" + "_interpolated_raw_raw_xx.nrrd"
                output_dir = path_input + "/curated/" + database + "_files/interpolated"
                interpolated_nrrd = interpolate(
                    dataset=dataset,
                    patient_id=patient_id,
                    data_type='ct',
                    path_to_nrrd=path_to_nrrd,
                    interpolation_type='nearest_neighbor',
                    spacing=(1, 1, 3),
                    return_type='numpy_array',
                    output_dir=output_dir)
            except:
                print("could not interpolate n")

       

def registration():
    """
    Rigid Registration - followed by top crop
    """
    image_type="CT"
    size_str = 'reg' #''

    for database in ['MDACC']: #MDACC','PMH','CHUS','CHUM']: 
        PATH0 = path_input + "/curated/" + database + "_files/0_image_raw_" + database + "/" # path to image folders Replace with CHUS, HGJ, HRM as needed
        PATH1 = path_input + "/curated/" + database + "_files/1_label_raw_" + database + "_named/" # path to label folders    
        for file in sorted(os.listdir(PATH0)): #LOOP Goes through nrrd raw images etc
            if not file.startswith('.') and '_' + image_type in file:
                patient_id = file.split('_')[1]
                modality = file.split('_')[2]
                print("patient ID: ", patient_id, " modality: ", modality)
                path_image = os.path.join(PATH0,file)            
                print("image path: ",path_image)   
                # 7. crop roi of defined size using a label # TRY STARTING AT SUPERIOR BORDER AND GOING DOWN 25 cm
                #Crop everything to smallest x-y use 'resize' function for this; then do z]
                #dataset = "HNexports"
                #patient_id = 
                if database == 'CHUM' or database == 'CHUS':
                    dataset = 'HNexports'
                else:
                    dataset = database
                path_to_image = path_input + "/curated/" + database + "_files/interpolated/" + dataset + "_" + patient_id + "_ct_interpolated_raw_raw_xx.nrrd"
                if not os.path.exists(path_input + "/curated/" + database + "_files/image" + "_" + size_str):
                    os.makedirs(path_input + "/curated/" + database + "_files/image" + "_" + size_str)
                path_to_image_output = path_input + "/curated/" + database + "_files/image" + "_" + size_str
                try:
                    fixed_image, moving_image, final_transform = nrrd_reg_rigid_ref(
                        database='DFCI', 
                        patient_id=patient_id, 
                        path_to_image=path_to_image, 
                        path_to_image_output=path_to_image_output, 
                        path_input=path_input)
                    print("image registered.")
                except:
                    print("image registration failed")
                
                for iden in ['','_p','_n']:
                    path_to_label = path_input + "/curated/" + database + "_files/interpolated/" + dataset + "_" + patient_id + "_label" + iden + "_interpolated_raw_raw_xx.nrrd"
                    print("path to label: ", path_to_label)
                    #plt.imshow(image_arr[15,:,:], cmap=plt.cm.gray)
                    #plt.show()
                    if not os.path.exists(path_input + "/curated/" + database + "_files/label" + iden + "_" + size_str):
                        os.makedirs(path_input + "/curated/" + database + "_files/label" + iden + "_" + size_str)                
                    path_to_label_output = path_input + "/curated/" + database + "_files/label" + iden + "_" + size_str
                    try:
                        moving_label = sitk.ReadImage(path_to_label, sitk.sitkFloat32)
                        moving_label_resampled = sitk.Resample(
                            moving_label, 
                            fixed_image, 
                            final_transform, 
                            sitk.sitkNearestNeighbor, 
                            0.0, 
                            moving_image.GetPixelID())
                        sitk.WriteImage(
                            moving_label_resampled, 
                            os.path.join(path_to_label_output, patient_id + "_label_registered.nrrd"))
                        print("label registered: ", iden)
                        #transform = sitk.ReadTransform('.tfm')
                    except:
                        print("label registration failed")



def cropping():
    ## with TOP-CROP HPC ### NEED TO RUN FOR image_crop, image_crop_p, and image_crop_n  
    ### WILL ONLY WORK WITH SPACING = 1,1,3
    image_type="CT"
    roi_size = (172,172,76) #x,y,z
    size_str = '172x172x76'

    for database in ['MDACC','PMH','CHUS','CHUM']: 
        PATH0 = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/0_image_raw_" + database + "/" # path to image folders Replace with CHUS, HGJ, HRM as needed
        PATH1 = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/1_label_raw_" + database + "_named/" # path to label folders
        for iden in ['','_p','_n']:
            for file in sorted(os.listdir(PATH0)): #LOOP Goes through nrrd raw images etc
                if not file.startswith('.') and '_' + image_type in file:
                    patient_id = file.split('_')[1]
                    modality = file.split('_')[2]
                    print("patient ID: ", patient_id, " modality: ", modality)
                    path_image = os.path.join(PATH0,file)            
                    print("image path: ",path_image)   
                    # 7. crop roi of defined size using a label # TRY STARTING AT SUPERIOR BORDER AND GOING DOWN 25 cm
                    #Crop everything to smallest x-y use 'resize' function for this; then do z]
                    #dataset = "HNexports"
                    #patient_id = 
                    if database == 'CHUM' or database == 'CHUS':
                        dataset = 'HNexports'
                    else:
                        dataset = database
                    path_to_image = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/image_reg/" + patient_id + "_registered.nrrd"
                    path_to_label = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/label" + iden + "_reg/" + patient_id + "_label_registered.nrrd"
                    #plt.imshow(image_arr[15,:,:], cmap=plt.cm.gray)
                    #plt.show()
                    try:
                        os.makedirs("/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/image_croptop" + "_" + size_str)
                    except:
                        print("directory already exists")
                    path_to_image_roi = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/image_croptop" + "_" + size_str
                    try:
                        os.makedirs("/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/label_croptop" + iden + "_" + size_str)                
                    except:
                        print("directory already exists")
                    path_to_label_roi = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/label_croptop" + iden + "_" + size_str
                    print("path_to_image: ", path_to_image)
                    print("path_to_label: ", path_to_label)
                    print("path_to_image_roi: ", path_to_image_roi)
                    print("path_to_label_roi: ", path_to_label_roi)
                    try:
                        image_obj, label_obj = crop_top(
                            dataset,
                            patient_id,
                            path_to_image,
                            path_to_label,
                            roi_size,
                            "sitk_object",
                            path_to_image_roi,
                            path_to_label_roi)
                    except:
                        print("crop failed!")
                        

if __name__ == '__main__':




