import os
import sys
from scipy import ndimage
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pprint import pprint
#
import tensorflow as tf

from keras.models import load_model
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
#
sys.path.append('/home/bhkann/git-repositories/hn-petct-net/3d-unet-petct/files')
from utils import generate_sitk_obj_from_npy_array, threshold, get_spacing, calculate_metrics, save_candidate_roi, multi_prediction, get_lr_metric
from plot_images import plot_images
from losses import precision_loss, dice_loss, tversky_loss, focal_tversky_loss, bce_loss, bce_dice_loss, wce_dice_loss

#from data import get_data
def path_helper(folder, file_tail):
    return "{}/{}/{}/{}_{}_{}.nrrd".format(MASTER_FOLDER, dataset, folder, dataset, patient_id, file_tail)

#tf.keras.backend.clear_session()

# TODO: deal with empty predictions, for localization
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


RUN = "1" #
NAME = "wce-dice-loss-0.001-160augment-hn-petct-pmhmdcon_gtvn_opc20220324-1623" #+ "_" + META#
#NAME = "wce-dice-loss-0.001-160augment-hn-petct-pmhmdchuschum_gtvn"
SAVE_CSV = True
print("segmentation test run # {}".format(RUN))

 ## SUBSET FOR TRAINING

IMAGE_SHAPE = (64, 160, 160) # z,x,y, z=128 or 96
SAVE_CANDIDATES = False # always false

IMAGE_TYPE = 'ct'
CROP_SHAPE = '_172x172x76' # need to start with underscore
MODEL_TO_USE = "" #_dsc" # "_final" # "" or "_final" or "_dsc"

# metric-specific
HAUSDORFF_PERCENT = 95
OVERLAP_TOLERANCE = 5
SURFACE_DICE_TOLERANCE = 5 # in millimeters,2 used for prior AIM work

# get data # mode E3311/Yale is for generation of segmentations for ENE model (no labeled GTVs)
mode = 'PMHMD'
if mode == 'E3311':
    from data_HPC_111320_ECOG import get_data 
elif mode == 'Yale':
    from data_HPC_111320_Yale import get_data 
else:
    from data_HPC_020621 import get_data # MODIFY WITH UPDATED DATA SCRIPT ## Need to fix the PET one

SAVE_DIR = '/home/bhkann/deeplearning/HN_PETSEG/output/{}_{}/{}_files/'.format(RUN,NAME,mode)


data = get_data("test", IMAGE_SHAPE, CROP_SHAPE, meta=False, MULTI_PREDICTION=False, image_format=IMAGE_TYPE, seg_target='gtvn', SAVE_CSV=True, SAVE_DIR=SAVE_DIR) #SAVE_CSV=False)

# folder should already exist from training run
model_dir = "/home/bhkann/deeplearning/HN_PETSEG/output/{}_{}".format(RUN, NAME)

#save_model_dir = "/home/mnt/aertslab/USERS/Ben/hn-pet-ct/output/" + RUN + "_" + NAME
# initiate vars
results = []
no_results = []

# load model
model = os.path.join(model_dir, "{}{}.h5".format(RUN, MODEL_TO_USE))
#model = '/home/bhkann/deeplearning/output/1multi_wce-dice-loss-0.0005-augment-hn-petct-chus_pet_gen/1multi_final.h5'
original_model = load_model(model, custom_objects={'InstanceNormalization': InstanceNormalization, 'wce_dice_loss': wce_dice_loss, 'lr':get_lr_metric})

## Combining separate GTVP and GTVN models ##
if not os.path.exists(SAVE_DIR + NAME):
    os.makedirs(SAVE_DIR + NAME)
## Run predictions
for patient in data:
    #### VARIABLES
    patient_id = patient["patient_id"]
    dataset = patient["dataset"]
    # formatted (cropped & reshaped) if MULTI_PREDICTION = False
    # not cropped or reshaped if MULTI_PREDICTION = True
    image = patient["image"]
    # original size
    image_sitk_obj = patient["image_sitk_obj"]
    #image_sitk_obj = sitk.ConstantPadImageFilter(image_sitk_obj,(172,172,76)
    if mode == 'PMHMD':
        label_sitk_obj = patient["label_sitk_obj"]
    spacing = get_spacing(image_sitk_obj)
    
    if IMAGE_TYPE == "ctpet":
        image_pet = patient["image_pet"]
        image_pet_sitk_obj = patient["image_pet_sitk_obj"]
        
    #### PREDICT
    if IMAGE_TYPE == "ctpet":
        label_prediction = original_model.predict(np.concatenate((image,image_pet),axis=1).reshape(1,2,*IMAGE_SHAPE),use_multiprocessing=False)
    elif IMAGE_TYPE == "ct":
        label_prediction = original_model.predict(image.reshape(1,*image.shape))
    label_prediction = threshold(np.squeeze(label_prediction)) # 0.5
    # if there are voxels predicted:
    if label_prediction[label_prediction==1].sum() > 0:
        
        # save model output as nrrd
        # this will pad the prediction to match the size of the originals
        # for localization, 80, 96, 96 => 84, 108, 108
        # for segmentation, 64, 160, 160 => 76, 196, 196
        #try:
        pred_sitk_obj = generate_sitk_obj_from_npy_array(
        image_sitk_obj,
        label_prediction,
        True,
        os.path.join(SAVE_DIR + NAME, "{}_{}_segmentation.nrrd".format(dataset, patient_id)))
        print("prediction nrrd saved: ",patient_id)
        #except:
        #    print("original and prediction shapes not matched. skipped nrrd save.")
        # get arrays from data
        image_arr_org = sitk.GetArrayFromImage(image_sitk_obj) # change to whatever you want image to look like, PET not displaying right
        if mode == 'PMHMD':
            label_arr_org = sitk.GetArrayFromImage(label_sitk_obj)
        # get arrays from prediction
        pred_arr_org = sitk.GetArrayFromImage(pred_sitk_obj)
        

        if mode == 'PMHMD':
            # metrics
            result, dice, bbox_metrics = calculate_metrics(patient_id, spacing, label_arr_org, pred_arr_org, HAUSDORFF_PERCENT, OVERLAP_TOLERANCE, SURFACE_DICE_TOLERANCE)
            
            # append
            results.append(result)
            
            # plot 5x3 views
            plot_images(dataset,
                        patient_id,
                        image_arr_org,
                        label_arr_org,
                        pred_arr_org,
                        model_dir, #can change to aertslab drive (save_model_dir) or HPC (model_dir)
                        True,
                        bbox_metrics,
                        dice)
            print ("{} done. dice :: {}".format(patient_id, result["dice"]))
            
            ### Split Nodes/Primaries to individual ###
            #TRY: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/35_Segmentation_Shape_Analysis.html
            
            # extract ROI from image_interpolated_resized
            if SAVE_CANDIDATES:
                # create folder
                dir = "{}/{}/{}".format(MASTER_FOLDER, dataset, IMAGE_INTERPOLATED_ROI_PR_FOLDER)
                if not os.path.exists(dir):
                    os.mkdir(dir)
                    print("directory {} created".format(dir))
                # save candidates
                save_candidate_roi(bbox_metrics,
                   spacing,
                   path_helper(IMAGE_INTERPOLATED_RESIZED_FOLDER, "image_interpolated_resized_raw_xx"),
                   CROP_SHAPE,
                   "{}/{}_{}_{}".format(dir, dataset, patient_id, "image_interpolated_roi_raw_pr<>.nrrd"))
        
        else:
            no_results.append(patient_id)
            # temporary for segmentation task
            result = {}
            result["patient_id"] = patient_id
            result["precision"] = 0
            result["recall"] = 0
            result["jaccard"] = 0
            result["dice"] = 0
            result["segmentation_score"] = 0
            result["x_distance"] = 0
            result["y_distance"] = 0
            result["z_distance"] = 0
            result["distance"] = 0
            result["average_surface_distance_gt_to_pr"] = 0
            result["average_surface_distance_pr_to_gt"] = 0
            result["robust_hausdorff"] = 0
            result["overlap_fraction_gt_with_pr"] = 0
            result["overlap_fraction_pr_with_gt"] = 0
            result["surface_dice"] = 0
            for axes in ["X", "Y", "Z"]:
                for location in ["min", "center", "max", "length"]:
                    result["prediction_{}_{}".format(axes, location)] = 0
            results.append(result)


print ("no results :: ", no_results)

# populate df
if SAVE_CSV and mode=='PMHMD':
    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(model_dir, "{}{}_{}.csv".format(RUN,NAME,MODEL_TO_USE))) # Can choose lab drive vs HPC to save
    #np.median(df['dice'])
    print("median Dice MDACC:",np.median(df['dice'][df['patient_id'].str.contains('HNSCC')]))
    #print("median Dice CHUS:",np.median(df['dice'][df['patient_id'].str.contains('CHUS')]))
    #print("median Dice CHUM:",np.median(df['dice'][df['patient_id'].str.contains('CHUM')]))
    print("median Dice PMH:",np.median(df['dice'][df['patient_id'].str.contains('OPC')]))

    print("median Dice overall:",np.median(df['dice']))

    print("median surface Dice:",np.median(df['surface_dice']))


    # merge meta-clinical-data
    df_pred = pd.read_csv(os.path.join(model_dir, "{}{}_{}.csv".format(RUN,NAME,MODEL_TO_USE)))
    df_clinmeta = pd.read_csv('/home/bhkann/git-repositories/hn-petct-net/clinical_meta_data.csv')

    df_clinmeta = df_clinmeta.rename(columns={'patientid': 'patient_id'})
    df_all = pd.merge(df_pred, df_clinmeta, how='left',on='patient_id')
    df_all.to_csv(model_dir + '/preds_clin_meta.csv')

    ## Filter by contrast
    df_contrast=df_all.loc[df_all.contrastbolusagent.notnull()]
    np.median(df_contrast['dice'])
        
    print("median Dice contrast MDACC:",np.median(df_contrast['dice'][df_contrast['patient_id'].str.contains('HNSCC')]))
    print("median Dice contrast PMH:",np.median(df_contrast['dice'][df_contrast['patient_id'].str.contains('OPC')]))

    print("median Dice contrast only:",np.median(df_contrast['dice']), "# contrast patients: ",len(df_contrast))

    print("median surface Dice contrast only:",np.median(df_contrast['surface_dice']))
