import nibabel as nib
import numpy as np
import os
import cv2
from tqdm import tqdm
import metrics



def get_scores_onelabel(pred_label_dir, gt_label_dir):
    
    print("\nConverting all predictions to one label...")
    all_scores = {
        'dice': [],
        'precision': [],
        'recall': [],
        'jaccard': [],
        'segmentation': []}
    for ids, scan in enumerate(tqdm(os.listdir(pred_label_dir))):        
        if ".nii.gz" not in scan:
            continue
        print("Looking at scan:", scan)
        scan_pred = nib.load(os.path.join(pred_label_dir, scan)).get_fdata()
        gt = nib.load(os.path.join(gt_label_dir, scan)).get_fdata()
        pred = np.where(scan_pred != 0.0, 1, 0)
        gt_seg = np.where(gt != 0.0, 1, 0)
        assert pred.shape == gt_seg.shape
        # get metrics
        dice_score = metrics.dice(pred, gt_seg)
        precision_score = metrics.precision(pred, gt_seg)
        recall_score = metrics.recall(pred, gt_seg)
        jaccard_score = metrics.jaccard(pred, gt_seg)
        segm_score = metrics.segmentation_score(pred, gt_seg, [1, 1, 1])
        # print(dice_score,precision_score,recall_score,jaccard_score,segm_score)
        # if np.count_nonzero(gt_seg) > 100:  
        # append scores
        all_scores['dice'].append(dice_score)
        all_scores['precision'].append(precision_score)
        all_scores['recall'].append(recall_score)
        all_scores['jaccard'].append(jaccard_score)
        all_scores['segmentation'].append(segm_score)
    # print(len([x for x in all_scores['dice'] if np.isnan(x) == False]))
    dice_mean = np.mean([x for x in all_scores['dice'] if np.isnan(x) == False])
    precision_mean = np.mean([x for x in all_scores['precision'] if np.isnan(x) == False])
    recall_mean = np.mean([x for x in all_scores['recall'] if np.isnan(x) == False])
    jaccard_mean = np.mean([x for x in all_scores['jaccard'] if np.isnan(x) == False])
    seg_score_mean = np.mean([x for x in all_scores['segmentation'] if np.isnan(x) == False])
    print("Mean DICE for all classes:", round(dice_mean, 3))
    print("Mean Precision for all classes:", round(precision_mean, 3))
    print("Mean Recall for all classes:", round(recall_mean, 3))
    print("Mean Jaccard for all classes:", round(jaccard_mean, 3))
    print("Mean Segmentation Score for all classes:", round(seg_score_mean, 3))


def get_scores(pred_label_dir, gt_label_dir):
    
    print("\nUsing all available labels...")
    all_scores = {
        'dice':[],
        'precision':[],
        'recall':[],
        'jaccard':[],
        'segmentation':[]}
    indiv_scores = {
        'dice':{0:[],1:[],2:[],3:[],},
        'precision':{0:[],1:[],2:[],3:[],},
        'recall':{0:[],1:[],2:[],3:[],},
        'jaccard':{0:[],1:[],2:[],3:[],},
        'segmentation':{0:[],1:[],2:[],3:[],}}
    # indiv_dices = {0:[],1:[],2:[],3:[],} 
    for ids, scan in enumerate(tqdm(os.listdir(pred_label_dir))):    
        if ".nii.gz" not in scan:
            continue
        print("Looking at scan:",scan)
        scan_pred = nib.load(os.path.join(pred_label_dir, scan)).get_fdata()
        gt = nib.load(os.path.join(gt_label_dir, scan)).get_fdata()
        for i in range(4):
            pred = np.where(scan_pred==i, 1, 0)
            gt_seg = np.where(gt==i, 1, 0)
            dice_score = metrics.dice(pred, gt_seg)
            precision_score = metrics.precision(pred, gt_seg)
            recall_score = metrics.recall(pred, gt_seg)
            jaccard_score = metrics.jaccard(pred, gt_seg)
            segm_score = metrics.segmentation_score(pred, gt_seg, [1, 1, 1])
            # if np.count_nonzero(gt_seg) > 100:
                # print("\tDice for label",i,":",dice(pred,gt_seg))
            indiv_scores['dice'][i].append(dice_score)
            indiv_scores['precision'][i].append(precision_score)
            indiv_scores['recall'][i].append(recall_score)
            indiv_scores['jaccard'][i].append(jaccard_score)
            indiv_scores['segmentation'][i].append(segm_score)
            if i != 0:
                all_scores['dice'].append(dice_score)
                all_scores['precision'].append(precision_score)
                all_scores['recall'].append(recall_score)
                all_scores['jaccard'].append(jaccard_score)
                all_scores['segmentation'].append(segm_score)
        # print(scan_pred.shape,np.where(scan_pred==1.0, True, False).shape)
        # print(gt.shape,gt[gt==1.0].shape)
    dice_mean = np.mean([x for x in all_scores['dice'] if np.isnan(x) == False])
    precision_mean = np.mean([x for x in all_scores['precision'] if np.isnan(x) == False])
    recall_mean = np.mean([x for x in all_scores['recall'] if np.isnan(x) == False])
    jaccard_mean = np.mean([x for x in all_scores['jaccard'] if np.isnan(x) == False])
    seg_score_mean = np.mean([x for x in all_scores['segmentation'] if np.isnan(x) == False])
    print("Mean DICE for all classes:", round(dice_mean, 3))
    # print("Mean DICE for background:",np.mean([x for x in indiv_scores['dice'][0] if np.isnan(x) == False]))
    # print("Mean DICE for label 1:",np.mean([x for x in indiv_scores['dice'][1] if np.isnan(x) == False]))
    # print("Mean DICE for label 2:",np.mean([x for x in indiv_scores['dice'][2] if np.isnan(x) == False]))
    # print("Mean DICE for label 3:",np.mean([x for x in indiv_scores['dice'][3] if np.isnan(x) == False]))
    print("Mean Precision for all classes:", round(precision_mean, 3))
    print("Mean Recall for all classes:", round(recall_mean, 3))
    print("Mean Jaccard for all classes:", round(jaccard_mean, 3))
    print("Mean Segmentation Score for all classes:", round(seg_score_mean, 3))


if __name__ == "__main__":
    
    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnunet'
    gt_label_dir = os.path.join(proj_dir, 'nnUNet_raw_data_base/nnUNet_raw_data/Task501_PN/labelsTs')
    pred_label_dir = os.path.join(proj_dir, 'results_test')
    
    #get_scores(pred_label_dir, gt_label_dir)
    get_scores_onelabel(pred_label_dir, gt_label_dir)




