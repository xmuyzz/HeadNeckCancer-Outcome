import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure
import sys
import numpy as np
import os
import utils



# mask_cmap = matplotlib.cm.get_cmap('Set1')
mask_cmap = [(57/255, 255/255, 20/255, 1.0), (1, 1, 1, 1.0)]

def getClosestSlice(centroidSciPy):
    '''
    com
    0, 1, 2
    axial = 0, Saggital = 2, coronal= 1
    '''
    return int( centroidSciPy[0] ),int( centroidSciPy[1] ),int( centroidSciPy[2] )


def show_axis(data, mask, axis, index, slice, bbox, show_bbox, mask_count):
    """
    Plots a single image.
    Data and mask should be numpy arrays.
    """
    axis.set_aspect('equal')
    axis.axis('off')
    vmax = 3071
    vmin = -1024
    line_width = 1
    Z_unrolled = data.shape[0]
    colors = ['blue', 'green']
    # flip the array if coronal or sagittal is passed
    # each view will need its own bbox coordinates
    if slice == "axial":
        rect = [bbox[4], bbox[2], bbox[8], bbox[7]]
    elif slice == "coronal":
        data = np.rollaxis(data, 1)
        data = np.flip(data, 1)
        mask = np.rollaxis(mask, 1)
        mask = np.flip(mask, 1)
        rect = [bbox[4], Z_unrolled - bbox[0] - bbox[6] , bbox[8], bbox[6]]
    elif slice == "sagittal":
        data = np.rollaxis(data, 2)
        data = np.flip(data, 1)
        mask = np.rollaxis(mask, 2)
        mask = np.flip(mask, 1)
        rect = [bbox[2], Z_unrolled - bbox[0] - bbox[6], bbox[7], bbox[6]]
    axis.title.set_text("{} of {}".format(str(index), data.shape[0]))

    # only if this is the first mask
    # plot both the image and the bbox of that first mask
    # subtract 0.5 to account for contours at 0.5 intervals
    # coming out of measure.find_contours
    axis.imshow(data[index], cmap="gray", vmin=vmin, vmax=vmax, interpolation='none')
    if show_bbox:
        rect = patches.Rectangle(
            (rect[0] - 0.5, rect[1] - 0.5), rect[2], rect[3], linewidth=line_width, 
             edgecolor=mask_cmap[mask_count], 
             #edgecolor=colors[mask_count],
             facecolor='none')
        axis.add_patch(rect)
    # plot contours, use 0.5 to exactly capture the midpoint between 0 pixels
    # and 1 pixels.. duh..
    contours =  measure.find_contours(mask[index], 0.5)
    for n, contour in enumerate(contours):
        axis.plot(contour[:, 1],
                  contour[:, 0], 
                  linewidth=line_width*2, 
                  #linewidth=2,
                  #color=colors[mask_count])
                  color=mask_cmap[mask_count])

def plot_figure(dataset, patient_id, data_arr, mask_arr_list, mask_list_names, com_gt, com_pred, 
                bbox_list, show_bbox, output_dir, distance, dice):
    """
    makes 3x5 plots: Axial, sagittal, coronal at the following intervals of the
    mask :begining, 1/4, COM, 3/4 and end.
    """
    bbox = bbox_list[0]
    # figure
    fig, ax = plt.subplots(3, 5)
    fig.set_size_inches(26, 14)
    gs1 = gridspec.GridSpec(3, 5)
    gs1.update(wspace=0.025, hspace=0.15)
    #title = "{}_{}\ndistance: {}mm\ngt bbox center: {} pred bbox center: {}\ndice: {}".format(
    #    dataset, patient_id, round(distance, 3), com_gt, com_pred, round(dice, 3))
    title = "{}: {}\n dice: {}, distance: {}mm, gt bbox center: {} pred bbox center: {}".format(
        dataset, patient_id, round(dice, 3), round(distance, 3), com_gt, com_pred) 
    name = "{}_{}".format(dataset, patient_id)
    fig.suptitle(title, fontsize=20, weight='bold')
    axial_idx = [bbox[0], bbox[0] + int((com_gt[0] - bbox[0])//2.), com_gt[0], 
                 bbox[1] - int((bbox[1] - com_gt[0])//2.), bbox[1]]
    coronal_idx = [bbox[2], bbox[2] + int((com_gt[1] - bbox[2])//2.), 
                   com_gt[1], bbox[3] - int((bbox[3] - com_gt[1])//2.), bbox[3]]
    sagittal_idx = [bbox[4], bbox[4] + int((com_gt[2] - bbox[4])//2.), 
        com_gt[2], bbox[5] - int((bbox[5] - com_gt[2])//2.), bbox[5]]
    for i, (mask_arr, bbox) in enumerate(zip(mask_arr_list, bbox_list)):
        if mask_arr is not None:
            for j in range(5):
                show_axis(data_arr, mask_arr, plt.subplot(gs1[j]), axial_idx[j], "axial", bbox, show_bbox, i)
                show_axis(data_arr, mask_arr, plt.subplot(gs1[5+j]), coronal_idx[j], "coronal", bbox, show_bbox, i)
                show_axis(data_arr, mask_arr, plt.subplot(gs1[10+j]), sagittal_idx[j], "sagittal", bbox, show_bbox, i)
    plot_legend(plt.subplot(gs1[0]), mask_list_names)
    fig.savefig(os.path.join(output_dir, name + '.png'), dpi=200)
    plt.cla()
    plt.clf()
    plt.close("all")


def plot_legend(ax, mask_list_names):
    """
    Plots a legend given mask list namesself.
    https://stackoverflow.com/questions/14531346/how-to-add-a-text-into-a-rectangle
    """
    rectangles = {}
    colors = ['blue', 'red']
    for i, mask_name in enumerate(mask_list_names):
        if mask_name is not None:
            rectangles[mask_name] = patches.Rectangle(
                (-70, i*50), 50, 30, clip_on=False, facecolor=mask_cmap[i], edgecolor="k", linewidth=3)
    for r in rectangles:
        ax.add_patch(rectangles[r])
        rx, ry = rectangles[r].get_xy()
        cx = rx + rectangles[r].get_width()/2.0
        cy = ry + rectangles[r].get_height()/2.0
        ax.annotate(r, (cx, cy), color='k', weight='bold', fontsize=20, ha='center', 
            va='center', annotation_clip=False)


def plot_images(dataset, patient_id, data_arr, gt_arr, pred_arr, output_dir, bbox_flag, bbox_metrics, dice):
    """
    Plots 15 different views of a given patient imaging data.
    # bbox metrics from distance calculation
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_data (str): Path to nrrd file containing the image.
        path_to_mask_list (list) List of strings paths to nrrd files containing contours.
        Files must be in named following the naming convention.
        At least one mask(contour) should be provided as this is used to set the viewing bounds ofthe image. 
        If multiple masks are provided, they are added up and the resultis used to set the bounds.
        Make sure to pass the masks in the same order(for each patient) so that the contour colors do not flip on you.
        output_dir (str): Path to folder where the png will be saved
        bbox_flag (bool): Boolean whether to show bounding box or not. 
        If True, it will be set based on the viewing bounds.
    Returns:
        None
    Raises:
        Exception if an error occurs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        gt_bbox_metrics = bbox_metrics["ground_truth_bbox_metrics"]
        pred_bbox_metrics = bbox_metrics["prediction_bbox_metrics"]
        pred_arr = utils.threshold(pred_arr)
        mask_arr_list = [gt_arr, pred_arr]
        mask_list_names = ["gt", "pd"]
        # bbox and centroid will be calculated based on gt_arr only
        # combined = utils.combine_masks(mask_arr_list)
        gt_bbox = utils.get_bbox(gt_arr)
        pred_bbox = utils.get_bbox(pred_arr)
        # decided to use geometric center as opposed to center of mass to show the actual 
        # center of the bounding box - although still called com, it is the geometric center
        # com = ndimage.measurements.center_of_mass(combined)
        # print(com)
        # com = getClosestSlice(com)
        # print(com)
        com_gt = getClosestSlice((gt_bbox_metrics["Z"]["center"],
        gt_bbox_metrics["Y"]["center"],gt_bbox_metrics["X"]["center"]))
        com_pred = getClosestSlice((pred_bbox_metrics["Z"]["center"],
        pred_bbox_metrics["Y"]["center"],pred_bbox_metrics["X"]["center"]))
        assert gt_bbox_metrics["Z"]["min"] == gt_bbox[0], "bbox calc incorrect"
        assert gt_bbox_metrics["Z"]["max"] == gt_bbox[1], "bbox calc incorrect"
        assert gt_bbox_metrics["Y"]["min"] == gt_bbox[2], "bbox calc incorrect"
        assert gt_bbox_metrics["Y"]["max"] == gt_bbox[3], "bbox calc incorrect"
        assert gt_bbox_metrics["X"]["min"] == gt_bbox[4], "bbox calc incorrect"
        assert gt_bbox_metrics["X"]["max"] == gt_bbox[5], "bbox calc incorrect"
        # plot
        plot_figure(dataset, patient_id, data_arr, mask_arr_list, mask_list_names, com_gt, 
                com_pred, [gt_bbox, pred_bbox], bbox_flag, output_dir, bbox_metrics["distance"], dice)

        print ("{}_{} saved".format(dataset, patient_id))
    except Exception as e:
        print ("Error in {}_{}, {}".format(dataset, patient_id, e))





