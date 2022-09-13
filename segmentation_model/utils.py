import numpy as np
import SimpleITK as sitk
import itertools
from metrics import precision, recall, jaccard, dice, segmentation_score, bbox_distance, surface_dice
from calculate_bbox_metrics import calculate_bbox_metrics



def threshold(pred, thresh=0.5):
    pred[pred<thresh] = 0
    pred[pred>=thresh] = 1
    return pred
 

def reduce_arr_dtype(arr, verbose=False):
    """ Change arr.dtype to a more memory-efficient dtype, without changing
    any element in arr. """

    if np.all(arr-np.asarray(arr,'uint8') == 0):
        if arr.dtype != 'uint8':
            if verbose:
                print('Converting '+str(arr.dtype)+' to uint8 np.ndarray')
            arr = np.asarray(arr, dtype='uint8')
    elif np.all(arr-np.asarray(arr,'int8') == 0):
        if arr.dtype != 'int8':
            if verbose:
                print('Converting '+str(arr.dtype)+' to int8 np.ndarray')
            arr = np.asarray(arr, dtype='int8')
    elif np.all(arr-np.asarray(arr,'uint16') == 0):
        if arr.dtype != 'uint16':
            if verbose:
                print('Converting '+str(arr.dtype)+' to uint16 np.ndarray')
            arr = np.asarray(arr, dtype='uint16')
    elif np.all(arr-np.asarray(arr,'int16') == 0):
        if arr.dtype != 'int16':
            if verbose:
                print('Converting '+str(arr.dtype)+' to int16 np.ndarray')
            arr = np.asarray(arr, dtype='int16')

    return arr


def generate_sitk_obj_from_npy_array(image_sitk_obj, pred_arr, resize=True, output_dir=""):

    """
    When resize==True: Used for saving predictions where padding needs to be added to increase the size of the prediction and match that of input to model. This function matches the size of the array in image_sitk_obj with the size of pred_arr, and saves it. This is done equally on all sides as the input to model and model output have different dims to allow for shift data augmentation.

    When resize==False: the image_sitk_obj is only used as a reference for spacing and origin. The numpy array is not resized.

    image_sitk_obj: sitk object of input to model
    pred_arr: returned prediction from model - should be squeezed.
    NOTE: image_arr.shape will always be equal or larger than pred_arr.shape, but never smaller given that
    we are always cropping in data.py
    """
    if resize==True:
        # get array from sitk object
        image_arr = sitk.GetArrayFromImage(image_sitk_obj)
        # change pred_arr.shape to match image_arr.shape
        # getting amount of padding needed on each side
        z_diff = int((image_arr.shape[0] - pred_arr.shape[0]) / 2)
        y_diff = int((image_arr.shape[1] - pred_arr.shape[1]) / 2)
        x_diff = int((image_arr.shape[2] - pred_arr.shape[2]) / 2)
        # pad, defaults to 0
        pred_arr = np.pad(pred_arr, ((z_diff, z_diff), (y_diff, y_diff), (x_diff, x_diff)), 'constant')
        assert image_arr.shape == pred_arr.shape, "oops.. The shape of the returned array does not match your requested shape."

    # save sitk obj
    new_sitk_object = sitk.GetImageFromArray(pred_arr)
    new_sitk_object.SetSpacing(image_sitk_obj.GetSpacing())
    new_sitk_object.SetOrigin(image_sitk_obj.GetOrigin())

    if output_dir != "":
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_dir)
        writer.SetUseCompression(True)
        writer.Execute(new_sitk_object)
    return new_sitk_object



def combine_masks(mask_list):
    if len(mask_list) >= 2:
        for a, b in itertools.combinations(mask_list, 2):
            assert a.shape == b.shape, "masks do not have the same shape"
            assert a.max() == b.max(), "masks do not have the same max value (1)"
            assert a.min() == b.min(), "masks do not have the same min value (0)"

        # we will ignore the fact that 2 masks at the same voxel will overlap and
        # cause that vixel to have a value of 2.
        # The get_bbox function doesnt really care about that - it just evaluates
        # zero vs non-zero
        combined = np.zeros((mask_list[0].shape))
        for mask in mask_list:
            if mask is not None:
                combined = combined + mask
        return combined
    elif len(mask_list) == 1:
        return mask_list[0]
    else:
        print ("No masks provided!")


def get_bbox(mask_data):
    # crop maskData to only the 1's
    # http://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    # maskData order is z,y,x because we already rolled it
    Z = np.any(mask_data, axis=(1, 2))
    Y = np.any(mask_data, axis=(0, 2))
    X = np.any(mask_data, axis=(0, 1))
    #
    Z_min, Z_max = np.where(Z)[0][[0, -1]]
    Y_min, Y_max = np.where(Y)[0][[0, -1]]
    X_min, X_max = np.where(X)[0][[0, -1]]
    # 1 is added to account for the final slice also including the mask
    return Z_min, Z_max, Y_min, Y_max, X_min, X_max, Z_max-Z_min+1, Y_max-Y_min+1, X_max-X_min+1


def append_helper(result, key_list, obj):
    for key in key_list:
        result[key] = obj[key]
    return result


def get_spacing(sitk_obj):
    """
    flip spacing from sitk (x,y,z) to numpy (z,y,x)
    """
    spacing = sitk_obj.GetSpacing()
    return (spacing[2], spacing[1], spacing[0])


def get_arr_from_nrrd(link_to_nrrd, type):
    '''
    Used for images or labels.
    '''
    sitk_obj = sitk.ReadImage(link_to_nrrd)
    spacing = get_spacing(sitk_obj)
    arr = sitk.GetArrayFromImage(sitk_obj)
    if type=="label":
        arr = threshold(arr)
        assert arr.min() == 0, "minimum value is not 0"
        assert arr.max() == 1, "minimum value is not 1"
        assert len(np.unique(arr)) == 2, "arr does not contain 2 unique values"
    return sitk_obj, arr, spacing


def calculate_metrics(patient_id, spacing, label_arr_org, pred_arr_org, hausdorff_percent, 
                      overlap_tolerance, surface_dice_tolerance):
    """
    metric calculation cleanup in test.py.
    """
    result = {}
    result["patient_id"] = patient_id
    result["precision"] = precision(label_arr_org, pred_arr_org)
    result["recall"] = recall(label_arr_org, pred_arr_org)
    result["jaccard"] = jaccard(label_arr_org, pred_arr_org)
    result["dice"] = dice(label_arr_org, pred_arr_org)
    result["segmentation_score"] = segmentation_score(label_arr_org, pred_arr_org, spacing)
    bbox_metrics = calculate_bbox_metrics(label_arr_org, pred_arr_org, spacing)
    result = append_helper(result, ["x_distance", "y_distance", "z_distance", "distance"], bbox_metrics)
    surface_dice_metrics = surface_dice(
        label_arr_org,
        pred_arr_org,
        spacing,
        hausdorff_percent, 
        overlap_tolerance,
        surface_dice_tolerance)
    result = append_helper(
        result, 
        ["average_surface_distance_gt_to_pr", "average_surface_distance_pr_to_gt", 
         "robust_hausdorff", "overlap_fraction_gt_with_pr", "overlap_fraction_pr_with_gt", "surface_dice"], 
        surface_dice_metrics)
    # get bbox center (indices) of prediction for next segmentation step
    for axes in ["X", "Y", "Z"]:
        for location in ["min", "center", "max", "length"]:
            result["prediction_{}_{}".format(axes, location)] = bbox_metrics["prediction_bbox_metrics"][axes][location]

    return result, result["dice"], bbox_metrics


def pad_helper(center, mid, right_lim, axis):
    """
    Helps in adjustting center points and calculating padding amounts on any axis.
    Args:
        center (int): center index from array_to_crop_from
        mid (int): midpoint of axes of shape to be cropped to
        right_lim (int): right limit of axes of shape to be cropped to, after which padding will be needed
        axis (str): string of which axes "X", "Y, or "Z". For debugging.
    Returns:
        center (int): adjusted center
        pad_l (int): amount of padding needed on the left side of axes
        pad_r (int): amount of padding needed on the right side of axes
    """
    pad_l = 0
    pad_r = 0

    if center < mid:
        pad_l = mid - center
        print ("{} left shift , padding :: {}, center :: {}, mid :: {}".format(axis, pad_l, center, mid))
        center = mid

    # if we are left padding, update the right_lim
    right_lim = right_lim + pad_l

    if center > right_lim:
        pad_r = center - right_lim
        print ("{} right shift , padding :: {}, center :: {}, right_lim :: {}".format(axis, pad_r, center, right_lim))
        # do not change center here

    return  center, pad_l, pad_r


def crop_and_pad(array_to_crop_from, shape_to_crop_to, center, pad_value):
    """
    Will crop a given size around the center, and pad if needed.
    Args:
        array_to_crop_from: array to crop form.
        shape_to_crop_to (list) shape to save cropped image  (z, y, x)
        center (list) indices of center (z, y, x)
        pad_value
    Returns:
        cropped array
    """
    # constant value halves of requested cropped output.
    Z_mid = shape_to_crop_to[0]//2
    Y_mid = shape_to_crop_to[1]//2
    X_mid = shape_to_crop_to[2]//2

    # right-side limits based on shape of input
    Z_right_lim = array_to_crop_from.shape[0]-Z_mid
    Y_right_lim = array_to_crop_from.shape[1]-Y_mid
    X_right_lim = array_to_crop_from.shape[2]-X_mid


    # calculate new center and shifts
    Z, z_pad_l, z_pad_r = pad_helper(center[0], Z_mid, Z_right_lim, "Z")
    Y, y_pad_l, y_pad_r = pad_helper(center[1], Y_mid, Y_right_lim, "Y")
    X, x_pad_l, x_pad_r = pad_helper(center[2], X_mid, X_right_lim, "X")

    # pad
    array_to_crop_from_padded = np.pad(array_to_crop_from,
    ((z_pad_l, z_pad_r),
    (y_pad_l, y_pad_r),
    (x_pad_l, x_pad_r)), 'constant', constant_values=pad_value)

    # get limits
    Z_start, Z_end = Z-Z_mid, Z+Z_mid
    Y_start, Y_end = Y-Y_mid, Y+Y_mid
    X_start, X_end = X-X_mid, X+X_mid

    return array_to_crop_from_padded[Z_start:Z_end, Y_start:Y_end, X_start:X_end]


# USING THIS
def save_candidate_roi(bbox_metrics, spacing, path_to_image_to_crop, crop_shape, output_path_image):

    ## PROBLEM!! using prediction center, but cropping gt label?!
    # CROP image and label based on previous pred center, can then compare label and pred
    """
        bbox_metrics (dict) dict of metrics calculated between gt and pred
        spacing (list) "old" spacing of arrays used in localization step i.e.(6,3,3) z,y,x
        path_to_image_to_crop (str) path to image to be cropped
        crop_shape (list) shape to save cropped image and nrrd (larger than input to segmentation model) (z,y,x)
        output_path_image (str) path to save image nrrd
    """

    # get image and label to be cropped + new spacing
    image_obj, image_arr, image_spacing = get_arr_from_nrrd(path_to_image_to_crop, "image")

    # get centers from predictions in localization step
    Z = int(bbox_metrics["prediction_bbox_metrics"]["Z"]["center"])
    Y = int(bbox_metrics["prediction_bbox_metrics"]["Y"]["center"])
    X = int(bbox_metrics["prediction_bbox_metrics"]["X"]["center"])

    # get new centers based on old and new spacing
    old_spacing = spacing
    new_spacing = [int(x) for x in image_spacing] # or label_spacing
    Z = int((Z * old_spacing[0]) // new_spacing[0])
    Y = int((Y * old_spacing[1]) // new_spacing[1])
    X = int((X * old_spacing[2]) // new_spacing[2])

    # crop and pad
    image_arr_crop = crop_and_pad(image_arr, crop_shape, (Z, Y, X), -1024)

    # HACK to save center to file name
    output_path_image = "{}_{}_{}_{}.{}".format(output_path_image.split("<>")[0], Z, Y, X, "nrrd")

    # save
    image_crop_sitk = generate_sitk_obj_from_npy_array(image_obj, image_arr_crop, resize=False, output_dir=output_path_image)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def multi_prediction(image, original_model, image_shape):
    """
    image: From data that has not been cropped or reshaped.
    original_model: model to run inference
    image_shape: tuple of shape to be fed into model (z, y, x)
    """
    org = image
    combined_label = np.zeros(org.shape)
    #
    z_diff = org.shape[0] - image_shape[0]
    y_diff = org.shape[1] - image_shape[1]
    x_diff = org.shape[2] - image_shape[2]

    offsets = [
        # counter clockwise on top
        {"z":0,
         "y":0,
         "x":0
        },
        {"z":0,
         "y":y_diff,
         "x":0
        },
        {"z":0,
         "y":y_diff,
         "x":x_diff
        },
        {"z":0,
         "y":0,
         "x":x_diff
        },
        # central
        {"z":z_diff//2,
         "y":y_diff//2,
         "x":x_diff//2
        },
        # counter clockwise on bottom
        {"z":z_diff,
         "y":0,
         "x":0
        },
        {"z":z_diff,
         "y":y_diff,
         "x":0
        },
        {"z":z_diff,
         "y":y_diff,
         "x":x_diff
        },
        {"z":z_diff,
         "y":0,
         "x":x_diff
        }
    ]

    for offset in offsets:
        Z = offset["z"]
        Y = offset["y"]
        X = offset["x"]
        crop_to_predict_on = org[Z:(Z+image_shape[0]),
                                 Y:(Y+image_shape[1]),
                                 X:(X+image_shape[2])]
        # run prediction
        pred = original_model.predict( crop_to_predict_on.reshape(1,1,*crop_to_predict_on.shape))

        pred = np.squeeze(pred)

        # should happen elsewhere for efficiency
        assert pred.shape == image_shape

        combined_label[Z:(Z+image_shape[0]),
                       Y:(Y+image_shape[1]),
                       X:(X+image_shape[2])] += pred

    # # divide and return
    # # thresholding happens in test.py
    # combined_label /= 9

    return combined_label









