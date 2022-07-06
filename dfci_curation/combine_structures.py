import SimpleITK as sitk
import os
import itertools
import numpy as np



def combine_structures(dataset, patient_id, data_type, mask_arr, path_to_reference_image_nrrd, 
                       binary, return_type, output_dir=""):
    
    """
    Combines single mask nrrd files into a single nrrd file. It will first combine all masks as integers (1,2,3..) 
    with 0 being background. This ensures that there are no overlapping masks (e.g. same voxel is both lung and tumor). 
    If binary==true, it will split them into binary arrays (this is likely what you model will need).
    
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        data_type (str): Type of data (e.g., ct, pet, mri, lung(mask), heart(mask)..)
        mask_arr (list): List of paths to mask nrrd files. The order dictates what has priority over the other.
        path_to_reference_image_nrrd (str): Path to nrrd image. This will be used as reference to construct the combined mask nrrd file.
        binary (int):
        If 0, the returned array is categorial (0=background, 1=..) - single channel.
        if 1, the returned array is binary but in multiple channels i.e. each mask in one channel.
        If 2, the returned array is binary but in a single channel i.e. masks are combined.
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_dir (str): Optional. If provided, nrrd file will be saved there. If not provided, file will not be saved.
    Returns:
        None
    Raises:
        Exception if an error occurs.
    """
    
    arrays = []
    for mask in mask_arr:
        img = sitk.ReadImage(mask)
        arr = sitk.GetArrayFromImage(img)
        arrays.append(arr)

    # ensure that all images have the same shape
    for a, b in itertools.combinations(arrays, 2):
        assert a.shape == b.shape, "masks do not have the same shape"
        assert a.max() == b.max(), "masks do not have the same max value (1)"
        assert a.min() == b.min(), "masks do not have the same min value (0)"
    combined = np.zeros(arrays[0].shape)
    # prioritizes maks based on order in list
    for label in reversed(range(1,len(mask_arr) + 1)):
        print('Label value: {} for {}.'.format(label, mask_arr[label - 1]))
        combined[arrays[label-1] == 1] = label
    # reduce datatype to save space, also for training
    combined = reduce_arr_dtype(combined)

    if binary==1:
        binary_arrays = []
        for i in range(1, len(mask_arr) + 1):
            arr = np.zeros(combined.shape)
            arr[combined==i] = 1
            binary_arrays.append(arr)
        combined = np.stack(binary_arrays, axis=3)
        combined = util.reduce_arr_dtype(combined)

    if binary==2:
        combined[combined != 0] = 1

    print ("Shape of output :: " , combined.shape)
    print ("Unique values in output :: " , np.unique(combined))

    # load reference patient image
    reference = sitk.ReadImage(path_to_reference_image_nrrd)
    new_sitk_object = sitk.GetImageFromArray(combined)
    new_sitk_object.SetSpacing(reference.GetSpacing())
    new_sitk_object.SetOrigin(reference.GetOrigin())

    # will not throw errors if channels exist
    if not (new_sitk_object.GetSize() == reference.GetSize()):
        print('WARNING! Size mismatch: {} vs {}'.format(new_sitk_object.GetSize(), reference.GetSize()))

    # write new nrrd
    if output_dir != "":
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(output_dir, "{}_{}_{}.nrrd".format(dataset, patient_id, data_type)))
        writer.SetUseCompression(True)
        writer.Execute(new_sitk_object)

    if return_type == "sitk_object":
        return new_sitk_object
    elif return_type == "numpy_array":
        return sitk.GetArrayFromImage(new_sitk_object)



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




