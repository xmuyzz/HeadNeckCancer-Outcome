import numpy as np
import keras
from scipy import ndimage
import numpy as np
from elastic_deformation import Elastic
import SimpleITK as sitk
from random import randint


class Generator(keras.utils.Sequence):
    """
    Generates data for Keras. Does the following (not in this order):
    * crop
    * affine transformation (rotate, shear, scale)
    * pepper
    * blur label
    * elastic deformation through the Elastic class
    """
    def __init__(self, images, labels, batch_size, final_shape, blur_label, augment, elastic, shuffle):
        # from init args:
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.final_shape = final_shape
        self.blur_label = blur_label
        self.augment = augment
        self.elastic = elastic
        self.shuffle = shuffle
        # defined here:
        self.rot_ang_rng = 8
        self.shr_ang_rng = 8
        self.red_rot_shr = 0.6 # reduces rot and shear range for specific axes
        self.scale_rng = 0.2 # +1 and -1
        self.blur_multiplier = 2.0
        self.blur_random_range = 0.6 # 60% + or -
        #
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch - some samples at the end are automatically skipped at each epoch.'
        n_batches = int(np.floor(len(self.images) / self.batch_size))
        return n_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_idx_list = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        return self.__data_generation(batch_idx_list)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_idx_list):
        """
        Generates data containing batch_size samples.
        """
        image_list = []
        label_list = []

        for index, (image, label) in enumerate( zip(self.images, self.labels) ):
            # if index is part of this batch
            if index in batch_idx_list:
                # augment
                selector1 = randint(0,6)
                selector2 = randint(0,6)
                selector3 = randint(0,6)
                if self.augment:
                    # apply elastic transformation
                    if self.elastic and np.random.uniform() < 0.5:
                        elastic = Elastic(image, label)
                        image, label = elastic.run()
                    # affine transformations + crop
                    image, label = self._affine_transform_arr(image, label)
                    image, label = self._crop_arr(image, label)
                    # noise
                    image = self._pepper_arr(image)
                    if self.blur_label:
                        label = self._blur_arr(label)
                    ## ADDED BY BHK 4/22/22 ####################################################
                    if selector1 < 4:
                        image = np.flip(image, 2) ## Flips L/R
                        label = np.flip(label, 2) ## Flips L/R
                    #if selector2 < 4:                      
                    #    image = np.flip(image, 1) ## Flips U/D
                    #    label = np.flip(label, 1) ## Flips U/D
                    #if selector3 < 2:
                    #    image = np.flip(image, 0) ## Flips Z
                    #    label = np.flip(label, 0) ## Flips Z
                # if no augmentation, just crop
                else:
                    image, label = self._crop_arr(image, label)

                label = self._ensure_binary_label(label)

                # if index == 12:
                #     siko = np.random.randint(0, 100)
                #
                #     writer = sitk.ImageFileWriter()
                #     writer.SetFileName("/output/test1/sik-image-{}.nrrd".format(siko))
                #     writer.SetUseCompression(True)
                #     writer.Execute(sitk.GetImageFromArray(image))
                #
                #     writer = sitk.ImageFileWriter()
                #     writer.SetFileName("/output/test1/sik-label-{}.nrrd".format(siko))
                #     writer.SetUseCompression(True)
                #     writer.Execute(sitk.GetImageFromArray(label))

                #print (image.min(), image.max(), image.shape, image.dtype)
                #print (label.min(), label.max(), label.shape, label.dtype)
                label = np.rint(label) # added b/c there was slight noise coming in from augmentation-ask Ahmed?
                if label.min()!=0 or label.max()!=1 or len(np.unique(label))!=2:
                    print("WARNING!!test CHECK YOUR LABELS at index {}".format(index), "label min: ", label.min(), " label max: ", label.max(), " label unique values: ",np.unique(label))

                # reshape
                image = image.reshape(1, *image.shape)
                label = label.reshape(1, *label.shape)
                # append
                image_list.append(image)
                label_list.append(label)


        return np.array(image_list,'float32'), np.array(label_list,'float32')

    def _ensure_binary_label(self, arr):
        """
        Affine transformation causes some voxels to have a value + or - its original value, i.e. 2 or -1 respectively. Usually one voxel per patient is changed.
        """
        arr[arr>1] = 1
        arr[arr<0] = 0
        return arr

    def _crop_helper(self, arr, z_shift, y_shift, x_shift):
        return arr[z_shift:z_shift+self.final_shape[0], 
                   y_shift:y_shift+self.final_shape[1], 
                   x_shift:x_shift+self.final_shape[2]]

    def _crop_arr(self, image, label):
        """
        TODO: remove assert statements and put elsewhere more efficient.
        """
        #
        z_diff = image.shape[0] - self.final_shape[0]
        y_diff = image.shape[1] - self.final_shape[1]
        x_diff = image.shape[2] - self.final_shape[2]
        #
        assert z_diff >= 0, "final shape cannot be bigger the input shape along the z axis"
        assert y_diff >= 0, "final shape cannot be bigger the input shape along the y axis"
        assert x_diff >= 0, "final shape cannot be bigger the input shape along the x axis"
        #
        z_shift = np.random.randint(0, z_diff+1)
        y_shift = np.random.randint(0, y_diff+1)
        x_shift = np.random.randint(0, x_diff+1)
        #
        image = self._crop_helper(image, z_shift, y_shift, x_shift)
        label = self._crop_helper(label, z_shift, y_shift, x_shift)
        return image, label

    def _blur_arr(self, label):
        """
        Sigma: standard deviation for gaussian blur - must be positive.
        The larger the sigma, the more powerful the applied blur.
        """
        ratio = (len(label[label==1]) / label.size) * 100
        sigma = ratio * self.blur_multiplier
        #
        start = sigma * self.blur_random_range * -1
        end = sigma * self.blur_random_range
        #
        random_shift = np.random.uniform(start, end)
        #
        sigma = sigma + random_shift
        label = ndimage.filters.gaussian_filter(label.astype(np.float32), sigma, mode='constant')
        return label

    def _pepper_arr(self, arr):
        """
        Adds Gausian noise with std of 25 HU, equivilant to 0.0061 because the images are already remaped.
        """
        noise = np.random.normal(0, 0.0061, arr.shape)
        return arr + noise


########## move to class

    def _get_rotation_matrix(self, z_angle, y_angle, x_angle):
        # https://en.wikipedia.org/wiki/Rotation_matrix
        # rotation in Z
        rotation_matrix_z = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                                      [np.sin(z_angle), np.cos(z_angle), 0],
                                      [0, 0, 1]])
        # rotation in Y
        rotation_matrix_y = np.array([[np.cos(y_angle), 0, np.sin(y_angle)],
                                      [0, 1, 0],
                                      [-np.sin(y_angle), 0, np.cos(y_angle)]])
        # rotation in X
        rotation_matrix_x = np.array([[1,0,0],
                                      [0, np.cos(x_angle), -np.sin(x_angle)],
                                      [0, np.sin(x_angle), np.cos(x_angle)]])
        # combine
        matrix = np.dot(rotation_matrix_z, rotation_matrix_y)
        matrix = np.dot(matrix, rotation_matrix_x)
        return matrix

    def _get_shear_matrix(self, z_angle, y_angle, x_angle):
        # https://www.gatevidyalay.com/3d-shearing-in-computer-graphics-definition-examples/
        return np.array([[1, np.sin(x_angle), np.sin(x_angle)],
                         [np.sin(y_angle), 1, np.sin(y_angle)],
                         [np.sin(z_angle), np.sin(z_angle), 1]])

    def _get_scale_matrix(self, z_scale, y_scale, x_scale):
        """
        For example, z_scale=5, y_scale=1, x_scale=1,
        will make the sagittal and coronal slices smaller and keep the axial untouched.
        """
        return np.array([[x_scale, 0, 0],
                         [0, y_scale, 0],
                         [0, 0, z_scale]])

    def _transform(self, arr, fill,
                    z_rot, y_rot, x_rot,
                    z_shr, y_shr, x_shr,
                    z_scale, y_scale, x_scale):
        # get one transformation matrix
        transformation_matrix = self._get_rotation_matrix(z_rot, y_rot, x_rot)
        transformation_matrix = np.dot(
        transformation_matrix, self._get_shear_matrix(z_shr, y_shr, x_shr))
        transformation_matrix = np.dot(
        transformation_matrix, self._get_scale_matrix(z_scale, y_scale, x_scale))
        # allow for offset to always ensure image is centered
        array_center = (np.array(arr.shape)-1)/2.0
        displacement = np.dot(transformation_matrix, array_center)
        offset = array_center - displacement
        # apply
        return ndimage.interpolation.affine_transform( arr, transformation_matrix, offset=offset, order=3, mode='constant', cval=fill)

    def _get_affine_transformation_params(self):
        rot_ang = self.rot_ang_rng
        rot_ang_red = self.red_rot_shr*rot_ang
        shr_ang = self.shr_ang_rng
        shr_ang_red = self.red_rot_shr*shr_ang
        scale_lower = 1-self.scale_rng
        scale_upper = 1+self.scale_rng
        return {
             # + or -
            "z_rot": np.deg2rad(np.random.uniform(-rot_ang, rot_ang)),
            "y_rot": np.deg2rad(np.random.uniform(-rot_ang_red, rot_ang_red)),
            "x_rot": np.deg2rad(np.random.uniform(-rot_ang_red, rot_ang_red)),
             # + or -
            "z_shr": np.deg2rad(np.random.uniform(-shr_ang_red, shr_ang_red)),
            "y_shr": np.deg2rad(np.random.uniform(-shr_ang, shr_ang)),
            "x_shr": np.deg2rad(np.random.uniform(-shr_ang, shr_ang)),
            # 0.8 to 1.2, <1 zooms in, >1 zooms out
            "z_scale": np.random.uniform(scale_lower, scale_upper),
            "y_scale": np.random.uniform(scale_lower, scale_upper),
            "x_scale": np.random.uniform(scale_lower, scale_upper)
        }

    def _affine_transform_arr(self, image, label):
        """
        Rotates, shears, and scales - in that order.
        """
        # get random numbers
        affine_transformation_params = self._get_affine_transformation_params()
        # apply to image and label together
        image = self._transform(image, 0,  **affine_transformation_params)
        label = self._transform(label, 0,  **affine_transformation_params)
        return image, label







