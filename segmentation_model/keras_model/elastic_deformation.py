import numpy as np
import SimpleITK as sitk

class Elastic(object):
    def __init__(self, image, label):
        """
        Currently, casts numpy array to sitk then back again.
        This is better done at loading time?
        elastic_num_ctrl_pts: applied to shortest side (i.e. sitk_obj.GetSize()[-1])
        """
        self.image_sitk = sitk.GetImageFromArray(image)
        self.label_sitk = sitk.GetImageFromArray(label)
        self.elastic_num_ctrl_pts = 4
        self.elastic_lower = 10
        self.elastic_upper = 20
        self.elastic_std = np.random.uniform(self.elastic_lower, self.elastic_upper)
        self._set_mesh_shape()
        self._set_bspline_transform()

    def _set_mesh_shape(self):
        size = self.image_sitk.GetSize()
        self.mesh_shape = ((size[0]//size[2])*self.elastic_num_ctrl_pts,
                           (size[1]//size[2])*self.elastic_num_ctrl_pts,
                           self.elastic_num_ctrl_pts)

    def _set_bspline_transform(self):
        """
        Set bspline transform (done once on image, same numbers for label)
        """
        bspline_transform = sitk.BSplineTransformInitializer(self.image_sitk, self.mesh_shape)
        bspline_params = np.random.rand(len(bspline_transform.GetParameters()))
        bspline_params -= 0.5  # to offset (center) the deformation
        bspline_params *= self.elastic_std
        bspline_transform.SetParameters(bspline_params)
        self.bspline_transform = bspline_transform

    def _resample(self, sitk_obj, interpolation, fill, return_type):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_obj)
        resampler.SetTransform(self.bspline_transform)
        resampler.SetInterpolator(interpolation)
        resampler.SetDefaultPixelValue(fill)
        resampler.SetOutputPixelType(return_type)
        resampled = resampler.Execute(sitk_obj)
        return sitk.GetArrayFromImage(resampled)

    def _cleanup(self, arr):
        """
        This elastic deformation class will mess up the first and final few layers in all 3 axes. It will create swiss cheese holes in these layers.
        To fix this, we replace them with 0's in the image. Label is unlikley to be affected, since the tumor is centered and away from the ends in all 3 axes.
        """
        n_slices = 3
        fill = 0 # for images and labels.
        arr[:n_slices, :, :] = fill
        arr[-n_slices:, :, :] = fill
        arr[:, :n_slices, :] = fill
        arr[:, -n_slices:, :] = fill
        arr[:, :, :n_slices] = fill
        arr[:, :, -n_slices:] = fill
        return arr

    def run(self):
        image = self._resample(self.image_sitk, sitk.sitkLinear, 0, sitk.sitkFloat64)
        label = self._resample(self.label_sitk, sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
        return self._cleanup(image), label





