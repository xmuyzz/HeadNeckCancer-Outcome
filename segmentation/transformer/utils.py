import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.optim as optim
from volumentations import *
import nibabel as nib
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import AsDiscrete, Activations
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction


def compute_dice(inputs, targets, smooth = 1e-7):
  #inputs = tnf.softmax(inputs)
  inputs = inputs.view(-1)
  targets = targets.view(-1)
  intersection = (inputs * targets).sum()
  dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
  return (2.*intersection), (inputs.sum() + targets.sum())


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    tensor = tensor[:, 1:, :, :, :]
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    input = flatten(input)
    target = flatten(target)
    target = target.float()
    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect
    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))



