import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image
from typing import Optional
from functools import partial
from torch import Tensor
from torchvision import transforms

from mmengine.hub import get_model  # segmentation
from transformers import DPTForDepthEstimation  # depth estimation


from torchvision.transforms import RandomCrop
import os




def image_grid(imgs, rows, cols):
    """Image grid for visualization."""
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def map_color_to_index(image):
    """Map colored segmentation image (RGB) into original label format (L).

    Args:
        image (torch.tensor): image tensor with shape (N, 3, H, W).
        dataset (str, optional): Dataset name. Defaults to 'ADE20K'.

    Returns:
        torch.tensor: mask tensor with shape (N, H, W).
    """
    palette = np.load('ade20k_palette.npy')

    image = image * 255
    palette_tensor = torch.tensor(palette, dtype=image.dtype, device=image.device)
    reshaped_image = image.permute(0, 2, 3, 1).reshape(-1, 3)

    # Calculate the difference of colors and find the index of the minimum distance
    indices = torch.argmin(torch.norm(reshaped_image[:, None, :] - palette_tensor, dim=-1), dim=-1)

    # Transform indices back to original shape
    return indices.view(image.shape[0], image.shape[2], image.shape[3])


def seg_label_transform(
        labels,
        output_size=(64, 64),
        interpolation=transforms.InterpolationMode.BILINEAR,
        max_size=None,
        antialias=True):
    """Adapt RGB seg_map into loss computation. \
    (1) Map the RGB seg_map into the original label format (Single Channel). \
    (2) Resize the seg_map into the same size as the output feature map. \
    (3) Remove background class if needed (usually for ADE20K).

    Args:
        labels (torch.tensor): Segmentation map. (N, 3, H, W) for ADE20K and (N, H, W) for COCO-Stuff.
        dataset_name (string): Dataset name. Default to 'ADE20K'.
        output_size (tuple): Resized image size, should be aligned with the output of segmentation models.
        interpolation (optional): _description_. Defaults to transforms.InterpolationMode.NEAREST.
        max_size (optional): Defaults to None.
        antialias (optional): Defaults to True.

    Returns:
        torch.tensor: formatted labels for loss computation.
    """

    # if dataset_name == 'limingcv/Captioned_ADE20K':
    labels = map_color_to_index(labels)
    labels = F.resize(labels, output_size, interpolation, max_size, antialias)

    # 0 means the background class in ADE20K
    # In a unified format, we use 255 to represent the background class for both ADE20K and COCO-Stuff
    labels = labels - 1
    labels[labels == -1] = 255

    return labels.long()

def depth_label_transform(
        labels,
        output_size=None,
        interpolation=transforms.InterpolationMode.BILINEAR,
        max_size=None,
        antialias=True
    ):

    if output_size is not None:
        labels = F.resize(labels, output_size, interpolation, max_size, antialias)
    return labels


def edge_label_transform(labels):
    return labels


def label_transform(labels, task, **args):
    if task in ['seg', 'inv_seg']:
        return seg_label_transform(labels, **args)
    elif task in ['depth', 'inv_depth']:
        return depth_label_transform(labels, **args)
    elif task in ['hed','inv_hed']:
        return edge_label_transform(labels, **args)
    else:
        raise NotImplementedError("Only support segmentation and edge detection for now.")


def group_random_crop(images, resolution):
    """
    Args:
        images (list of PIL Image or Tensor): List of images to be cropped.

    Returns:
        List of PIL Image or Tensor: List of cropped image.
    """

    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    for idx, image in enumerate(images):
        i, j, h, w = RandomCrop.get_params(image, output_size=resolution)
        images[idx] = F.crop(image, i, j, h, w)

    return images


    
