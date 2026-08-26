import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_crop_bbox(img_h, img_w, crop_size, divisible=1):
    """
    Randomly get a crop bounding box for an image.

    Args:
        img_h (int): Height of the input image.
        img_w (int): Width of the input image.
        crop_size (tuple): Desired crop size (height, width).
        divisible (int): The crop offset will be divisible by this value. Default: 1.

    Returns:
        tuple: (crop_y1, crop_y2, crop_x1, crop_x2) coordinates for cropping.
    """
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)
    margin_h = max(img_h - crop_size[-2], 0)
    margin_w = max(img_w - crop_size[-1], 0)
    offset_h = np.random.randint(0, (margin_h + 1) // divisible) * divisible
    offset_w = np.random.randint(0, (margin_w + 1) // divisible) * divisible
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


def crop(img, crop_bbox):
    """
    Crop a region from the input image/tensor according to the given bounding box.

    Args:
        img (torch.Tensor): Input image or tensor. Can be 2D, 3D, or 4D.
        crop_bbox (tuple): (crop_y1, crop_y2, crop_x1, crop_x2) coordinates for cropping.

    Returns:
        torch.Tensor: Cropped image/tensor.

    Raises:
        NotImplementedError: If the input tensor dimension is not 2, 3, or 4.
    """
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        # For 4D tensor: [N, C, H, W]
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        # For 3D tensor: [C, H, W]
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        # For 2D tensor: [H, W]
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img


def match_shape(tensor, target_shape, mode='bilinear', align_corners=False):
    """
    Ensure the input tensor matches the target shape by resizing if necessary.

    Args:
        tensor (torch.Tensor): Input tensor to be checked and resized.
        target_shape (tuple): Target shape (height, width) to match.
        mode (str): Interpolation mode for resizing. Default is 'bilinear'.
        align_corners (bool): Whether to align corners during resizing. Default is False.

    Returns:
        torch.Tensor: Tensor with the matched shape.
    """
    if tensor.shape[-2:] != target_shape:
        tensor = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=align_corners)
    return tensor


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    """
    Resize the input tensor using interpolation.

    Args:
        input (torch.Tensor): The input tensor to be resized.
        size (tuple, optional): The target spatial size (height, width).
        scale_factor (float or tuple, optional): The scaling factor for resizing.
        mode (str): Interpolation mode. Default is 'nearest'.
        align_corners (bool, optional): If True, the corner pixels of input and output tensors are aligned.
        warning (bool): Whether to show alignment warnings. Default is True.

    Returns:
        torch.Tensor: The resized tensor.
    """
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            # Show warning if align_corners is set and the output size is not perfectly aligned
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}_{name}'] = value

    return outputs

# 
def downscale_label_ratio(gt,
                          scale_factor,
                          min_ratio,
                          num_classes,
                          ignore_index=255):
    """
    Downscale the ground truth label tensor by a given scale factor,
    using average pooling and ratio thresholding to determine valid regions.

    Args:
        gt (torch.Tensor): Input ground truth tensor of shape [bs, 1, H, W].
        scale_factor (int): Downscaling factor (must be > 1).
        min_ratio (float): Minimum ratio threshold for valid class assignment.
        num_classes (int): Number of classes.
        ignore_index (int): Index to ignore during training/evaluation. Default is 255.

    Returns:
        torch.Tensor: Downscaled label tensor of shape [bs, 1, H//scale_factor, W//scale_factor].
    """
    assert scale_factor > 1
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = num_classes

    # Replace ignore_index with a substitute class for pooling
    out = gt.clone()  # otw. next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    
    # One-hot encode and permute to [bs, num_classes + 1, H, W]
    out = F.one_hot(
        out.squeeze(1), num_classes=num_classes + 1).permute(0, 3, 1, 2)  # [bs, num_classes + 1, orig_h, orig_w]
    assert list(out.shape) == [bs, num_classes + 1, orig_h, orig_w], out.shape
    
    # Average pooling to downscale
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    
    # For each pixel, get the max ratio and corresponding class
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)  # [2, 1 , trg_h, trg_w], 
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class Upsample(nn.Module):
    
    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)