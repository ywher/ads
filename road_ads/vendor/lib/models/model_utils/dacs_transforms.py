# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    """
    Apply the full "strong" augmentation pipeline used by DACS:
      1) Class-wise mixing (one_mix) using a binary mask.
      2) Color jitter in the pixel space (with de/renormalization).
      3) Gaussian blur.

    Args:
        param (dict): Augmentation parameters. Expected keys:
            - 'mix': list/tuple of binary masks for mixing (or None).
            - 'color_jitter': float threshold to enable jitter.
            - 'color_jitter_s': strength or dict passed to kornia ColorJitter.
            - 'color_jitter_p': probability threshold for applying jitter.
            - 'mean': per-channel mean used for normalization.
            - 'std': per-channel std used for normalization.
            - 'blur': float threshold to enable Gaussian blur.
        data (Tensor|None): Input images tensor of shape [B, 3, H, W].
        target (Tensor|None): Corresponding labels tensor of shape [B, H, W].

    Returns:
        (Tensor|None, Tensor|None): Transformed data and target.
    """
    assert ((data is not None) or (target is not None))
    # Class-wise mixing for data and label
    data, target = one_mix(mask=param['mix'], data=data, target=target)  # (2,3,H,W),(2,H,W)->(1,3,H,W),(1,H,W)
    # Color jitter with de/renorm
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    # Gaussian blur
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target

def strong_transform_wo_mix(param, data=None, target=None):
    """
    Strong augmentation without the class-wise mixing step.
    Useful when you only want color jitter + blur.

    Args:
        param (dict): Same keys as strong_transform.
        data (Tensor|None): [B, 3, H, W]
        target (Tensor|None): [B, H, W]

    Returns:
        (Tensor|None, Tensor|None): Transformed data and target.
    """
    assert ((data is not None) or (target is not None))
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target

def get_mean_std(img_metas, dev):
    """
    Stack per-sample mean/std from the data pipeline into tensors.

    Args:
        img_metas (list[dict]): Each dict contains 'img_norm_cfg' with 'mean'/'std'.
        dev (torch.device): Target device.

    Returns:
        (Tensor, Tensor): mean, std with shapes [B, 3, 1, 1].
    """
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std

def get_mean_std_self(img_mean, img_std, length, dev):
    """
    Build mean/std tensors by repeating provided per-channel values.

    Args:
        img_mean (sequence[float]): Per-channel mean of length 3.
        img_std (sequence[float]): Per-channel std of length 3.
        length (int): Batch size to replicate.
        dev (torch.device): Target device.

    Returns:
        (Tensor, Tensor): mean, std with shapes [B, 3, 1, 1].
    """
    mean = [
        torch.as_tensor(img_mean, device=dev)
        for i in range(length)
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_std, device=dev)
        for i in range(length)
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std

def denorm(img, mean, std):
    """
    Convert normalized images back to [0,1] float space.

    Args:
        img (Tensor): Normalized image [B,3,H,W].
        mean (Tensor): [B,3,1,1]
        std (Tensor): [B,3,1,1]

    Returns:
        Tensor: Denormalized images in [0,1].
    """
    return img.mul(std).add(mean) / 255.0

def denorm_(img, mean, std):
    """
    In-place denormalization helper used before color jitter.

    Args:
        img (Tensor): [B,3,H,W]
        mean (Tensor): [B,3,1,1]
        std (Tensor): [B,3,1,1]
    """
    img.mul_(std).add_(mean).div_(255.0)

def renorm_(img, mean, std):
    """
    In-place re-normalization after color jitter.

    Args:
        img (Tensor): [B,3,H,W]
        mean (Tensor): [B,3,1,1]
        std (Tensor): [B,3,1,1]
    """
    img.mul_(255.0).sub_(mean).div_(std)

def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    """
    Apply color jitter with probability threshold.

    Steps:
      - Temporarily de-normalize images to real color space.
      - Apply kornia ColorJitter.
      - Normalize back.

    Args:
        color_jitter (float): Control switch; if > p, jitter is applied.
        mean (Tensor): [B,3,1,1]
        std (Tensor): [B,3,1,1]
        data (Tensor|None): [B,3,H,W]
        target (Tensor|None): Labels, passed through unchanged.
        s (float|dict): Jitter strength or dict of args for ColorJitter.
        p (float): Threshold to enable jitter.

    Returns:
        (Tensor|None, Tensor|None): Possibly jittered data and unchanged target.
    """
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)  # de-normalize before jitter
                data = seq(data)
                renorm_(data, mean, std)  # re-normalize after jitter
    return data, target

def gaussian_blur(blur, data=None, target=None):
    """
    Apply Gaussian blur with random sigma and kernel size proportional to image size.

    Args:
        blur (float): Control switch; if > 0.5, blur is applied.
        data (Tensor|None): [B,3,H,W]
        target (Tensor|None): Labels, passed through unchanged.

    Returns:
        (Tensor|None, Tensor|None): Possibly blurred data and unchanged target.
    """
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                # Kernel size is ~10% of spatial size, forced to be odd
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target

def get_class_masks(labels, class_ratio=0.5):
    """
    Randomly select a subset of classes for each label map and build binary masks.

    Args:
        labels (Tensor): Batch of label maps [B,H,W].
        class_ratio (float): Ratio of present classes to select (0~1].

    Returns:
        class_masks (list[Tensor]): Each is [1,H,W] with selected classes set to 1.
        num_class_choice (list[int]): Number of classes selected per sample.

    Notes:
        - Fast path: if class_ratio <= 0, return all-zero masks and zeros for counts.
    """
    # Fast path: non-positive ratio means no class is selected
    if class_ratio <= 0:
        class_masks = [torch.zeros(1, *label.shape, device=label.device, dtype=torch.long)
                       for label in labels]
        num_class_choice = [0 for _ in range(len(labels))]
        return class_masks, num_class_choice

    class_masks = []
    num_class_choice = []
    for label in labels:
        # Classes present in current label map (including ignore index if present)
        classes = torch.unique(label)
        nclasses = classes.shape[0]

        # If no class or computed selection is 0, return all-zero mask for this sample
        num_classes_to_choose = int(nclasses * class_ratio)
        if nclasses == 0 or num_classes_to_choose <= 0:
            class_masks.append(torch.zeros(1, *label.shape, device=label.device, dtype=torch.long))
            num_class_choice.append(0)
            continue

        # Randomly choose class indices from the present ones
        class_choice = np.random.choice(nclasses, num_classes_to_choose, replace=False)
        num_class_choice.append(len(class_choice))
        classes = classes[torch.tensor(class_choice).long()]
        # Build binary mask for selected classes
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks, num_class_choice

def get_context_class_masks(labels, class_ratio=0.5, num_classes=19):
    """
    Select classes randomly (as in get_class_masks), then augment the selection
    with context rules (Group1/Group2) based on dataset taxonomy.

    Group1 (object):
      - 6 (traffic light) <-> 5 (pole)
      - 7 (traffic sign)  <-> 5 (pole)
      - If pole is selected and light/sign exist in the image, also include them.

    Group2 (human-vehicle):
      - For 19-class datasets: 12 (rider) -> 18 (bicycle), 17 (motorcycle)
      - For 16-class datasets: 11 (rider) -> 15 (bicycle), 14 (motorcycle)

    Args:
        labels (Tensor): [B,H,W] label maps; 255 is treated as ignore and excluded.
        class_ratio (float): Base random selection ratio.
        num_classes (int): Dataset class count that decides Group2 ids.

    Returns:
        (list[Tensor], list[int]): class_masks (each [1,H,W]) and initial random
        selection counts (before adding the contextual supplements).
    """
    # Fast path: return all-zero masks
    if class_ratio <= 0:
        class_masks = [torch.zeros(1, *label.shape, device=label.device, dtype=torch.long)
                       for label in labels]
        num_class_choice = [0 for _ in range(len(labels))]
        return class_masks, num_class_choice
    
    class_masks = []
    num_class_choice = []

    # Group1 fixed ids
    pole_id = 5
    light_id = 6
    sign_id = 7

    # Group2 ids depend on dataset taxonomy
    if num_classes >= 19:
        rider_id = 12
        bicycle_id = 18
        motorcycle_id = 17
        enable_group2 = True
    elif num_classes == 16:
        rider_id = 11
        bicycle_id = 15
        motorcycle_id = 14
        enable_group2 = True
    else:
        # Unknown mapping; skip Group2 rules
        rider_id = bicycle_id = motorcycle_id = -1
        enable_group2 = False

    for label in labels:
        device = label.device
        # Collect present classes and exclude ignore index 255
        all_classes = torch.unique(label)
        all_classes = all_classes[all_classes != 255]
        nclasses = int(all_classes.numel())
        
        # If nothing to select
        num_choose = int(nclasses * class_ratio)
        if nclasses == 0 or num_choose <= 0:
            class_masks.append(torch.zeros(1, *label.shape, device=device, dtype=torch.long))
            num_class_choice.append(0)
            continue

        # Sample base classes
        if num_choose >= nclasses:
            chosen_classes = all_classes
            num_choose = nclasses
        else:
            idx = np.random.choice(nclasses, num_choose, replace=False)
            idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
            chosen_classes = all_classes[idx_t]
        num_class_choice.append(int(num_choose))

        # Start from the random set and then add contextual classes
        categories_new = chosen_classes.clone()

        def contains(cset, cid):
            """Utility: check if class id exists in a 1D class tensor."""
            if cid < 0:
                return False
            return (cset == cid).any()

        # ---------------- Group1 rules ----------------
        # If light/sign selected and pole exists in the image -> add pole
        if contains(categories_new, light_id) and contains(all_classes, pole_id):
            categories_new = torch.unique(torch.cat([categories_new, torch.tensor([pole_id], device=device)]))
        if contains(categories_new, sign_id) and contains(all_classes, pole_id):
            categories_new = torch.unique(torch.cat([categories_new, torch.tensor([pole_id], device=device)]))
        # If pole selected -> add light/sign if present in the image
        if contains(categories_new, pole_id):
            if contains(all_classes, light_id):
                categories_new = torch.unique(torch.cat([categories_new, torch.tensor([light_id], device=device)]))
            if contains(all_classes, sign_id):
                categories_new = torch.unique(torch.cat([categories_new, torch.tensor([sign_id], device=device)]))

        # ---------------- Group2 rules ----------------
        if enable_group2 and contains(categories_new, rider_id):
            if contains(all_classes, bicycle_id):
                categories_new = torch.unique(torch.cat([categories_new, torch.tensor([bicycle_id], device=device)]))
            if contains(all_classes, motorcycle_id):
                categories_new = torch.unique(torch.cat([categories_new, torch.tensor([motorcycle_id], device=device)]))

        # Build final mask with selected + context classes
        mask = generate_class_mask(label, categories_new).unsqueeze(0)  # (1,H,W)
        class_masks.append(mask)

    return class_masks, num_class_choice

def generate_class_mask(label, classes):
    """
    Create a binary mask for a set of class ids.

    Args:
        label (Tensor): Label map [H,W].
        classes (Tensor): 1D tensor of class ids to keep.

    Returns:
        Tensor: Binary mask [1,H,W] where selected classes are 1, others 0.

    Notes:
        - Uses broadcasting to compare each pixel with all selected ids,
          then sums over class dimension to obtain a single-channel mask.
    """
    label, classes = torch.broadcast_tensors(label, classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask

def one_mix(mask, data=None, target=None):
    """
    Perform class-wise mixing for a pair of images/labels (DA-style cutmix):
      out = mask * a + (1-mask) * b
    Applied independently to data and target if provided.

    Args:
        mask (Tensor|list|None): Binary mask(s) aligned with data/target. If None, do nothing.
        data (Tensor|None): Input images [2,3,H,W] to be mixed into [1,3,H,W].
        target (Tensor|None): Input labels [2,H,W] to be mixed into [1,H,W].

    Returns:
        (Tensor|None, Tensor|None): Mixed data and target.
    """
    if mask is None:
        return data, target
    if not (data is None):
        # Broadcast first mask to match the first image, then mix with the second
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
