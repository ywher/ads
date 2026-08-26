import torch
import numpy as np


import torch

def calculate_iou_gpu(gt, pred, ignore_id=255):
    """
    Calculate class-level IoU and mean IoU (mIoU) for semantic segmentation, GPU accelerated version.

    Parameters:
        gt (torch.Tensor): Ground truth labels, shape [batch, h, w] or [h, w], type torch.long
        pred (torch.Tensor): Predicted labels, same shape as gt, type torch.long
        ignore_id (int): Class ID to ignore, default is 255

    Returns:
        dict: Dictionary of IoU for each class {class ID: IoU value}
        float: Mean IoU (mIoU) for all classes
    """
    assert gt.shape == pred.shape, "gt and pred must have the same shape"
    device = gt.device  # Automatically get the device of the input tensor

    # Flatten tensors and filter out ignored regions
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    mask = (gt_flat != ignore_id)
    valid_gt = gt_flat[mask]
    valid_pred = pred_flat[mask]

    # Get all valid classes
    classes = torch.unique(valid_gt)
    if classes.numel() == 0:
        return {}, 0.0  # Return empty if no valid classes

    # Vectorized computation of intersection and union
    cls_matrix = classes[:, None]  # [C, 1]
    gt_eq_cls = (valid_gt == cls_matrix).to(torch.int64)  # [C, N]
    pred_eq_cls = (valid_pred == cls_matrix).to(torch.int64)  # [C, N]

    intersection = (gt_eq_cls & pred_eq_cls).sum(dim=1).float()  # [C]
    union = (gt_eq_cls | pred_eq_cls).sum(dim=1).float()  # [C]

    # Handle division by zero
    epsilon = 1e-8
    iou = intersection / (union + epsilon)  # [C]

    # Convert to dictionary
    iou_dict = {cls.item(): iou[i].item() * 100 for i, cls in enumerate(classes)}
    miou = iou.mean().item() if iou.numel() > 0 else 0.0
    miou *= 100  # Convert to percentage

    return iou_dict, miou


def calculate_iou_numpy(gt: np.ndarray, pred: np.ndarray, ignore_id: int = 255):
    """
    Compute IoU and mIoU for each class in semantic segmentation.
    
    Parameters:
    - gt: Ground truth labels, size [h, w]
    - pred: Predicted labels, size [h, w]
    - ignore_id: Class ID to ignore, default is 255
    
    Returns:
    - iou_dict: Dictionary, keys are class IDs, values are corresponding IoU
    """
    assert gt.shape == pred.shape, "Ground truth and prediction must have the same shape"
    
    class_ids = np.unique(gt)
    class_ids = class_ids[class_ids != ignore_id]  # Filter out ignore_id
    
    iou_dict = {}
    for class_id in class_ids:
        gt_mask = (gt == class_id)
        pred_mask = (pred == class_id)
        
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        
        iou = intersection / union if union > 0 else 0.0
        iou_dict[class_id] = iou * 100
    
    miou = np.mean(list(iou_dict.values()))
    
    return iou_dict, miou


def calculate_iberhu_gpu(pred, gt, ignore_index=0, mask=None):
    """
    Calculate inverse Berhu loss metrics for depth estimation, GPU accelerated version.
    
    Parameters:
        pred (torch.Tensor): Predicted depth maps, shape [batch, 1, h, w] or [batch, h, w] or [h, w]
        gt (torch.Tensor): Ground truth depth maps, same shape as pred
        ignore_index (int): Depth value to ignore, default is 0 (typically for invalid depth)
        mask (torch.Tensor, optional): Optional binary mask for additional filtering
        
    Returns:
        dict: Dictionary of metrics {'rmse': RMSE value, 'abs_rel': Absolute Relative difference, 
                                    'sq_rel': Square Relative difference, 'berhu': BerHu error}
        float: Mean BerHu error (for comparison between models)
    """
    # Ensure tensors have the same shape
    assert pred.shape == gt.shape, "Prediction and ground truth shapes do not match"
    
    # Handle different input shapes
    if pred.dim() == 4:  # [batch, 1, h, w]
        pred = pred.squeeze(1)
        gt = gt.squeeze(1)
    
    device = pred.device
    
    # Create a valid mask excluding ignore values
    valid_mask = (gt != ignore_index)
    if mask is not None:
        valid_mask = valid_mask & mask.bool()
    
    # Skip calculation if no valid pixels
    if valid_mask.sum() == 0:
        return {
            'rmse': float('inf'),
            'abs_rel': float('inf'),
            'sq_rel': float('inf'),
            'berhu': float('inf')
        }, float('inf')
    
    # Apply mask to predictions and ground truth
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    # Compute l1 norm (absolute difference)
    diff = torch.abs(pred_valid - gt_valid)
    
    # Compute the threshold c for BerHu loss
    c = 0.2 * torch.max(diff).item()
    
    # Compute BerHu error
    berhu = torch.zeros_like(diff)
    mask_linear = diff <= c
    mask_quadratic = ~mask_linear
    
    # Linear region
    berhu[mask_linear] = diff[mask_linear]
    
    # Quadratic region
    if mask_quadratic.any():
        berhu[mask_quadratic] = (diff[mask_quadratic]**2 + c**2) / (2*c)
    
    mean_berhu = berhu.mean().item()
    
    # Additional metrics commonly used for depth evaluation
    rmse = torch.sqrt(torch.mean((pred_valid - gt_valid)**2)).item()
    abs_rel = torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item()
    sq_rel = torch.mean(((pred_valid - gt_valid)**2) / gt_valid).item()
    
    metrics = {
        'rmse': rmse,
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'berhu': mean_berhu
    }
    
    return metrics, mean_berhu


def calculate_iberhu_numpy(pred, gt, ignore_index=0, mask=None):
    """
    Calculate inverse Berhu loss metrics for depth estimation using NumPy.
    
    Parameters:
        pred (np.ndarray): Predicted depth maps, shape [h, w]
        gt (np.ndarray): Ground truth depth maps, same shape as pred
        ignore_index (int): Depth value to ignore, default is 0 (typically for invalid depth)
        mask (np.ndarray, optional): Optional binary mask for additional filtering
        
    Returns:
        dict: Dictionary of metrics {'rmse': RMSE value, 'abs_rel': Absolute Relative difference, 
                                    'sq_rel': Square Relative difference, 'berhu': BerHu error}
        float: Mean BerHu error (for comparison between models)
    """
    # Ensure tensors have the same shape
    assert pred.shape == gt.shape, "Prediction and ground truth shapes do not match"
    
    # Create a valid mask excluding ignore values
    valid_mask = (gt != ignore_index)
    if mask is not None:
        valid_mask = valid_mask & (mask > 0)
    
    # Skip calculation if no valid pixels
    if not np.any(valid_mask):
        return {
            'rmse': float('inf'),
            'abs_rel': float('inf'),
            'sq_rel': float('inf'),
            'berhu': float('inf')
        }, float('inf')
    
    # Apply mask to predictions and ground truth
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    # Compute l1 norm (absolute difference)
    diff = np.abs(pred_valid - gt_valid)
    
    # Compute the threshold c for BerHu loss
    c = 0.2 * np.max(diff)
    
    # Compute BerHu error
    berhu = np.zeros_like(diff)
    mask_linear = diff <= c
    mask_quadratic = ~mask_linear
    
    # Linear region
    berhu[mask_linear] = diff[mask_linear]
    
    # Quadratic region
    if np.any(mask_quadratic):
        berhu[mask_quadratic] = (diff[mask_quadratic]**2 + c**2) / (2*c)
    
    mean_berhu = np.mean(berhu)
    
    # Additional metrics commonly used for depth evaluation
    rmse = np.sqrt(np.mean((pred_valid - gt_valid)**2))
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    sq_rel = np.mean(((pred_valid - gt_valid)**2) / gt_valid)
    
    metrics = {
        'rmse': rmse,
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'berhu': mean_berhu
    }
    
    return metrics, mean_berhu


def calculate_pa_gpu(gt, pred, ignore_id=255):
    """
    PyTorch GPU version of per-class pixel accuracy calculation (supports batch input).
    
    Parameters:
        gt (torch.Tensor): Ground truth labels, shape [batch, h, w] or [h, w]
        pred (torch.Tensor): Predicted labels, same shape as gt
        ignore_id (int): Class ID to ignore
    
    Returns:
        dict: Pixel accuracy for each class {class ID: accuracy}
        float: Mean pixel accuracy
    """
    assert gt.shape == pred.shape, "Input shapes do not match"
    device = gt.device
    
    # Flatten tensors and filter out invalid regions
    gt_flat = gt.view(-1)
    pred_flat = pred.view(-1)
    mask = (gt_flat != ignore_id)
    valid_gt = gt_flat[mask]
    valid_pred = pred_flat[mask]
    
    # Get valid classes
    classes = torch.unique(valid_gt)
    if classes.numel() == 0:
        return {}, 0.0
    
    # Vectorized computation
    cls_matrix = classes[:, None]  # [C, 1]
    gt_eq_cls = (valid_gt == cls_matrix)          # [C, N]
    correct = (valid_pred == cls_matrix) & gt_eq_cls  # [C, N]
    
    # Count correct and total pixels
    correct_counts = correct.sum(dim=1).float()  # [C]
    total_counts = gt_eq_cls.sum(dim=1).float()  # [C]
    
    # Calculate accuracy
    epsilon = 1e-8
    acc = correct_counts / (total_counts + epsilon)  # [C]
    
    # Convert to dictionary
    acc_dict = {cls.item(): acc[i].item() * 100 for i, cls in enumerate(classes)}
    mean_acc = acc.mean().item() if acc.numel() > 0 else 0.0
    mean_acc *= 100  # Convert to percentage
    
    return acc_dict, mean_acc


def calculate_pa_numpy(gt, pred, ignore_id=255):
    """
    NumPy version of per-class pixel accuracy calculation.
    
    Parameters:
        gt (np.ndarray): Ground truth labels, shape [h, w]
        pred (np.ndarray): Predicted labels, same shape as gt
        ignore_id (int): Class ID to ignore
    
    Returns:
        dict: Pixel accuracy for each class {class ID: accuracy}
        float: Mean pixel accuracy
    """
    # Filter out invalid regions
    mask = (gt != ignore_id)
    valid_gt = gt[mask]
    valid_pred = pred[mask]
    
    # Get valid classes
    classes = np.unique(valid_gt)
    if len(classes) == 0:
        return {}, 0.0
    
    # Calculate per-class accuracy
    acc_dict = {}
    for cls in classes:
        # Pixels of the current class in ground truth
        gt_cls_mask = (valid_gt == cls)
        # Correctly predicted pixels
        correct = (valid_pred[gt_cls_mask] == cls).sum()
        total = gt_cls_mask.sum()
        
        # Handle division by zero
        acc = correct / total if total > 0 else 0.0
        acc_dict[cls] = acc * 100
    
    # Calculate mean accuracy
    mean_acc = np.mean(list(acc_dict.values())) if acc_dict else 0.0
    mean_acc *= 100  # Convert to percentage
    
    return acc_dict, mean_acc