import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def parse_losses(losses):
    """
    Parse model output losses and format them.

    Args:
        losses (dict): Raw output dict from network containing losses
            Example: {'loss_seg': tensor(0.2), 'loss_aux': tensor(0.1)}

    Returns:
        tuple[Tensor, dict]: 
            - loss: Weighted sum of all losses (Tensor)
            - log_vars: Dict with all logging variables as Python scalars
    """
    log_vars = OrderedDict()
    # 优化1: 一次循环处理所有情况，无需类型判断分开
    for name, value in losses.items():
        if torch.is_tensor(value):
            mean_val = value.mean()
        elif isinstance(value, list):
            mean_val = sum(v.mean() for v in value)
        else:
            raise TypeError(f"{name} is not a tensor or list of tensors")
        log_vars[name] = mean_val

    # 优化2: 用生成器直接累加 loss，避免再次遍历
    loss = sum(val for key, val in log_vars.items() if "loss" in key)
    log_vars["loss"] = loss

    # 优化3: 只检查一次分布式状态，提升效率
    dist_on = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if dist_on else 1

    # 优化4: 用in-place减少clone与item调用
    for name in list(log_vars):
        v = log_vars[name]
        if dist_on:
            v = v.clone()  # 避免in-place影响原始数据
            dist.all_reduce(v)
            v /= world_size
        log_vars[name] = v.item()

    return loss, log_vars

def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

def get_patch_weights(image_size, patch_size, ignore_height):
    """
    Args:
        image_size: tuple (H, W) 表示图像尺寸
        patch_size: tuple (ph, pw) 表示patch尺寸
        ignore_height: int, 底部忽略的像素高度

    Returns:
        patch_weights: 长度为 n_rows * n_cols 的二值 numpy 数组，
                       对应每个 patch 是否参与损失计算 (1 表示参与, 0 表示忽略)
    """
    H, W = image_size
    ph, pw = patch_size
    # 总的 patch 数
    n_rows = H // ph
    n_cols = W // pw
    patch_weights = []
    # 有效区域高度
    valid_height = H - ignore_height
    # 一个 patch 如果其底边 (r+1)*ph 小于等于有效高度，则参与损失计算
    for r in range(n_rows):
        weight_row = 1 if (r+1)*ph <= valid_height else 0
        patch_weights.extend([weight_row] * n_cols)
    return np.array(patch_weights)

class CrossEntropyLoss(nn.Module): # Or you might have named it CrossEntropyLoss2d
    """
    Unified 2D Cross Entropy Loss function that supports optional class weights and pixel-wise weights.
    """
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        """
        Args:
            weight (torch.Tensor, optional): A manual rescaling weight given to each class.
                                             If given, has to be a Tensor of size C (number of classes).
                                             Default: None.
            ignore_index (int, optional): Specifies a target value that is ignored
                                          and does not contribute to the input gradient.
                                          Default: 255.
            reduction (str, optional): Specifies the reduction to apply to the output:
                                       'none' | 'mean' | 'sum'.
                                       'none': no reduction will be applied.
                                       'mean': the sum of the output will be divided by the number of elements in the output.
                                       'sum': the output will be summed.
                                       Default: 'mean'.
        """
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight  # Store class weights for the loss function
        self.ignore_index = ignore_index  # Store the target index to be ignored during loss calculation
        self.reduction = reduction  # Store the type of reduction to apply to the final loss

        # Use PyTorch's built-in CrossEntropyLoss, but set its reduction to 'none' initially.
        # This allows us to manually apply pixel-wise weights if provided,
        # and then apply the final desired reduction.
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none'  # Always 'none' here to get per-pixel losses first
        )

    def forward(self, inputs, targets, pixel_weights=None):
        """
        Args:
            inputs (torch.Tensor): The predicted logits from the model, with shape [B, C, H, W].
                                   B: Batch size, C: Number of classes, H: Height, W: Width.
            targets (torch.Tensor): The ground truth labels, with shape [B, H, W].
            pixel_weights (torch.Tensor, optional): Weights to be applied to the loss of each pixel,
                                                    with shape [B, H, W]. If None, no pixel-wise
                                                    weighting is applied. Default: None.
        Returns:
            torch.Tensor: The calculated loss. If reduction is 'none', the shape is [B, H, W] (per-pixel losses),
                          otherwise, it's a scalar (mean or sum of losses).
        """
        # Calculate the raw, unreduced cross-entropy loss (one value per pixel)
        loss = self.ce_loss(inputs, targets)  # Output shape: [B, H, W]

        # If pixel_weights are provided, apply them element-wise to the per-pixel losses
        if pixel_weights is not None:
            # Ensure that the shapes of loss and pixel_weights match for element-wise multiplication
            if loss.shape != pixel_weights.shape:
                raise ValueError(f"Loss shape {loss.shape} does not match pixel_weights shape {pixel_weights.shape}")
            loss = loss * pixel_weights # Apply pixel-wise weighting

        # Apply the specified reduction to the (potentially weighted) per-pixel losses
        if self.reduction == 'mean':
            loss = torch.mean(loss) # Compute the mean of all elements in the loss tensor
        elif self.reduction == 'sum':
            loss = torch.sum(loss)  # Compute the sum of all elements in the loss tensor
        elif self.reduction == 'none':
            pass  # No reduction applied, return per-pixel losses
        else:
            # Raise an error if an unsupported reduction type is provided
            raise ValueError(f"Unsupported reduction type: {self.reduction}. Supported types are 'none', 'mean', 'sum'.")

        return loss


class DyCELoss(nn.Module):
    """Dynamic class-balanced CE loss from SemiDAViL.

    DyCE first keeps the hardest pixels in the current mini-batch, then
    balances the selected pixels by their in-batch class frequency. Optional
    pixel weights are applied after hard-pixel selection, so confidence masks
    can still suppress unreliable pseudo-label regions.
    """

    def __init__(
        self,
        ignore_index=255,
        top_k_percent=0.2,
        omega=0.5,
        min_kept=1,
        eps=1e-6,
        reduction='mean',
    ):
        super(DyCELoss, self).__init__()
        if top_k_percent <= 0:
            raise ValueError('top_k_percent must be positive.')
        if omega < 0:
            raise ValueError('omega must be non-negative.')
        if reduction not in ('mean', 'sum'):
            raise ValueError("DyCELoss supports 'mean' and 'sum' reductions.")
        self.ignore_index = ignore_index
        self.top_k_percent = float(top_k_percent)
        self.omega = float(omega)
        self.min_kept = int(min_kept) if min_kept is not None else 1
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, inputs, targets, pixel_weights=None):
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        targets = targets.long()

        per_pixel = F.cross_entropy(
            inputs,
            targets,
            ignore_index=self.ignore_index,
            reduction='none',
        )
        valid = targets.ne(self.ignore_index)
        if pixel_weights is not None:
            if pixel_weights.dim() == 4 and pixel_weights.size(1) == 1:
                pixel_weights = pixel_weights.squeeze(1)
            if pixel_weights.shape != targets.shape:
                raise ValueError(
                    f"pixel_weights shape {pixel_weights.shape} does not match "
                    f"targets shape {targets.shape}")
            pixel_weights = pixel_weights.to(device=inputs.device, dtype=per_pixel.dtype)
            valid = valid & pixel_weights.gt(0)

        valid_losses = per_pixel[valid]
        if valid_losses.numel() == 0:
            return inputs.sum() * 0.0

        num_valid = valid_losses.numel()
        if self.top_k_percent <= 1.0:
            num_hard = int(np.ceil(num_valid * self.top_k_percent))
        else:
            num_hard = int(self.top_k_percent)
        num_hard = min(num_valid, max(self.min_kept, num_hard))
        hard_indices = torch.topk(valid_losses.detach(), k=num_hard).indices

        valid_labels = targets[valid]
        hard_labels = valid_labels[hard_indices]
        hard_losses = valid_losses[hard_indices]
        if pixel_weights is not None:
            hard_weights = pixel_weights[valid][hard_indices]
            hard_losses = hard_losses * hard_weights

        num_classes = inputs.shape[1]
        class_counts = torch.bincount(
            hard_labels.clamp(min=0, max=num_classes - 1),
            minlength=num_classes,
        ).to(device=inputs.device, dtype=hard_losses.dtype).clamp_min(1.0)
        class_balance = class_counts[hard_labels].pow(1.0 - self.omega)
        balanced = hard_losses / class_balance.clamp_min(self.eps)

        if self.reduction == 'sum':
            return balanced.sum()
        return balanced.sum() / (float(num_hard) ** self.omega + self.eps)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss
    
class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.CE(output, target)
        loss = torch.mean(loss * pixelWiseWeight)
        return loss

class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
    
    def forward(self, output, target):
        loss = self.BCE(output, target)
        return loss
    
class BCEWithLogitsLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)

    def forward(self, output, target):
        output = output.view(-1)
        target = target.view(-1)
        loss = self.BCE(output, target)
        return loss
    
class BCEWithLogitsLoss2d_Batch_Weighted(nn.Module):
    def __init__(self, weight=None):
        super(BCEWithLogitsLoss2d_Batch_Weighted, self).__init__()
        # 使用 reduction='none' 保留每个元素的 loss
        self.BCE = nn.BCEWithLogitsLoss(weight=weight, reduction='none')
    
    def forward(self, output, target, batch_weights):
        """
        Args:
            output: [batch, patch, num_classes]
            target: [batch, patch, num_classes]
            batch_weights: [batch] 或单个值
        """
        # 保证 batch_weights 为 tensor 且在同一设备上
        if not torch.is_tensor(batch_weights):
            batch_weights = torch.tensor(batch_weights, device=output.device, dtype=output.dtype)
        else:
            batch_weights = batch_weights.to(output.device).type_as(output)
        
        batch_size = output.size(0)
        # 将输出和标签展平：[batch * patch * num_classes]
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        loss = self.BCE(output_flat, target_flat)  # [batch * patch * num_classes]
        
        # 按样本重新 reshape，再对每个样本取平均
        loss_per_sample = loss.view(batch_size, -1).mean(dim=1)  # [batch]
        loss_weighted = loss_per_sample * batch_weights  # 每个样本乘以对应权重
        total_loss = loss_weighted.mean()
        return total_loss
    
class BCEWithLogitsLoss2d_Batch_Patch_Weighted(nn.Module):
    def __init__(self, weight=None, patch_weights=None):
        """
        Args:
            weight: BCE 权重参数
            patch_weights: 长度为 patch 数量的二值列表或数组，表示每个 patch 是否参与损失计算
        """
        super(BCEWithLogitsLoss2d_Batch_Patch_Weighted, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(weight=weight, reduction='none')
        if patch_weights is not None:
            # 保证 patch_weights 为 1D tensor，数据类型为 float
            self.patch_weights = torch.tensor(patch_weights, dtype=torch.float32)
        else:
            self.patch_weights = None

    def forward(self, output, target, batch_weights):
        """
        Args:
            output: [batch, patch, num_classes]
            target: [batch, patch, num_classes]
            batch_weights: [batch] 或单个值
        """
        batch_size, num_patches, num_classes = output.size()
        
        # 保证 batch_weights 为 tensor 且在同一设备上
        if not torch.is_tensor(batch_weights):
            batch_weights = torch.tensor(batch_weights, device=output.device, dtype=output.dtype)
        else:
            batch_weights = batch_weights.to(output.device).type_as(output)
        
        # 如果提供了 patch_weights，检查长度是否匹配
        if self.patch_weights is not None:
            if self.patch_weights.numel() != num_patches:
                raise ValueError(f"patch_weights 长度 ({self.patch_weights.numel()}) 与 patch 数量 ({num_patches}) 不匹配!")
            patch_weights = self.patch_weights.to(output.device)
        else:
            patch_weights = None

        # 计算每个 patch 的 loss（对每个 patch 内所有类别取平均）
        output_flat = output.contiguous().view(batch_size * num_patches, num_classes)
        target_flat = target.contiguous().view(batch_size * num_patches, num_classes)
        loss_per_patch = self.BCE(output_flat, target_flat).mean(dim=1)  # [batch * patch]
        loss_per_patch = loss_per_patch.view(batch_size, num_patches)    # [batch, patch]
        
        if patch_weights is not None:
            # 应用 patch 权重
            loss_per_patch = loss_per_patch * patch_weights  # [batch, patch]
            # 对每个样本归一化（除以有效 patch 数量，防止有效 patch 数量差异影响数值尺度）
            valid_counts = patch_weights.sum()
            valid_count = valid_counts if valid_counts > 0 else num_patches
            loss_per_sample = loss_per_patch.sum(dim=1) / valid_count
        else:
            # 未提供 patch_weights，则对所有 patch 求平均
            loss_per_sample = loss_per_patch.mean(dim=1)
        
        # 应用 batch 权重
        loss_per_sample = loss_per_sample * batch_weights
        total_loss = loss_per_sample.mean()
        return total_loss
    
class OhemCELoss(nn.Module):
    """Online hard-example mining cross entropy for segmentation logits."""

    def __init__(self, thresh=0.7, min_kept=None, ignore_index=255):
        super(OhemCELoss, self).__init__()
        if thresh <= 0 or thresh >= 1:
            raise ValueError(f'thresh should be in (0, 1), got {thresh}')
        self.thresh = float(thresh)
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels, pixel_weights=None):
        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        if (
            pixel_weights is not None
            and pixel_weights.dim() == 4
            and pixel_weights.size(1) == 1
        ):
            pixel_weights = pixel_weights.squeeze(1)

        losses = self.criteria(logits, labels)
        valid_mask = labels != self.ignore_index
        if pixel_weights is not None:
            valid_mask = valid_mask & (pixel_weights > 0)
            losses = losses * pixel_weights

        losses = losses[valid_mask]
        if losses.numel() == 0:
            return logits.sum() * 0

        if self.min_kept is None:
            min_kept = max(1, losses.numel() // 16)
        else:
            min_kept = min(int(self.min_kept), losses.numel())

        thresh = -torch.log(logits.new_tensor(self.thresh))
        hard_losses = losses[losses > thresh]
        if hard_losses.numel() < min_kept:
            hard_losses, _ = losses.topk(min_kept)
        return hard_losses.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255, use_sigmoid=False):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.use_sigmoid = use_sigmoid

    def forward(self, output, target):
        '''
        output: [N, C, H, W]
        target: [N, H, W]
        '''
        if self.use_sigmoid:
            output = torch.sigmoid(output)
        else:
            output = F.softmax(output, dim=1)
            
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target = target.clone()
                target[target == self.ignore_index] = target.min()
                
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True, use_sigmoid=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.use_sigmoid = use_sigmoid
        
        if use_sigmoid:
            self.focal_loss_fn = self._sigmoid_focal_loss
        else:
            if alpha is not None:
                if isinstance(alpha, (float, int)): 
                    alpha = torch.Tensor([alpha, 1 - alpha]).cuda()
                if isinstance(alpha, list): 
                    assert len(alpha) == 2
                    alpha = torch.Tensor(alpha).cuda()
            self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)
            self.focal_loss_fn = self._softmax_focal_loss

    def _softmax_focal_loss(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        return loss

    def _sigmoid_focal_loss(self, output, target):
        # 需要处理sigmoid情况下的focal loss
        target_one_hot = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        target_one_hot = target_one_hot.squeeze(1)
        
        p = torch.sigmoid(output)
        ce_loss = F.binary_cross_entropy_with_logits(output, target_one_hot, reduction='none')
        p_t = p * target_one_hot + (1 - p) * (1 - target_one_hot)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        return loss.mean(dim=1)

    def forward(self, output, target):
        loss = self.focal_loss_fn(output, target)
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

class BerhuLoss(nn.Module):
    """ Inverse Huber Loss """
    def __init__(self, ignore_index = 0):
        super(BerhuLoss, self).__init__()
        self.ignore_index = ignore_index
        self.l1 = torch.nn.L1Loss(reduction = 'none')

    def forward(self, prediction, ground_truth, imagemask=None):
        if imagemask is not None:
            mask = (ground_truth != self.ignore_index) & imagemask.to(torch.bool)
        else:
            mask = (ground_truth != self.ignore_index)
        difference = self.l1(torch.masked_select(prediction, mask), torch.masked_select(ground_truth, mask))
        with torch.no_grad():
            c = 0.2*torch.max(difference)
            mask = (difference <= c)

        lin = torch.masked_select(difference, mask)
        num_lin = lin.numel()

        non_lin = torch.masked_select(difference, ~mask)
        num_non_lin = non_lin.numel()

        total_loss_lin = torch.sum(lin)
        total_loss_non_lin = torch.sum((torch.pow(non_lin, 2) + (c**2))/(2*c))

        return (total_loss_lin + total_loss_non_lin)/(num_lin + num_non_lin)

class KLDivLossWithIgnore(nn.Module):
    def __init__(self, ignore_index=None, reduction='batchmean'):
        """
        KL散度损失，支持屏蔽某些类别不参与损失计算。

        Args:
            ignore_index (int or list of int): 忽略的类别索引（如背景类）19
            reduction (str): 'mean' or 'sum'
        """
        super().__init__()
        if ignore_index is None:
            ignore_index = []
        elif isinstance(ignore_index, int):
            ignore_index = [ignore_index]
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): [B, C]，模型预测 logits
            target (Tensor): [B, C]，目标 soft label（已归一化）

        Returns:
            Tensor: 单个标量损失
        """
        assert pred.shape == target.shape, "Shape mismatch between pred and target"

        log_q = F.log_softmax(pred, dim=1)
        log_p = torch.log(torch.clamp(target, min=1e-8))

        kl = target * (log_p - log_q)  # shape: [B, C]

        # mask ignored classes
        if self.ignore_index:
            mask = torch.ones_like(kl)
            mask[:, self.ignore_index] = 0
            kl = kl * mask

        # reduction
        loss = kl.sum(dim=1)  # sum over classes, resulting in [B,]
        if self.reduction == 'batchmean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # [B], no reduction

class GlobalClsLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 num_classes=19,
                 alpha=None,
                 hard_region_label=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.alpha = alpha
        self.hard_region_label = hard_region_label
        self.loss_fn = KLDivLossWithIgnore(ignore_index=num_classes, reduction='batchmean')
        # self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        # self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, cls_score, seg_label, return_seg_lb=False, input_cls_label=False):
        """
        cls_score: [B, C] or [B, R, C] or [B, R1, R2, C]
        seg_label: [B, H, W]  with label=255 ignored
        """
        if len(cls_score.shape) == 2:
            # Global classification loss
            if input_cls_label:
                seg_label_soft = seg_label
            else:
                seg_label_soft = self.compute_soft_label(seg_label, self.num_classes)  # [B, C]

            if self.alpha is not None:
                hard_label = F.one_hot(torch.argmax(seg_label_soft, dim=-1), num_classes=self.num_classes + 1).float()
                seg_label_soft = self.alpha * seg_label_soft + (1 - self.alpha) * hard_label

            log_prob = F.log_softmax(cls_score, dim=-1)
            loss_cls = self.loss_fn(log_prob, seg_label_soft)

            return (loss_cls, seg_label_soft) if return_seg_lb else loss_cls

        else:
            # Local classification loss (e.g., [B, R1, R2, C])
            B, *regions, C = cls_score.shape
            total_regions = torch.tensor(regions).prod().item()
            H, W = seg_label.shape[1:]

            # Convert seg_label to one-hot: [B, C, H, W]
            one_hot = F.one_hot(seg_label.clamp(0, self.num_classes - 1), num_classes=self.num_classes)
            one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
            one_hot[seg_label == 255] = 0  # ignore mask

            # Pool into region-level class ratios
            pooled = F.adaptive_avg_pool2d(one_hot, output_size=regions)  # [B, C, R1, R2]
            seg_label_soft = pooled.permute(0, 2, 3, 1).reshape(-1, C)  # [B * R1 * R2, C]

            if self.hard_region_label:
                seg_label_soft = torch.argmax(seg_label_soft, dim=-1)  # [B * R,] long
            elif self.alpha is not None:
                hard_label = F.one_hot(torch.argmax(seg_label_soft, dim=-1), num_classes=self.num_classes).float()
                seg_label_soft = self.alpha * seg_label_soft + (1 - self.alpha) * hard_label

            cls_score = cls_score.reshape(-1, self.num_classes)  # [B * R, C]

            if isinstance(seg_label_soft, torch.LongTensor) or seg_label_soft.dtype == torch.long:
                loss_cls = F.cross_entropy(cls_score, seg_label_soft)
            else:
                log_prob = F.log_softmax(cls_score, dim=-1)
                loss_cls = self.loss_fn(log_prob, seg_label_soft)

            return (loss_cls, seg_label_soft) if return_seg_lb else loss_cls

    @staticmethod
    def compute_soft_label(seg_label, num_classes):
        """
        计算每张图像中各个类别的像素比例，包括背景类（将255映射为背景类）
        
        Args:
            seg_label (Tensor): [B, H, W] 分割标签
            num_classes (int): 类别数量

        Returns:
            class_ratio (Tensor): [B, C+1] 每张图像中各类别的像素比例，包含背景类
        """
        B, H, W = seg_label.shape
        total_classes = num_classes + 1  # 包括背景类
        device = seg_label.device

        # 将255映射为 num_classes，作为背景类
        label_mapped = seg_label.clone()
        label_mapped[seg_label == 255] = num_classes

        # one-hot 编码，维度为 [B, H, W, C+1]
        one_hot = F.one_hot(label_mapped, num_classes=total_classes).float()

        # reshape 为 [B, H*W, C+1] 后在 dim=1 上求和，再除以总像素数
        class_count = one_hot.view(B, -1, total_classes).sum(dim=1)  # [B, C+1]
        class_ratio = class_count / (H * W)  # [B, C+1]

        return class_ratio

if __name__ == '__main__':
    # 设置图像尺寸、patch 尺寸、以及底部忽略高度
    img_size = (512, 512)
    patch_size = (32, 32)
    ignore_height = 0
    patch_weights = get_patch_weights(img_size, patch_size, ignore_height)
    
    rows, cols = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
    # 打印每一行的 patch 权重
    for i in range(rows):
        print(patch_weights[i*cols:(i+1)*cols])
    print("总有效 patch 数：", patch_weights.sum())
    print(f'patch weights shape: {patch_weights.shape}')
    patch_weights_reshape = patch_weights.reshape(rows, cols)
    print(patch_weights_reshape)
    
    # 定义损失函数
    criteria = BCEWithLogitsLoss2d_Batch_Patch_Weighted(patch_weights=patch_weights)
    criteria2 = BCEWithLogitsLoss2d_Batch_Weighted()
    
    # 模拟输入数据
    # 假设输出和目标张量 shape: [batch, patch, num_classes]
    batch = 2
    num_patches = rows * cols
    num_classes = 3
    output = torch.randn(batch, num_patches, num_classes)
    target = torch.randint(0, 2, (batch, num_patches, num_classes)).float()
    batch_weights = torch.tensor([1, 1], dtype=torch.float32)
    
    # 计算损失
    loss_patch = criteria(output, target, batch_weights)
    loss_batch = criteria2(output, target, batch_weights)
    
    print("带 patch 权重的损失：", loss_patch.item())
    print("仅带 batch 权重的损失：", loss_batch.item())
    
