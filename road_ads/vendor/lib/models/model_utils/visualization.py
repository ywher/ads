# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from utils.color_map import color_map
from utils.func import calculate_iou_gpu, calculate_pa_gpu, calculate_iberhu_numpy
from .dacs_transforms import denorm
from .plot_patch_mlcls import visualize_patch_level_src, visualize_patch_level_tar

Cityscapes_palette = [
    128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153,
    153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130,
    180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100,
    0, 0, 230, 119, 11, 32, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128,
    128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128,
    192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128,
    64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64,
    0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64,
    128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64,
    0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64,
    64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192,
    192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128,
    160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0,
    224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64,
    0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192,
    128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64,
    128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32,
    128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128,
    192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0,
    192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64,
    160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96,
    64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192,
    96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0,
    0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32,
    0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192,
    160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128,
    96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0,
    192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32,
    64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0,
    160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160,
    64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128,
    96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192,
    128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96,
    192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32,
    160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160,
    128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32,
    128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160,
    224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0,
    224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224,
    128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32,
    32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32,
    64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192,
    224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96,
    192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64,
    96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 0, 0, 0
]

SynCity_palette = [
    value
    for idx in range(len(color_map['syn_city']))
    for value in color_map['syn_city'][idx]
]


def get_debug_palette(num_classes=None, class_set=None):
    """Return the palette matching the active train/eval class set.

    SYNTHIA->Cityscapes uses the compressed 16-class `syn_city` order. Its
    class ids after vegetation differ from Cityscapes-19, so reusing the
    Cityscapes palette would make debug masks look valid but semantically
    miscolored.

    根据当前类别集合选择 debug 调色板。SYNTHIA->Cityscapes 使用压缩后的
    16 类 `syn_city` 顺序，vegetation 后的类别 id 与 Cityscapes-19 不一致。
    """
    if class_set == 'syn_city' or num_classes == 16:
        return SynCity_palette.copy()
    return Cityscapes_palette.copy()


# def colorize_mask(mask, palette):
#     zero_pad = 256 * 3 - len(palette)
#     for i in range(zero_pad):
#         palette.append(0)
#     new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#     new_mask.putpalette(palette)
#     return new_mask

def colorize_mask(mask, palette):
    palette = list(palette)
    # 计算需要填充的零的数量，以确保调色板长度为 256 * 3
    zero_pad = 256 * 3 - len(palette)
    
    # 将零填充到调色板中
    for i in range(zero_pad):
        palette.append(0)
    
    # 确保mask是2D数组
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # 处理维度问题
    if mask.ndim > 2:
        # 压缩多余的维度
        mask = mask.squeeze()
    
    # 确保是2D
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeezing, got shape: {mask.shape}")
    
    # 将掩码转换为 PIL 图像，并设置为调色板模式 ('P')
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    
    # 应用调色板到新掩码图像
    new_mask.putpalette(palette)
    
    # 返回彩色掩码图像
    return new_mask


def _colorize(img, cmap, mask_zero=False):
    vmin = np.min(img)
    vmax = np.max(img)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image

def get_segmentation_error_vis(seg, gt):
    error_mask = seg != gt
    error_mask[gt == 255] = 0
    out = seg.copy()
    out[error_mask == 0] = 255
    return out

def is_integer_array(a):
    return np.all(np.equal(np.mod(a, 1), 0))

def prepare_debug_out(title, out, mean, std):
    if 'Pseudo W' in title:
        title = 'Pseudo W: '
        unique_vals = np.unique(out)
        for unique_val in unique_vals:
            title += f'{unique_val:.2f}, '
        title = title[:-2]
    if len(out.shape) == 4 and out.shape[0] == 1:
        out = out[0]
    if len(out.shape) == 2:
        out = np.expand_dims(out, 0)
    assert len(out.shape) == 3
    if out.shape[0] == 3:
        if mean is not None:
            out = torch.clamp(denorm(out, mean, std), 0, 1)[0]
        out = dict(title=title, img=out)
    elif out.shape[0] > 3:
        out = torch.softmax(torch.from_numpy(out), dim=0).numpy()
        out = np.argmax(out, axis=0)
        out = dict(title=title, img=out, cmap='cityscapes')
    elif out.shape[0] == 1:
        if is_integer_array(out) and np.max(out) > 1:
            out = dict(title=title, img=out[0], cmap='cityscapes')
        elif np.min(out) >= 0 and np.max(out) <= 1:
            out = dict(title=title, img=out[0], cmap='viridis', vmin=0, vmax=1)
        else:
            out = dict(
                title=title, img=out[0], cmap='viridis', range_in_title=True)
    else:
        raise NotImplementedError(out.shape)
    return out

def subplotimg(ax,
               img,
               title,
               range_in_title=False,
               palette=Cityscapes_palette,
               **kwargs):
    if img is None:
        return
    with torch.no_grad():
        if torch.is_tensor(img):
            img = img.cpu()
        if len(img.shape) == 2:
            if torch.is_tensor(img):
                img = img.numpy()
        elif img.shape[0] == 1:
            if torch.is_tensor(img):
                img = img.numpy()
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            img = img.permute(1, 2, 0)
            if not torch.is_tensor(img):
                img = img.numpy()
        if kwargs.get('cmap', '') == 'cityscapes':
            kwargs.pop('cmap')
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize_mask(img, palette)

    if range_in_title:
        vmin = np.min(img)
        vmax = np.max(img)
        title += f' {vmin:.3f}-{vmax:.3f}'

    ax.imshow(img, **kwargs)
    ax.set_title(title)
    
'''
def visualize_multilabel_bars(pred_tensor, gt_tensor, title, class_names=None, 
                             fig_size=(10, 6), save_path=None, 
                             width=0.3, show=False):
    """
    优化后的可视化函数，解决标签溢出问题
    参数：
    - pred_tensor: torch.tensor, 形状为 (num_class,)，在 GPU 上
    - gt_tensor: torch.tensor, 形状为 (num_class,)，在 GPU 上
    - title: 图表标题
    - class_names: 类别名称列表，如果为 None, 则使用默认的索引
    - fig_size: 图像的尺寸，格式为 (width, height)
    - save_path: 图像保存路径，如果为 None, 则不保存
    - width: 柱状图的宽度
    - show: 是否显示图像
    """
    with torch.no_grad():
        pred = pred_tensor.cpu().numpy().squeeze()
        gt = gt_tensor.cpu().numpy().squeeze()

    assert pred.ndim == 1 and gt.ndim == 1, "pred_tensor 和 gt_tensor 必须是 1D 张量"
    num_classes = len(pred)
    
    x = np.arange(num_classes)
    class_names = class_names or [f'Class {i}' for i in range(num_classes)]
    
    # 创建一个新的 Figure 和 Axes
    fig, ax = plt.subplots(figsize=fig_size)
    
    # 绘制柱状图
    ax.bar(x - width/2, pred, width, label='Pred', color='tab:blue', alpha=0.7)
    ax.bar(x + width/2, gt, width, label='GT', color='tab:orange', alpha=0.7)

    # 优化标签设置
    ax.set_xticks(x)
    ax.set_xticklabels(
        class_names,
        rotation=90,
        ha='right',
        va='top',
        fontsize=8,
        rotation_mode='anchor'
    )
    
    ax.tick_params(axis='x', which='major', pad=2)
    
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, num_classes-0.5)
    
    ax.set_title(title, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.95))
    
    # 数值标签设置
    for i in x:
        ax.text(i - width/2, pred[i] + 0.02, f'{pred[i]:.2f}', va='bottom', ha='center', fontsize=6)
        ax.text(i + width/2, gt[i] + 0.02, f'{gt[i]:.0f}', va='bottom', ha='center', fontsize=6)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98], pad=0.5, h_pad=0.5, w_pad=0.5)
    
    # 如果类数超过 20，进一步调整字体
    if num_classes > 20:
        for label in ax.get_xticklabels():
            label.set_fontsize(6)
        ax.tick_params(axis='x', pad=1)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    if show:
        plt.show()
    else:
        plt.close(fig)
'''

def visualize_two_bars(pred_tensor, gt_tensor, title, class_names=None, ax=None, fig_size=(10, 6), width=0.3, show=False, adjust_y_range=False):
    """
    Optimized visualization function to solve label overflow issues.
    Args:
    - pred_tensor: torch.tensor, shape (num_class,), on GPU
    - gt_tensor: torch.tensor, shape (num_class,), on GPU
    - title: Title of the chart
    - class_names: List of class names, if None, default indices will be used
    - ax: matplotlib axes object (optional)
    - fig_size: Size of the figure, format (width, height)
    - width: Width of the bars in the bar chart
    - show: Whether to display the image
    """
    with torch.no_grad():
        pred = pred_tensor.cpu().numpy().squeeze()
        gt = gt_tensor.cpu().numpy().squeeze()

    assert pred.ndim == 1 and gt.ndim == 1, "pred_tensor and gt_tensor must be 1D tensors"
    num_classes = len(pred)
    
    x = np.arange(num_classes)
    class_names = class_names or [f'Class {i}' for i in range(num_classes)]
    
    # Use existing Axes if ax is provided; otherwise, create new Figure and Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.get_figure()
    
    # Plot bar chart
    ax.bar(x - width/2, pred, width, label='Pred', color='tab:blue', alpha=0.7)
    ax.bar(x + width/2, gt, width, label='GT', color='tab:orange', alpha=0.7)

    # Optimize label settings
    ax.set_xticks(x)
    ax.set_xticklabels(
        class_names,
        rotation=90,
        ha='right',
        va='top',
        fontsize=8,
        rotation_mode='anchor'
    )
    
    ax.tick_params(axis='x', which='major', pad=2)
    
    if adjust_y_range:
        y_max = max(np.max(pred), np.max(gt))
        ax.set_ylim(0, y_max + 0.05)
    else:
        ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.5, num_classes-0.5)
    
    ax.set_title(title, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.95))
    
    # Set value labels
    for i in x:
        ax.text(i - width/2, pred[i] + 0.02, f'{pred[i]:.2f}', ha='center', va='bottom', fontsize=6)
        ax.text(i + width/2, gt[i] + 0.02, f'{gt[i]:.2f}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98], pad=0.5, h_pad=0.5, w_pad=0.5)
    
    # Further adjust font size if the number of classes exceeds 20
    if num_classes > 20:
        for label in ax.get_xticklabels():
            label.set_fontsize(6)
        ax.tick_params(axis='x', pad=1)
    
    # if show:
    #     plt.show()
    # else:
    #     plt.close(fig)
    return ax


def visualize_three_bars_with_two_preds(pred1_tensor, pred2_tensor, gt_tensor, title, class_names=None, ax=None, fig_size=(12, 6), width=0.25, show=False, adjust_y_range=False):
    """
    绘制两个预测结果与一个真实标签的类别分布对比图。
    
    Args:
        pred1_tensor: torch.tensor, shape (num_class,)
        pred2_tensor: torch.tensor, shape (num_class,)
        gt_tensor: torch.tensor, shape (num_class,)
        title: 图标题
        class_names: 类别名称列表
        ax: matplotlib.axes.Axes 对象（可选）
        fig_size: 图像大小
        width: 单个柱子的宽度
        show: 是否展示图像
        adjust_y_range: 是否根据最大值自动调整 Y 轴范围
    Returns:
        ax: matplotlib.axes.Axes 对象
    """
    with torch.no_grad():
        pred1 = pred1_tensor.cpu().numpy().squeeze()
        pred2 = pred2_tensor.cpu().numpy().squeeze()
        gt = gt_tensor.cpu().numpy().squeeze()

    assert pred1.ndim == pred2.ndim == gt.ndim == 1, "所有输入必须为1维向量"
    num_classes = len(pred1)
    x = np.arange(num_classes)

    class_names = class_names or [f'Class {i}' for i in range(num_classes)]

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.get_figure()

    # 绘制三个柱状图
    ax.bar(x - width,     pred1, width=width, label='EMA Pred 1', color='tab:blue', alpha=0.7)
    ax.bar(x,             pred2, width=width, label='STU Pred 2', color='tab:green', alpha=0.7)
    ax.bar(x + width,     gt,    width=width, label='GT',     color='tab:orange', alpha=0.7)

    # X轴类别标签
    ax.set_xticks(x)
    ax.set_xticklabels(
        class_names,
        rotation=90,
        ha='right',
        va='top',
        fontsize=8,
        rotation_mode='anchor'
    )
    ax.tick_params(axis='x', which='major', pad=2)

    # Y轴范围
    if adjust_y_range:
        y_max = max(np.max(pred1), np.max(pred2), np.max(gt))
        ax.set_ylim(0, y_max + 0.05)
    else:
        ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.5, num_classes - 0.5)

    ax.set_title(title, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.95))

    # 标注数值
    for i in x:
        ax.text(i - width,     pred1[i] + 0.02, f'{pred1[i]:.2f}', ha='center', va='bottom', fontsize=6)
        ax.text(i,             pred2[i] + 0.02, f'{pred2[i]:.2f}', ha='center', va='bottom', fontsize=6)
        ax.text(i + width,     gt[i]    + 0.02, f'{gt[i]:.2f}',    ha='center', va='bottom', fontsize=6)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98], pad=0.5, h_pad=0.5, w_pad=0.5)

    if num_classes > 20:
        for label in ax.get_xticklabels():
            label.set_fontsize(6)
        ax.tick_params(axis='x', pad=1)

    # if show:
    #     plt.show()
    # else:
    #     plt.close(fig)

    return ax

def save_multilabel_plots(batch_index, out_dir, local_iter, dataset_class,
                          srx_aux_pred, mix_aux_pred, tar_aux_pred, 
                          src_multi_cls_lb, mix_multi_cls_lb, tar_multi_cls_lb, 
                          critetia_src_mlcls, critetia_mix_mlcls, pseudo_weight):
    with torch.no_grad():
        src_mlcls_loss = critetia_src_mlcls(srx_aux_pred, src_multi_cls_lb)
        mix_mlcls_loss = critetia_mix_mlcls(mix_aux_pred, mix_multi_cls_lb, torch.max(pseudo_weight))
        tar_mlcls_loss = critetia_src_mlcls(tar_aux_pred, tar_multi_cls_lb)
        save_path = os.path.join(out_dir, f'{(local_iter + 1):06d}_{batch_index}_src_tar_mlcls.pdf')
        fig, axes = plt.subplots(1, 3, figsize=(27, 6))
        visualize_multilabel_bars(
            pred_tensor=torch.sigmoid(srx_aux_pred),
            gt_tensor=src_multi_cls_lb,
            title=f"MultiLabel Pred SRC, loss: {src_mlcls_loss.item():.3f}",
            class_names=dataset_class,
            ax=axes[0],
            show=False
        )
        visualize_multilabel_bars(
            pred_tensor=torch.sigmoid(mix_aux_pred),
            gt_tensor=mix_multi_cls_lb,
            title=f"MultiLabel Pred MIX, loss: {mix_mlcls_loss.item():.3f}",
            class_names=dataset_class,
            ax=axes[1],
            show=False
        )
        visualize_multilabel_bars(
            pred_tensor=torch.sigmoid(tar_aux_pred),
            gt_tensor=tar_multi_cls_lb,
            title=f"MultiLabel Pred TRG, loss: {tar_mlcls_loss.item():.3f}",
            class_names=dataset_class,
            ax=axes[2],
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def save_debug_hrda_images(out_dir, local_iter, seg_debug, batch_size, means, stds,
                           palette=None):
    """
    Save debug images for HRDA Encoder Decoder model.
    
    Args:
        out_dir (str): Output directory path
        local_iter (int): Current iteration number
        seg_debug (dict): Dictionary containing debug information with structure:
            {
                'Source': {
                    'Image': tensor,
                    'Seg Pred': array,
                    'Seg GT': array,
                    ...
                },
                'Target': {...},
                'Mix': {...}
            }
        batch_size (int): Number of images in the batch
        means (tensor): Mean values for image denormalization
        stds (tensor): Standard deviation values for image denormalization
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save debug images for each sample in batch
    # if seg_debug['Source'] is not None and seg_debug:
    if seg_debug:
        for j in range(batch_size):
            # Create figure with subplots
            rows, cols = len(seg_debug), max(len(seg_debug[k]) for k in seg_debug.keys())
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(5 * cols, 5 * rows),
                gridspec_kw={
                    'hspace': 0.1,
                    'wspace': 0,
                    'top': 0.95,
                    'bottom': 0,
                    'right': 1,
                    'left': 0
                },
                squeeze=False,
            )

            # Plot each type of debug information
            for k1, (n1, outs) in enumerate(seg_debug.items()):
                for k2, (n2, out) in enumerate(outs.items()):
                    debug_out = prepare_debug_out(
                        f'{n1} {n2}', out[j], means, stds)
                    if debug_out.get('cmap') == 'cityscapes' and palette is not None:
                        debug_out['palette'] = palette
                    subplotimg(axs[k1][k2], **debug_out)
                    # # Handle RGB images
                    # if out.shape[1] == 3:
                    #     vis = torch.clamp(
                    #         denorm(out, means, stds), 0, 1)
                    #     subplotimg(axs[k1][k2], vis[j], f'{n1} {n2}')
                    # else:
                    #     # Handle segmentation masks and other single-channel images
                    #     if out.ndim == 3:
                    #         args = dict(cmap='cityscapes')
                    #     else:
                    #         args = dict(cmap='gray', vmin=0, vmax=1)
                    #     subplotimg(axs[k1][k2], out[j], f'{n1} {n2}',
                    #             **args)
            
            # Turn off axes for all subplots
            for ax in axs.flat:
                ax.axis('off')

            # Save the figure
            if isinstance(local_iter, int):
                img_path = os.path.join(out_dir, f'{(local_iter + 1):06d}_{j}_debug.png')
            elif isinstance(local_iter, str):
                img_path = os.path.join(out_dir, f'{local_iter}_{j}_debug.png')
            plt.savefig(img_path)
            plt.close()
       
def save_multilabel_patch_plots(batch_index, out_dir, local_iter,       
                                tar_patch_size, dataset_class,
                                vis_img, vis_trg_img, 
                                gt_semantic_seg, target_seg, pseudo_label,
                                srx_aux_pred, tar_aux_pred, 
                                multi_cls_lb, multi_cls_lb_tar):
    # plot for the source image
    visualize_patch_level_src(
                    image_tensor=vis_img,
                    seg_gt_tensor=gt_semantic_seg,
                    mlc_pred_tensor=torch.sigmoid(srx_aux_pred),
                    mlc_gt_tensor=multi_cls_lb,
                    patch_size=(tar_patch_size, tar_patch_size),
                    dataset_class=dataset_class,
                    save_path=os.path.join(out_dir, f'{(local_iter + 1):06d}_{batch_index}_src_mlcls_patch.pdf'))
    # plot for target image
    visualize_patch_level_tar(
                    image_tensor=vis_trg_img,
                    seg_gt_tensor=target_seg,
                    pseudo_label_tensor=pseudo_label,
                    mlc_pred_tensor=torch.sigmoid(tar_aux_pred),
                    mlc_gt_tensor=multi_cls_lb_tar,
                    patch_size=(tar_patch_size, tar_patch_size),
                    dataset_class=dataset_class,
                    save_path=os.path.join(out_dir, f'{(local_iter + 1):06d}_{batch_index}_tar_mlcls_patch.pdf'))


def visualize_multilabel_bars(pred_tensor, gt_tensor, title, class_names=None, ax=None, fig_size=(10, 6), width=0.3, show=False):
    """
    优化后的可视化函数，解决标签溢出问题
    """
    with torch.no_grad():
        pred = pred_tensor.cpu().numpy().squeeze()
        gt = gt_tensor.cpu().numpy().squeeze()

    assert pred.ndim == 1 and gt.ndim == 1
    num_classes = len(pred)
    
    x = np.arange(num_classes)
    class_names = class_names or [f'Class {i}' for i in range(num_classes)]
    
    # Use existing Axes if ax is provided; otherwise, create new Figure and Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # 绘制柱状图
    ax.bar(x - width/2, pred, width, label='Pred', color='tab:blue', alpha=0.7)
    ax.bar(x + width/2, gt, width, label='GT', color='tab:orange', alpha=0.7)

    # 优化标签设置
    ax.set_xticks(x)
    ax.set_xticklabels(
        class_names,
        rotation=90,
        ha='right',
        va='top',
        fontsize=8,
        rotation_mode='anchor'
    )
    
    ax.tick_params(axis='x', which='major', pad=2)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-0.5, num_classes-0.5)
    
    ax.set_title(title, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.95))
    
    # 数值标签设置
    for i in x:
        ax.text(i - width/2, pred[i] + 0.02, f'{pred[i]:.2f}', 
                ha='center', va='bottom', fontsize=6)
        ax.text(i + width/2, gt[i] + 0.02, f'{gt[i]:.0f}',
                ha='center', va='bottom', fontsize=6)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98], pad=0.5, h_pad=0.5, w_pad=0.5)
    
    if num_classes > 20:
        for label in ax.get_xticklabels():
            label.set_fontsize(6)
        ax.tick_params(axis='x', pad=1)
    
    return ax


def save_debug_images(self, 
                    batch_size, means, stds, dataset_class,
                    src_img, tar_img, mix_img, 
                    src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                    num_class_choice, mix_masks, mixed_seg_weight, fdist_mask=None, gt_rescale=None,
                    src_seg_pred=None, mix_seg_pred=None, dep_tar=None, dep_mix=None):
    # output dir
    out_dir = os.path.join(self.cfg.respth, 'debug')
    os.makedirs(out_dir, exist_ok=True)

    # denorm the images
    vis_src_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
    vis_tar_img = torch.clamp(denorm(tar_img, means, stds), 0, 1)
    vis_mix_img = torch.clamp(denorm(mix_img, means, stds), 0, 1)
    seg_kwargs = dict(
        cmap='cityscapes',
        palette=getattr(
            self,
            'debug_palette',
            get_debug_palette(getattr(self, 'num_classes', None),
                              getattr(self, 'class_set', None)),
        ),
    )


    for j in range(batch_size):
        if dep_tar is not None and dep_mix is not None:
            rows, cols = 5, 4
        else:
            rows, cols = 4, 4
        fig, axs = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )
        ### first row
        # plot the src and tar images in the first column
        subplotimg(axs[0][0], vis_src_img[j], 'Src_Img')
        subplotimg(axs[0][1], vis_tar_img[j], 'Tar_Img')
        subplotimg(axs[0][2], vis_mix_img[j], 'Mix_Img')
        
        mix_mask_ratio = torch.sum(mix_masks[j][0][0]) / (mix_masks[j][0][0].shape[0] * mix_masks[j][0][0].shape[1])
        subplotimg(axs[0][3], mix_masks[j][0], f'Mix_Mask: {num_class_choice[j]}, {mix_mask_ratio:.2f}', cmap='gray')

        
        ### second row
        # src_seg_lbl
        num_class = len(torch.unique(src_seg_lbl[j]))
        if 255 in torch.unique(src_seg_lbl[j]):
            num_class -= 1
        subplotimg(axs[1][0], src_seg_lbl[j], f'Src_Seg, cls: {num_class}', **seg_kwargs)
        # tar_seg_lbl
        num_class = len(torch.unique(tar_seg_lbl[j]))
        if 255 in torch.unique(tar_seg_lbl[j]):
            num_class -= 1
        subplotimg(axs[1][1], tar_seg_lbl[j], f'Tar_Seg, cls: {num_class}', **seg_kwargs)
        # mix_pseudo_label
        if mix_seg_lbl is not None:
            num_clas = len(torch.unique(mix_seg_lbl[j]))
            if 255 in torch.unique(mix_seg_lbl[j]):
                num_clas -= 1
            subplotimg(axs[1][2], mix_seg_lbl[j], f'Mix_L, cls: {num_clas}', **seg_kwargs)
        if fdist_mask is not None:
            subplotimg(axs[1][3], fdist_mask[j][0], 'FDist Mask', cmap='gray')
        # # mix mask old
        # mix_w_old_str = 'Mix W Old: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(mix_seg_weight_old[j]))
        # subplotimg(axs[1][3], mix_seg_weight_old[j], mix_w_old_str, vmin=0, vmax=1)
        
        ### third row
        # src_pred
        src_miou = calculate_iou_gpu(src_seg_pred[j], src_seg_lbl[j])[1]
        src_mpa = calculate_pa_gpu(src_seg_pred[j], src_seg_lbl[j])[1]
        subplotimg(axs[2][0], src_seg_pred[j], f'Src_Pred, mIoU: {src_miou:.1f}, mPA: {src_mpa:.1f}', **seg_kwargs)
        
        # tar_pred
        pl_miou = calculate_iou_gpu(pseudo_label[j], tar_seg_lbl[j])[1]
        pl_mpa = calculate_pa_gpu(pseudo_label[j], tar_seg_lbl[j])[1]
        subplotimg(axs[2][1], pseudo_label[j], f'Tar_PL, mIoU: {pl_miou:.1f}, mPA: {pl_mpa:.1f}', **seg_kwargs)
        
        # mix pred
        num_class = len(torch.unique(mix_seg_pred[j]))
        subplotimg(axs[2][2], mix_seg_pred[j], f'Mix_Pred, cls: {num_class}', **seg_kwargs)
        
        # mix weight
        # mix_w_str = 'Mix W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(mixed_seg_weight[j]))
        mix_w_str = 'Mix W'
        subplotimg(axs[2][3], mixed_seg_weight[j], mix_w_str, vmin=0, vmax=1)
        
        
        ### fourth row

        # for debug
        pl_w_str = 'Pseudo W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(pseudo_weight[j]))
        subplotimg(axs[3][0], pseudo_weight[j], pl_w_str, vmin=0, vmax=1)
        
        # mask_target_pseudo_label
        mask_target_pseudo = pseudo_label[j].clone()
        mask_target_pseudo[pseudo_mask[j] == 0] = 255
        mask_target_label = tar_seg_lbl[j].clone()
        mask_target_label[pseudo_mask[j] == 0] = 255

        mpl_miou = calculate_iou_gpu(mask_target_label, mask_target_pseudo)[1]
        mpl_mpa = calculate_pa_gpu(mask_target_label, mask_target_pseudo)[1]
        subplotimg(axs[3][1], mask_target_pseudo, f'Mask Tar_PL, mIoU: {mpl_miou:.1f}, mPA: {mpl_mpa:.1f}', **seg_kwargs)
        
        # mask_target_label
        subplotimg(axs[3][2], mask_target_label, 'Mask Tar_L', **seg_kwargs)
        
        if gt_rescale is not None:
            subplotimg(axs[3][3], gt_rescale[j], 'Scaled GT', **seg_kwargs)
            
        
        ### fifth row
        if dep_tar is not None and dep_mix is not None:
            # plot the depth tar and mix
            subplotimg(axs[4][0], vis_tar_img[j], 'Tar_Img_Depth')
            subplotimg(axs[4][1], dep_tar[j], 'Tar_Depth', cmap='viridis')
            subplotimg(axs[4][2], vis_mix_img[j], 'Mix_Img_Depth')
            subplotimg(axs[4][3], dep_mix[j], 'Mix_Depth', cmap='viridis')
            # plot the depth tar and mix

        for ax in axs.flat:
            ax.axis('off')

        plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
        plt.close(fig)
    
def mask_tar_img_with_token_mask(tar_img, token_mask, patch_size):
    # tar_img: [3, H, W], token_mask: [L], patch_size: int
    img = tar_img.clone()
    C, H, W = img.shape
    num_patches = token_mask.shape[0]
    grid_h = H // patch_size
    grid_w = W // patch_size
    assert grid_h * grid_w == num_patches, "patch数量与图片尺寸不匹配"
    mask = token_mask.reshape(grid_h, grid_w)
    for ph in range(grid_h):
        for pw in range(grid_w):
            if mask[ph, pw]:
                img[:, ph*patch_size:(ph+1)*patch_size, pw*patch_size:(pw+1)*patch_size] = 0
    return img
    
def save_debug_tkm_images(self, 
                    batch_size, means, stds, dataset_class, patch_size,
                    src_img, tar_img, mix_img, token_mask,
                    src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                    num_class_choice, mix_masks, mixed_seg_weight, fdist_mask=None, gt_rescale=None,
                    src_seg_pred=None, mix_seg_pred=None, tkm_seg_pred=None, dep_tar=None, dep_mix=None):
    # output dir
    out_dir = os.path.join(self.cfg.respth, 'debug')
    os.makedirs(out_dir, exist_ok=True)

    # denorm the images
    vis_src_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
    vis_tar_img = torch.clamp(denorm(tar_img, means, stds), 0, 1)
    vis_mix_img = torch.clamp(denorm(mix_img, means, stds), 0, 1)


    for j in range(batch_size):
        if dep_tar is not None and dep_mix is not None:
            rows, cols = 5, 5
        else:
            rows, cols = 4, 5
        fig, axs = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )
        ### first row
        # plot the src and tar images in the first column
        subplotimg(axs[0][0], vis_src_img[j], 'Src_Img')
        subplotimg(axs[0][1], vis_tar_img[j], 'Tar_Img')
        subplotimg(axs[0][2], vis_mix_img[j], 'Mix_Img')
        
        # patch_size = 16  # 根据实际模型设置    
        masked_tar_img = mask_tar_img_with_token_mask(vis_tar_img[j], token_mask[j], patch_size)
        subplotimg(axs[0][3], masked_tar_img, 'Tar_Img_Masked')
        
        mix_mask_ratio = torch.sum(mix_masks[j][0][0]) / (mix_masks[j][0][0].shape[0] * mix_masks[j][0][0].shape[1])
        subplotimg(axs[0][4], mix_masks[j][0], f'Mix_Mask: {num_class_choice[j]}, {mix_mask_ratio:.2f}', cmap='gray')

        
        ### second row
        # src_seg_lbl
        num_class = len(torch.unique(src_seg_lbl[j]))
        if 255 in torch.unique(src_seg_lbl[j]):
            num_class -= 1
        subplotimg(axs[1][0], src_seg_lbl[j], f'Src_Seg, cls: {num_class}', cmap='cityscapes')
        # tar_seg_lbl
        num_class = len(torch.unique(tar_seg_lbl[j]))
        if 255 in torch.unique(tar_seg_lbl[j]):
            num_class -= 1
        subplotimg(axs[1][1], tar_seg_lbl[j], f'Tar_Seg, cls: {num_class}', cmap='cityscapes')
        # mix_pseudo_label
        if mix_seg_lbl is not None:
            num_clas = len(torch.unique(mix_seg_lbl[j]))
            if 255 in torch.unique(mix_seg_lbl[j]):
                num_clas -= 1
            subplotimg(axs[1][2], mix_seg_lbl[j], f'Mix_L, cls: {num_clas}', cmap='cityscapes')
        if fdist_mask is not None:
            subplotimg(axs[1][3], fdist_mask[j][0], 'FDist Mask', cmap='gray')
        # # mix mask old
        # mix_w_old_str = 'Mix W Old: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(mix_seg_weight_old[j]))
        # subplotimg(axs[1][3], mix_seg_weight_old[j], mix_w_old_str, vmin=0, vmax=1)
        
        ### third row
        # src_pred
        src_miou = calculate_iou_gpu(src_seg_pred[j], src_seg_lbl[j])[1]
        src_mpa = calculate_pa_gpu(src_seg_pred[j], src_seg_lbl[j])[1]
        subplotimg(axs[2][0], src_seg_pred[j], f'Src_Pred, mIoU: {src_miou:.1f}, mPA: {src_mpa:.1f}', cmap='cityscapes')
        
        # tar_pred
        pl_miou = calculate_iou_gpu(pseudo_label[j], tar_seg_lbl[j])[1]
        pl_mpa = calculate_pa_gpu(pseudo_label[j], tar_seg_lbl[j])[1]
        subplotimg(axs[2][1], pseudo_label[j], f'Tar_PL, mIoU: {pl_miou:.1f}, mPA: {pl_mpa:.1f}', cmap='cityscapes')
        
        # mix pred
        num_class = len(torch.unique(mix_seg_pred[j]))
        subplotimg(axs[2][2], mix_seg_pred[j], f'Mix_Pred, cls: {num_class}', cmap='cityscapes')
        
        # tkm pred
        num_class = len(torch.unique(tkm_seg_pred[j]))
        subplotimg(axs[2][3], tkm_seg_pred[j], f'TKM_Pred, cls: {num_class}', cmap='cityscapes')
        
        # mix weight
        # mix_w_str = 'Mix W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(mixed_seg_weight[j]))
        mix_w_str = 'Mix W'
        subplotimg(axs[2][4], mixed_seg_weight[j], mix_w_str, vmin=0, vmax=1)
        
        
        ### fourth row

        # for debug
        pl_w_str = 'Pseudo W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(pseudo_weight[j]))
        subplotimg(axs[3][0], pseudo_weight[j], pl_w_str, vmin=0, vmax=1)
        
        # mask_target_pseudo_label
        mask_target_pseudo = pseudo_label[j].clone()
        mask_target_pseudo[pseudo_mask[j] == 0] = 255
        mask_target_label = tar_seg_lbl[j].clone()
        mask_target_label[pseudo_mask[j] == 0] = 255

        mpl_miou = calculate_iou_gpu(mask_target_label, mask_target_pseudo)[1]
        mpl_mpa = calculate_pa_gpu(mask_target_label, mask_target_pseudo)[1]
        subplotimg(axs[3][1], mask_target_pseudo, f'Mask Tar_PL, mIoU: {mpl_miou:.1f}, mPA: {mpl_mpa:.1f}', cmap='cityscapes')
        
        # mask_target_label
        subplotimg(axs[3][2], mask_target_label, 'Mask Tar_L', cmap='cityscapes')
        
        if gt_rescale is not None:
            subplotimg(axs[3][3], gt_rescale[j], 'Scaled GT', cmap='cityscapes')
            
        
        ### fifth row
        if dep_tar is not None and dep_mix is not None:
            # plot the depth tar and mix
            subplotimg(axs[4][0], vis_tar_img[j], 'Tar_Img_Depth')
            subplotimg(axs[4][1], dep_tar[j], 'Tar_Depth', cmap='viridis')
            subplotimg(axs[4][2], vis_mix_img[j], 'Mix_Img_Depth')
            subplotimg(axs[4][3], dep_mix[j], 'Mix_Depth', cmap='viridis')
            # plot the depth tar and mix

        for ax in axs.flat:
            ax.axis('off')

        plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
        plt.close(fig)
        
def save_debug_mic_images(self, 
                    batch_size, means, stds, dataset_class,
                    src_img, tar_img, mix_img, masked_img, 
                    src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl, masked_gt, masked_pseudo_weight,
                    num_class_choice, mix_masks, mixed_seg_weight, fdist_mask=None, gt_rescale=None,
                    src_seg_pred=None, mix_seg_pred=None, masked_pred=None, dep_tar=None, dep_mix=None):
    # output dir
    out_dir = os.path.join(self.cfg.respth, 'debug')
    os.makedirs(out_dir, exist_ok=True)

    # denorm the images
    vis_src_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
    vis_tar_img = torch.clamp(denorm(tar_img, means, stds), 0, 1)
    vis_mix_img = torch.clamp(denorm(mix_img, means, stds), 0, 1)
    vis_masked_img = torch.clamp(denorm(masked_img, means, stds), 0, 1)


    for j in range(batch_size):
        rows, cols = 4, 5
        if dep_tar is not None and dep_mix is not None:
            rows, cols = 5, 5
        fig, axs = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )
        ### first row
        # plot the src and tar images in the first column
        subplotimg(axs[0][0], vis_src_img[j], 'Src_Img')
        subplotimg(axs[0][1], vis_tar_img[j], 'Tar_Img')
        subplotimg(axs[0][2], vis_mix_img[j], 'Mix_Img')
        subplotimg(axs[0][3], vis_masked_img[j], 'Masked_Img')
        
        mix_mask_ratio = torch.sum(mix_masks[j][0][0]) / (mix_masks[j][0][0].shape[0] * mix_masks[j][0][0].shape[1])
        subplotimg(axs[0][4], mix_masks[j][0], f'Mix_Mask: {num_class_choice[j]}, {mix_mask_ratio:.2f}', cmap='gray')

        
        ### second row
        # src_seg_lbl
        num_class = len(torch.unique(src_seg_lbl[j]))
        if 255 in torch.unique(src_seg_lbl[j]):
            num_class -= 1
        subplotimg(axs[1][0], src_seg_lbl[j], f'Src_Seg, cls: {num_class}', cmap='cityscapes')
        # tar_seg_lbl
        num_class = len(torch.unique(tar_seg_lbl[j]))
        if 255 in torch.unique(tar_seg_lbl[j]):
            num_class -= 1
        subplotimg(axs[1][1], tar_seg_lbl[j], f'Tar_Seg, cls: {num_class}', cmap='cityscapes')
        # mix_pseudo_label
        if mix_seg_lbl is not None:
            num_class = len(torch.unique(mix_seg_lbl[j]))
            if 255 in torch.unique(mix_seg_lbl[j]):
                num_class -= 1
            subplotimg(axs[1][2], mix_seg_lbl[j], f'Mix_L, cls: {num_class}', cmap='cityscapes')
        
        num_class = len(torch.unique(masked_gt[j]))
        if masked_gt is not None:
            subplotimg(axs[1][3], masked_gt[j], f'Masked GT, cls: {num_class}', cmap='cityscapes')
        
        if fdist_mask is not None:
            subplotimg(axs[1][4], fdist_mask[j][0], 'FDist Mask', cmap='gray')
        # # mix mask old
        # mix_w_old_str = 'Mix W Old: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(mix_seg_weight_old[j]))
        # subplotimg(axs[1][3], mix_seg_weight_old[j], mix_w_old_str, vmin=0, vmax=1)
        
        ### third row
        # src_pred
        src_miou = calculate_iou_gpu(src_seg_pred[j], src_seg_lbl[j])[1]
        src_mpa = calculate_pa_gpu(src_seg_pred[j], src_seg_lbl[j])[1]
        subplotimg(axs[2][0], src_seg_pred[j], f'Src_Pred, mIoU: {src_miou:.1f}, mPA: {src_mpa:.1f}', cmap='cityscapes')
        
        # tar_pred
        pl_miou = calculate_iou_gpu(pseudo_label[j], tar_seg_lbl[j])[1]
        pl_mpa = calculate_pa_gpu(pseudo_label[j], tar_seg_lbl[j])[1]
        subplotimg(axs[2][1], pseudo_label[j], f'Tar_PL, mIoU: {pl_miou:.1f}, mPA: {pl_mpa:.1f}', cmap='cityscapes')
        
        # mix pred
        num_class = len(torch.unique(mix_seg_pred[j]))
        subplotimg(axs[2][2], mix_seg_pred[j], f'Mix_Pred, cls: {num_class}', cmap='cityscapes')
        
        
        # masked pred
        num_class = len(torch.unique(masked_pred[j]))
        subplotimg(axs[2][3], masked_pred[j], f'Masked_Pred, cls: {num_class}', cmap='cityscapes')
        
        # mix weight
        # mix_w_str = 'Mix W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(mixed_seg_weight[j]))
        mix_w_str = 'Mix W'
        subplotimg(axs[2][4], mixed_seg_weight[j], mix_w_str, vmin=0, vmax=1)
        
        
        ### fourth row

        # for debug
        pl_w_str = 'Pseudo W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(pseudo_weight[j]))
        subplotimg(axs[3][0], pseudo_weight[j], pl_w_str, vmin=0, vmax=1)
        
        # mask_target_pseudo_label
        mask_target_pseudo = pseudo_label[j].clone()
        mask_target_pseudo[pseudo_mask[j] == 0] = 255
        mask_target_label = tar_seg_lbl[j].clone()
        mask_target_label[pseudo_mask[j] == 0] = 255

        mpl_miou = calculate_iou_gpu(mask_target_label, mask_target_pseudo)[1]
        mpl_mpa = calculate_pa_gpu(mask_target_label, mask_target_pseudo)[1]
        subplotimg(axs[3][1], mask_target_pseudo, f'Mask Tar_PL, mIoU: {mpl_miou:.1f}, mPA: {mpl_mpa:.1f}', cmap='cityscapes')
        
        # mask_target_label
        subplotimg(axs[3][2], mask_target_label, 'Mask Tar_L', cmap='cityscapes')
        
        # masked pseudo weight
        masked_pl_w_str = 'Masked Pseudo W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in np.unique(masked_pseudo_weight[j]))
        subplotimg(axs[3][3], masked_pseudo_weight[j], masked_pl_w_str, vmin=0, vmax=1)
        
        if gt_rescale is not None:
            subplotimg(axs[3][4], gt_rescale[j], 'Scaled GT', cmap='cityscapes')
            
        ### fifth row
        if dep_tar is not None and dep_mix is not None:
            # plot the depth tar and mix
            subplotimg(axs[4][0], vis_tar_img[j], 'Tar_Img_Depth')
            subplotimg(axs[4][1], dep_tar[j], 'Tar_Depth', cmap='viridis')
            subplotimg(axs[4][2], vis_mix_img[j], 'Mix_Img_Depth')
            subplotimg(axs[4][3], dep_mix[j], 'Mix_Depth', cmap='viridis')

        for ax in axs.flat:
            ax.axis('off')

        plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
        plt.close(fig)
        
        
def save_cls_debug_images(self, 
                    batch_size, means, stds, dataset_class,
                    src_img, tar_img, mix_img, 
                    src_seg_lbl, tar_seg_lbl,
                    src_cls_lbl, tar_cls_lbl, mix_cls_lbl,
                    src_cls_pred, tar_ema_pred, tar_cls_pred, mix_cls_pred,):
    """_summary_

    Args:
        batch_size (_type_): _description_
        means (_type_): _description_
        stds (_type_): _description_
        dataset_class (_type_): _description_
        src_img (_type_): _description_
        tar_img (_type_): _description_
        mix_img (_type_): _description_
        src_seg_lbl (_type_): _description_
        tar_seg_lbl (_type_): _description_
        src_cls_lbl (_type_): _description_
        tar_cls_lbl (_type_): _description_
        mix_cls_lbl (_type_): _description_
        src_cls_pred (_type_): before softmax
        tar_ema_pred (_type_): after softmax
        tar_cls_pred (_type_): before softmax
        mix_cls_pred (_type_): before softmax
    """
    # output dir
    out_dir = os.path.join(self.cfg.respth, 'debug')

    # denorm the images
    vis_src_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
    vis_tar_img = torch.clamp(denorm(tar_img, means, stds), 0, 1)
    vis_mix_img = torch.clamp(denorm(mix_img, means, stds), 0, 1)
    
    if len(dataset_class) < src_cls_pred.shape[1]:
        dataset_class += ['ignore']


    for j in range(batch_size):
        rows, cols = 2, 3
        fig, axs = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )
        ### first row
        # plot the src and tar images in the first column
        subplotimg(axs[0][0], vis_src_img[j], 'Src_Img')
        subplotimg(axs[0][1], vis_tar_img[j], 'Tar_Img')
        subplotimg(axs[0][2], vis_mix_img[j], 'Mix_Img')
        
        ### second row
        # src_seg_lbl
        num_class = len(torch.unique(src_seg_lbl[j]))
        if 255 in torch.unique(src_seg_lbl[j]):
            num_class -= 1
        subplotimg(axs[1][0], src_seg_lbl[j], f'Src_Seg, cls: {num_class}', cmap='cityscapes')
        # tar_seg_lbl
        num_class = len(torch.unique(tar_seg_lbl[j]))
        if 255 in torch.unique(tar_seg_lbl[j]):
            num_class -= 1
        subplotimg(axs[1][1], tar_seg_lbl[j], f'Tar_Seg, cls: {num_class}', cmap='cityscapes')

        for ax in axs.flat:
            ax.axis('off')

        plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
        plt.close(fig)
        
        # 如果有分类预测结果，创建三域的条形图
        if all(x is not None for x in [src_cls_pred, src_cls_lbl, 
                                     tar_ema_pred, tar_cls_pred,tar_cls_lbl,
                                     mix_cls_pred, mix_cls_lbl]):
            # 创建一个包含三个子图的Figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
            
            # 绘制源域的条形图
            visualize_two_bars(
                pred_tensor=torch.softmax(src_cls_pred[j], dim=0),
                gt_tensor=src_cls_lbl[j],
                title="Source Domain Prediction",
                class_names=dataset_class,
                ax=ax1,
                show=False,
                adjust_y_range=True,
            )
            
            # 绘制目标域的条形图
            visualize_three_bars_with_two_preds(
                pred1_tensor=tar_ema_pred[j],
                pred2_tensor=torch.softmax(tar_cls_pred[j], dim=0),
                gt_tensor=tar_cls_lbl[j],
                title="Target Domain Prediction",
                class_names=dataset_class,
                ax=ax2,
                show=False,
                adjust_y_range=True,
            )
            
            # 绘制混合域的条形图
            visualize_two_bars(
                pred_tensor=torch.softmax(mix_cls_pred[j], dim=0),
                gt_tensor=mix_cls_lbl[j],
                title="Mixed Domain Prediction",
                class_names=dataset_class,
                ax=ax3,
                show=False,
                adjust_y_range=True,
            )

            # 调整子图之间的间距
            plt.tight_layout(pad=3.0)
            
            # 保存组合后的条形图
            plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}_cls.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def save_debug_auxhead_images(self, 
                      batch_size, means, stds, dataset_class,
                      img, target_img, mixed_img, 
                      gt_semantic_seg, target_seg, pseudo_label, pseudo_weight, pseudo_mask, mixed_lbl,
                      num_class_choice, mix_masks, mixed_seg_weight, mix_seg_weight_old,
                      srx_aux_pred=None, mix_aux_pred=None,
                      src_multi_cls_lb=None, mix_multi_cls_lb=None,
                      critetia_src_mlcls=None, critetia_mix_mlcls=None):
    # output dir
    out_dir = os.path.join(self.cfg.respth, 'debug')
    os.makedirs(out_dir, exist_ok=True)

    # denorm the images
    vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
    vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
    vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)

    # get the target aux prediction
    if self.with_aux_head:
        with torch.no_grad():
            self.get_model().eval()
            _, tar_aux_pred, _ = self.get_model().forward_train(target_img, return_feat=True)
            tar_multi_cls_lb = self.get_model().auxiliary_head.convert_seg_to_multilabel(target_seg)
        self.get_model().train()

    for j in range(batch_size):
        rows, cols = 2, 7
        if self.debug_fdist_mask is not None or self.debug_gt_rescale is not None:
            cols = 8
        fig, axs = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )
        # plot the src and tar images in the first column
        subplotimg(axs[0][0], vis_img[j], 'Src_Img')
        subplotimg(axs[1][0], vis_trg_img[j], 'Tar_Img')

        # plot the src label and tar pseudo label in the second column
        num_class = len(torch.unique(gt_semantic_seg[j]))
        if 255 in torch.unique(gt_semantic_seg[j]):
            num_class -= 1
        subplotimg(axs[0][1], gt_semantic_seg[j], f'Src_L, cls: {num_class}', cmap='cityscapes')

        pl_miou = calculate_iou_gpu(pseudo_label[j], target_seg[j])[1]
        pl_mpa = calculate_pa_gpu(pseudo_label[j], target_seg[j])[1]
        subplotimg(axs[1][1], pseudo_label[j], f'Tar_PL, mIoU: {pl_miou:.1f}, mPA: {pl_mpa:.1f}', cmap='cityscapes')

        # plot the mixed image and label in the third column
        subplotimg(axs[0][2], vis_mixed_img[j], 'Mix_Img')
        if mixed_lbl is not None:
            num_clas = len(torch.unique(mixed_lbl[j]))
            if 255 in torch.unique(mixed_lbl[j]):
                num_clas -= 1
            subplotimg(axs[1][2], mixed_lbl[j], f'Mix_L, cls: {num_clas}', cmap='cityscapes')
        
        # plot the domain mask and mixed weight in the fourth column
        mix_mask_ratio = torch.sum(mix_masks[j][0][0]) / (mix_masks[j][0][0].shape[0] * mix_masks[j][0][0].shape[1])
        subplotimg(axs[0][3], mix_masks[j][0], f'Domain Mask: {num_class_choice[j]}, {mix_mask_ratio:.2f}', cmap='gray')

        mix_w_str = 'Mix W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(mixed_seg_weight[j]))
        subplotimg(axs[1][3], mixed_seg_weight[j], mix_w_str, vmin=0, vmax=1)
        
        # for debug
        pl_w_str = 'Pseudo W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(pseudo_weight[j]))
        subplotimg(axs[0][4], pseudo_weight[j], pl_w_str, vmin=0, vmax=1)
        # 注释掉有问题的代码，因为 mixed_seg_weight_old 参数不存在
        # mix_w_old_str = 'Mix W Old: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(mixed_seg_weight_old[j]))
        # subplotimg(axs[1][4], mixed_seg_weight_old[j], mix_w_old_str, vmin=0, vmax=1)
        subplotimg(axs[1][4], mixed_seg_weight[j], 'Mix W (copy)', vmin=0, vmax=1)

        # plot the target label and mask (confidence) target pseudo label in the fifth column
        subplotimg(axs[0][5], target_seg[j], 'Tar_L', cmap='cityscapes')
        mask_target_pseudo = pseudo_label[j].clone()
        mask_target_pseudo[pseudo_mask[j] == 0] = 255
        mask_target_label = target_seg[j].clone()
        mask_target_label[pseudo_mask[j] == 0] = 255

        mpl_miou = calculate_iou_gpu(mask_target_label, mask_target_pseudo)[1]
        mpl_mpa = calculate_pa_gpu(mask_target_label, mask_target_pseudo)[1]
        subplotimg(axs[1][5], mask_target_pseudo, f'Mask Tar_PL, mIoU: {mpl_miou:.1f}, mPA: {mpl_mpa:.1f}', cmap='cityscapes')

        # plot the confidence mask target image and label in the sixth column
        mask_target_image = vis_trg_img[j].clone()
        mask_target_image[:, pseudo_mask[j] == 0] = 0
        subplotimg(axs[0][6], mask_target_image, 'Mask Tar_Img')
        subplotimg(axs[1][6], mask_target_label, 'Mask Tar_L', cmap='cityscapes')

        # plot the fdist mask and gt rescale image if available
        if self.debug_fdist_mask is not None:
            subplotimg(axs[0][7], self.debug_fdist_mask[j][0], 'FDist Mask', cmap='gray')
        if self.debug_gt_rescale is not None:
            subplotimg(axs[1][7], self.debug_gt_rescale[j], 'Scaled GT', cmap='cityscapes')

        for ax in axs.flat:
            ax.axis('off')

        plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
        plt.close(fig)

        if self.with_aux_head and self.debug_imgs:
            if self.debug_imgs == 'patch':
                tar_patch_size = self.cfg.model['aux_head']['decoder_config']['tar_patch_size']
                save_multilabel_patch_plots(
                    j, out_dir, self.local_iter, 
                    tar_patch_size, dataset_class,
                    vis_img[j], vis_trg_img[j], 
                    gt_semantic_seg[j], target_seg[j], pseudo_label[j], 
                    srx_aux_pred[j], tar_aux_pred[j], 
                    src_multi_cls_lb[j], tar_multi_cls_lb[j]
                )
            elif self.debug_imgs == 'image':
                save_multilabel_plots(j, out_dir, self.local_iter, dataset_class,
                                      srx_aux_pred[j], mix_aux_pred[j], tar_aux_pred[j],
                                      src_multi_cls_lb[j], mix_multi_cls_lb[j], tar_multi_cls_lb[j],
                                      critetia_src_mlcls, critetia_mix_mlcls, pseudo_weight[j])
            else:
                raise ValueError(f'Invalid debug_imgs type: {self.debug_imgs}')


def save_debug_sup_predictions(out_dir, local_iter, batch_size, vis_img, gt_seg,
                               pred, palette=None):
    """
    保存调试预测结果的可视化图像。

    Args:
        out_dir (str): 输出目录。
        local_iter (int): 当前迭代次数。
        batch_size (int): 批量大小。
        vis_img (torch.Tensor): 可视化图像 (已去归一化)。
        gt_seg (torch.Tensor): Ground truth 分割标签。
        pred (torch.Tensor): 模型预测结果。
    """
    os.makedirs(out_dir, exist_ok=True)
    preds = torch.argmax(pred, dim=1)

    for j in range(batch_size):
        # 创建主图像
        rows, cols = 2, 2
        fig, axs = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )

        # 绘制图像和分割结果
        subplotimg(axs[0][0], vis_img[j], 'Image')
        seg_kwargs = dict(cmap='cityscapes')
        if palette is not None:
            seg_kwargs['palette'] = palette
        subplotimg(axs[0][1], gt_seg[j], 'Seg GT', **seg_kwargs)
        
        miou = calculate_iou_gpu(preds[j], gt_seg[j])[1]
        mpa = calculate_pa_gpu(preds[j], gt_seg[j])[1]
        show_str = f'Pred, mIoU: {miou:.1f}, mPA: {mpa:.1f}'
        subplotimg(axs[1][1], preds[j], show_str, **seg_kwargs)

        # 关闭坐标轴
        for ax in axs.flat:
            ax.axis('off')

        # 保存主图像
        plt.savefig(os.path.join(out_dir, f'{(local_iter + 1):06d}_{j}.png'))
        plt.close()


def _select_main_debug_logits(logits):
    """Select primary segmentation logits from tensor/list/tuple outputs.

    从 tensor/list/tuple 输出中选择主分割 logits；HRDA 训练输出使用 fused logits。
    """
    if isinstance(logits, (list, tuple)):
        return logits[0]
    return logits


def _resize_logits_for_debug(logits, target_shape):
    """Resize logits to the label resolution for debug metrics and display.

    将 logits 对齐到标签分辨率，保证 debug 指标和可视化尺寸一致。
    """
    logits = _select_main_debug_logits(logits)
    if logits.shape[-2:] != target_shape:
        logits = F.interpolate(logits, size=target_shape, mode='bilinear', align_corners=False)
    return logits


def save_debug_sadg_images(out_dir, local_iter, batch_size,
                           src_img, aug_img, gt_seg,
                           src_logits, aug_logits,
                           means, stds, aug_info=None):
    """Save Stage-1 SADG source/augmented prediction comparison images.

    保存 Stage-1 SADG 的原图/增强图预测对比图，用于检查增强效果和预测一致性。
    """
    os.makedirs(out_dir, exist_ok=True)
    aug_info = aug_info or {}

    if gt_seg.dim() == 4 and gt_seg.size(1) == 1:
        gt_seg = gt_seg[:, 0]

    target_shape = gt_seg.shape[-2:]
    src_logits = _resize_logits_for_debug(src_logits, target_shape)
    aug_logits = _resize_logits_for_debug(aug_logits, target_shape)
    src_preds = torch.argmax(src_logits, dim=1)
    aug_preds = torch.argmax(aug_logits, dim=1)

    vis_src_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
    vis_aug_img = torch.clamp(denorm(aug_img, means, stds), 0, 1)

    aug_type = aug_info.get('aug_type', 'unknown')
    severity = float(aug_info.get('severity', 0.0))
    alpha = float(aug_info.get('alpha', 0.0))
    beta = float(aug_info.get('beta', 0.0))

    for j in range(batch_size):
        rows, cols = 2, 3
        fig, axs = plt.subplots(
            rows, cols, figsize=(4 * cols, 4 * rows),
            gridspec_kw={
                'hspace': 0.12,
                'wspace': 0,
                'top': 0.92,
                'bottom': 0,
                'right': 1,
                'left': 0,
            },
        )

        subplotimg(axs[0][0], vis_src_img[j], 'Source Img')
        subplotimg(
            axs[0][1],
            vis_aug_img[j],
            f'SADG {aug_type}\nr={severity:.2f}, a={alpha:.2f}, b={beta:.2f}')
        subplotimg(axs[0][2], gt_seg[j], 'Source GT', cmap='cityscapes')

        src_miou = calculate_iou_gpu(src_preds[j], gt_seg[j])[1]
        src_mpa = calculate_pa_gpu(src_preds[j], gt_seg[j])[1]
        aug_miou = calculate_iou_gpu(aug_preds[j], gt_seg[j])[1]
        aug_mpa = calculate_pa_gpu(aug_preds[j], gt_seg[j])[1]
        subplotimg(
            axs[1][0],
            src_preds[j],
            f'Source Pred\nmIoU: {src_miou:.1f}, mPA: {src_mpa:.1f}',
            cmap='cityscapes')
        subplotimg(
            axs[1][1],
            aug_preds[j],
            f'Aug Pred\nmIoU: {aug_miou:.1f}, mPA: {aug_mpa:.1f}',
            cmap='cityscapes')

        valid = gt_seg[j] != 255
        pred_diff = (src_preds[j] != aug_preds[j]) & valid
        diff_ratio = pred_diff[valid].float().mean().item() if valid.any() else 0.0
        subplotimg(
            axs[1][2],
            pred_diff.cpu().numpy().astype(np.uint8),
            f'Pred Diff\nratio: {diff_ratio:.3f}',
            cmap='gray')

        for ax in axs.flat:
            ax.axis('off')

        plt.savefig(os.path.join(out_dir, f'{(local_iter + 1):06d}_{j}_sadg.png'))
        plt.close(fig)


def save_weather_aligned_mix_debug_images(out_dir, local_iter, batch_size,
                                          src_img, weather_src_img,
                                          tar_img, mix_img, mix_masks,
                                          means, stds,
                                          weather_types=None,
                                          aug_types=None,
                                          severities=None):
    """Save source-weather alignment panels used before ClassMix."""
    os.makedirs(out_dir, exist_ok=True)
    weather_types = weather_types or ['unknown'] * batch_size
    aug_types = aug_types or ['none'] * batch_size
    severities = severities or [0.0] * batch_size

    vis_src_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
    vis_weather_src_img = torch.clamp(denorm(weather_src_img, means, stds), 0, 1)
    vis_tar_img = torch.clamp(denorm(tar_img, means, stds), 0, 1)
    vis_mix_img = torch.clamp(denorm(mix_img, means, stds), 0, 1)

    for j in range(batch_size):
        fig, axs = plt.subplots(
            1, 5, figsize=(20, 4),
            gridspec_kw={
                'hspace': 0.12,
                'wspace': 0,
                'top': 0.92,
                'bottom': 0,
                'right': 1,
                'left': 0,
            },
        )
        weather_type = weather_types[j] if j < len(weather_types) else 'unknown'
        aug_type = aug_types[j] if j < len(aug_types) else 'none'
        severity = float(severities[j]) if j < len(severities) else 0.0

        subplotimg(axs[0], vis_src_img[j], 'Src Original')
        subplotimg(
            axs[1],
            vis_weather_src_img[j],
            f'Src Weather\n{aug_type}, r={severity:.2f}')
        subplotimg(axs[2], vis_tar_img[j], f'Target\n{weather_type}')
        subplotimg(axs[3], vis_mix_img[j], 'Weather Mix')
        subplotimg(axs[4], mix_masks[j][0], 'Mix Mask', cmap='gray')

        for ax in axs.flat:
            ax.axis('off')

        plt.savefig(os.path.join(out_dir, f'{(local_iter + 1):06d}_{j}_weather_mix.png'))
        plt.close(fig)


def save_debug_cross_view_distillation(out_dir, local_iter, batch_size, 
                                       weak_img, strong_img, 
                                       weak_lb, strong_lb,
                                       teacher_weak_pred, teacher_strong_pred, teacher_weak_pred_aligned, teacher_strong_pred_aligned,
                                       student_weak_pred, student_strong_pred,
                                       means, stds):
    """
    保存交叉视图对称蒸馏的调试图像。
    
    Args:
        out_dir (str): 输出目录
        local_iter (int): 当前迭代次数
        batch_size (int): 批量大小
        weak_img (torch.Tensor): 弱增强图像 [B, 3, H, W]
        strong_img (torch.Tensor): 强增强图像 [B, 3, H, W]
        weak_lb (torch.Tensor): 弱增强真值标签 [B, H, W]
        strong_lb (torch.Tensor): 强增强真值标签 [B, H, W]
        teacher_weak_pred (torch.Tensor): 教师模型对弱增强的预测 [B, C, H, W]
        teacher_strong_pred (torch.Tensor): 教师模型对强增强的预测 [B, C, H, W]
        teacher_weak_pred_aligned (torch.Tensor): 教师模型对弱增强的预测（对齐）[B, C, H, W]
        teacher_strong_pred_aligned (torch.Tensor): 教师模型对强增强的预测（对齐）[B, C, H, W]
        student_weak_pred (torch.Tensor): 学生模型对弱增强的预测 [B, C, H, W]
        student_strong_pred (torch.Tensor): 学生模型对强增强的预测 [B, C, H, W]
        means (tensor): 均值用于反归一化
        stds (tensor): 标准差用于反归一化
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 转换预测为类别
    teacher_weak_preds = torch.argmax(teacher_weak_pred, dim=1)
    teacher_strong_preds = torch.argmax(teacher_strong_pred, dim=1)
    teacher_weak_preds_aligned = torch.argmax(teacher_weak_pred_aligned, dim=1)
    teacher_strong_preds_aligned = torch.argmax(teacher_strong_pred_aligned, dim=1)
    student_weak_preds = torch.argmax(student_weak_pred, dim=1)
    student_strong_preds = torch.argmax(student_strong_pred, dim=1)
    
    # 反归一化图像
    vis_weak_img = torch.clamp(denorm(weak_img, means, stds), 0, 1)
    vis_strong_img = torch.clamp(denorm(strong_img, means, stds), 0, 1)
    
    for j in range(batch_size):
        # 创建 2x5 的子图布局
        # 行: 弱增强、强增强
        # 列: 图像、真值、教师预测、学生预测、教师预测（对齐）
        rows, cols = 2, 5
        fig, axs = plt.subplots(
            rows, cols, figsize=(5 * cols, 5 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )
        
        # 第一行：弱增强图像和真值
        subplotimg(axs[0][0], vis_weak_img[j], 'Weak Augmentation')
        subplotimg(axs[0][1], weak_lb[j], 'Weak GT', cmap='cityscapes')
        
        # 计算教师模型在弱增强上的性能
        teacher_weak_miou = calculate_iou_gpu(teacher_weak_preds[j], weak_lb[j])[1]
        teacher_weak_mpa = calculate_pa_gpu(teacher_weak_preds[j], weak_lb[j])[1]
        student_weak_miou = calculate_iou_gpu(student_weak_preds[j], weak_lb[j])[1]
        student_weak_mpa = calculate_pa_gpu(student_weak_preds[j], weak_lb[j])[1]
        subplotimg(axs[0][2], teacher_weak_preds[j], 
                  f'Teacher Weak Pred\nmIoU: {teacher_weak_miou:.1f}, mPA: {teacher_weak_mpa:.1f}',
                  cmap='cityscapes')
        subplotimg(axs[0][3], student_weak_preds[j],
                  f'Student Weak Pred\nmIoU: {student_weak_miou:.1f}, mPA: {student_weak_mpa:.1f}',
                  cmap='cityscapes')
        subplotimg(axs[0][4], teacher_weak_preds_aligned[j],
                  f'Teacher Weak Pred Aligned\nmIoU: {teacher_weak_miou:.1f}, mPA: {teacher_weak_mpa:.1f}',
                  cmap='cityscapes')

        # 第二行：强增强图像和真值
        subplotimg(axs[1][0], vis_strong_img[j], 'Strong Augmentation')
        subplotimg(axs[1][1], strong_lb[j], 'Strong GT', cmap='cityscapes')
        
        # 计算教师模型在强增强上的性能
        teacher_strong_miou = calculate_iou_gpu(teacher_strong_preds[j], strong_lb[j])[1]
        teacher_strong_mpa = calculate_pa_gpu(teacher_strong_preds[j], strong_lb[j])[1]
        student_strong_miou = calculate_iou_gpu(student_strong_preds[j], strong_lb[j])[1]
        student_strong_mpa = calculate_pa_gpu(student_strong_preds[j], strong_lb[j])[1]
        subplotimg(axs[1][2], teacher_strong_preds[j],
                  f'Teacher Strong Pred\nmIoU: {teacher_strong_miou:.1f}, mPA: {teacher_strong_mpa:.1f}',
                  cmap='cityscapes')
        subplotimg(axs[1][3], student_strong_preds[j],
                  f'Student Strong Pred\nmIoU: {student_strong_miou:.1f}, mPA: {student_strong_mpa:.1f}',
                  cmap='cityscapes')
        subplotimg(axs[1][4], teacher_strong_preds_aligned[j],
                  f'Teacher Strong Pred Aligned\nmIoU: {teacher_strong_miou:.1f}, mPA: {teacher_strong_mpa:.1f}',
                  cmap='cityscapes')
        
        # 关闭所有子图的坐标轴
        for ax in axs.flat:
            ax.axis('off')
        
        # 保存图像
        plt.savefig(os.path.join(out_dir, f'{(local_iter + 1):06d}_{j}_distill.png'))
        plt.close()

def save_debug_cls_predictions(out_dir, local_iter, batch_size, vis_img, gt_seg, cls_pred=None, cls_lb=None, 
                           critetia_cls=None, dataset_class=None, debug_imgs=False, adjust_y_range=False):
    """
    保存调试预测结果的可视化图像。

    Args:
        out_dir (str): 输出目录。
        local_iter (int): 当前迭代次数。
        batch_size (int): 批量大小。
        vis_img (torch.Tensor): 可视化图像 (已去归一化)。
        gt_seg (torch.Tensor): Ground truth 分割标签。
        cls_pred (torch.Tensor, optional): 辅助头预测结果。
        cls_lb (torch.Tensor, optional): 多标签 ground truth
        critetia_cls (callable, optional): 多标签损失函数
        dataset_class (list, optional): 数据集类别名称
        debug_imgs (bool, optional): 是否启用调试图像保存。
    """
    # os.makedirs(out_dir, exist_ok=True)

    for j in range(batch_size):
        # 创建主图像
        rows, cols = 1, 2
        fig, axs = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )

        # 绘制图像和分割结果
        subplotimg(axs[0], vis_img[j], 'Image')
        subplotimg(axs[1], gt_seg[j], 'Seg GT', cmap='cityscapes')

        # 关闭坐标轴
        for ax in axs.flat:
            ax.axis('off')

        # 保存主图像
        plt.savefig(os.path.join(out_dir, f'{(local_iter + 1):06d}_{j}.png'))
        plt.close()

        # 如果启用调试图像，绘制多标签预测结果
        if cls_pred is not None and cls_lb is not None:
            with torch.no_grad():
                mlcls_loss_j = critetia_cls(cls_pred[j].unsqueeze(0), gt_seg[j].unsqueeze(0)) if critetia_cls else 0.0
                save_path = os.path.join(out_dir, f'{(local_iter + 1):06d}_{j}_cls.pdf')
                fig, ax = plt.subplots(figsize=(9, 6))
                visualize_multilabel_bars(
                    pred_tensor=torch.softmax(cls_pred[j], dim=0),
                    gt_tensor=cls_lb[j],
                    title=f"Cls Pred, loss: {mlcls_loss_j.item():.3f}",
                    class_names=dataset_class,
                    ax=ax,
                    show=False,
                    adjust_y_range=adjust_y_range,
                )
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)


def save_debug_sup_mlcls_predictions(out_dir, local_iter, batch_size, vis_img, gt_seg, pred, aux_pred=None, multi_cls_lb=None, 
                           criteria_sup=None, critetia_mlcls=None, dataset_class=None, debug_imgs=False):
    """
    保存调试预测结果的可视化图像。

    Args:
        out_dir (str): 输出目录。
        local_iter (int): 当前迭代次数。
        batch_size (int): 批量大小。
        vis_img (torch.Tensor): 可视化图像 (已去归一化)。
        gt_seg (torch.Tensor): Ground truth 分割标签。
        pred (torch.Tensor): 模型预测结果。
        aux_pred (torch.Tensor, optional): 辅助头预测结果。
        multi_cls_lb (torch.Tensor, optional): 多标签 ground truth。
        criteria_sup (callable, optional): 分割损失函数。
        critetia_mlcls (callable, optional): 多标签损失函数。
        dataset_class (list, optional): 数据集类别名称。
        debug_imgs (bool, optional): 是否启用调试图像保存。
    """
    os.makedirs(out_dir, exist_ok=True)
    preds = torch.argmax(pred, dim=1)

    for j in range(batch_size):
        # 创建主图像
        rows, cols = 2, 2
        fig, axs = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )

        # 绘制图像和分割结果
        subplotimg(axs[0][0], vis_img[j], 'Image')
        subplotimg(axs[0][1], gt_seg[j], 'Seg GT', cmap='cityscapes')
        
        miou = calculate_iou_gpu(preds[j], gt_seg[j])[1]
        mpa = calculate_pa_gpu(preds[j], gt_seg[j])[1]
        show_str = f'Pred, mIoU: {miou:.1f}, mPA: {mpa:.1f}'
        subplotimg(axs[1][0], preds[j], show_str, cmap='cityscapes')

        # 关闭坐标轴
        for ax in axs.flat:
            ax.axis('off')

        # 保存主图像
        plt.savefig(os.path.join(out_dir, f'{(local_iter + 1):06d}_{j}.png'))
        plt.close()

        # 如果启用调试图像，绘制多标签预测结果
        if debug_imgs and aux_pred is not None and multi_cls_lb is not None:
            with torch.no_grad():
                mlcls_loss_j = critetia_mlcls(aux_pred[j], multi_cls_lb[j]) if critetia_mlcls else 0.0
                save_path = os.path.join(out_dir, f'{(local_iter + 1):06d}_{j}_mlcls.pdf')
                fig, ax = plt.subplots(figsize=(9, 6))
                visualize_multilabel_bars(
                    pred_tensor=torch.sigmoid(aux_pred[j]),
                    gt_tensor=multi_cls_lb[j],
                    title=f"MultiLabel Pred, loss: {mlcls_loss_j.item():.3f}",
                    class_names=dataset_class,
                    ax=ax,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)


def visualize_depth(ax, depth_map, title, gt_range=None, pred_range=None):
    """增强的深度图可视化函数
    
    Args:
        ax: matplotlib axes对象
        depth_map: 深度图数据
        title: 图像标题
        gt_range: 真值的(min, max)范围，可选
        pred_range: 预测值的(min, max)范围，可选
    """
    # 获取深度图的实际范围
    actual_min = float(np.min(depth_map))
    actual_max = float(np.max(depth_map))
    
    # 使用真值范围进行主要可视化
    if gt_range is not None:
        gt_min, gt_max = gt_range
        im = ax.imshow(depth_map, vmin=gt_min, vmax=gt_max, cmap='Spectral_r')
        
        # # 添加颜色条
        # plt.colorbar(im, ax=ax)
        
        # 在标题中显示范围信息
        range_info = f"[{gt_min:.0f}, {gt_max:.0f}]"
        range_info += f",[{actual_min:.0f}, {actual_max:.0f}]"
        title += range_info
        
        # 标记超出范围的区域
        if actual_min < gt_min or actual_max > gt_max:
            mask_under = depth_map < gt_min
            mask_over = depth_map > gt_max
            
            # 使用半透明的红色和蓝色标记超出范围的区域
            if mask_under.any():
                ax.imshow(mask_under, cmap='Blues', alpha=0.3)
            if mask_over.any():
                ax.imshow(mask_over, cmap='Reds', alpha=0.3)
    else:
        # 如果没有真值范围，使用实际范围
        im = ax.imshow(depth_map, vmin=actual_min, vmax=actual_max, cmap='Spectral_r')
        plt.colorbar(im, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')

cmap_depth = matplotlib.colormaps.get_cmap('Spectral_r') 
def save_debug_depth_images(self, 
                    batch_size, means, stds, dataset_class,
                    src_img, tar_img, mix_img, 
                    src_seg, tar_seg, pseudo_label, pseudo_weight, pseudo_mask, mix_lbl, src_dep, tar_dep, mix_dep,
                    num_class_choice, mix_masks, mixed_seg_weight, mixed_seg_weight_old,
                    src_seg_pred, src_seg_init_pred, src_dep_pred, src_dep_init_pred, tar_dep_pred, tar_dep_init_pred,
                    mix_seg_pred, mix_seg_init_pred, mix_dep_pred, mix_dep_init_pred,):
    # output dir
    out_dir = os.path.join(self.cfg.respth, 'debug')
    os.makedirs(out_dir, exist_ok=True)

    # denorm the images
    vis_src_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
    vis_tar_img = torch.clamp(denorm(tar_img, means, stds), 0, 1)
    vis_mix_img = torch.clamp(denorm(mix_img, means, stds), 0, 1)

    # get the target aux prediction
    if self.with_aux_head:
        with torch.no_grad():
            self.get_model().eval()
            _, tar_aux_pred, _ = self.get_model().forward_train(tar_img, return_feat=True)
            tar_multi_cls_lb = self.get_model().auxiliary_head.convert_seg_to_multilabel(tar_seg)
        self.get_model().train()

    for j in range(batch_size):
        rows, cols = 7, 4
        fig, axs = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )
        ### first row
        # plot the src and tar images in the first column
        subplotimg(axs[0][0], vis_src_img[j], 'Src_Img')
        subplotimg(axs[0][1], vis_tar_img[j], 'Tar_Img')
        subplotimg(axs[0][2], vis_mix_img[j], 'Mix_Img')
        
        mix_mask_ratio = torch.sum(mix_masks[j][0][0]) / (mix_masks[j][0][0].shape[0] * mix_masks[j][0][0].shape[1])
        subplotimg(axs[0][3], mix_masks[j][0], f'Mix_Mask: {num_class_choice[j]}, {mix_mask_ratio:.2f}', cmap='gray')

        
        ### second row
        # src_seg
        num_class = len(torch.unique(src_seg[j]))
        if 255 in torch.unique(src_seg[j]):
            num_class -= 1
        subplotimg(axs[1][0], src_seg[j], f'Src_Seg, cls: {num_class}', cmap='cityscapes')
        # tar_seg
        num_class = len(torch.unique(tar_seg[j]))
        if 255 in torch.unique(tar_seg[j]):
            num_class -= 1
        subplotimg(axs[1][1], tar_seg[j], f'Tar_Seg, cls: {num_class}', cmap='cityscapes')
        # mix_pseudo_label
        if mix_lbl is not None:
            num_clas = len(torch.unique(mix_lbl[j]))
            if 255 in torch.unique(mix_lbl[j]):
                num_clas -= 1
            subplotimg(axs[1][2], mix_lbl[j], f'Mix_L, cls: {num_clas}', cmap='cityscapes')
        # mix mask old
        mix_w_old_str = 'Mix W Old: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(mixed_seg_weight_old[j]))
        subplotimg(axs[1][3], mixed_seg_weight_old[j], mix_w_old_str, vmin=0, vmax=1)
        
        ### third row
        # src_pred
        src_miou = calculate_iou_gpu(src_seg_pred[j], src_seg[j])[1]
        src_mpa = calculate_pa_gpu(src_seg_pred[j], src_seg[j])[1]
        subplotimg(axs[2][0], src_seg_pred[j], f'Src_Pred, mIoU: {src_miou:.1f}, mPA: {src_mpa:.1f}', cmap='cityscapes')
        
        # tar_pred
        pl_miou = calculate_iou_gpu(pseudo_label[j], tar_seg[j])[1]
        pl_mpa = calculate_pa_gpu(pseudo_label[j], tar_seg[j])[1]
        subplotimg(axs[2][1], pseudo_label[j], f'Tar_PL, mIoU: {pl_miou:.1f}, mPA: {pl_mpa:.1f}', cmap='cityscapes')
        
        # mix pred
        num_class = len(torch.unique(mix_seg_pred[j]))
        subplotimg(axs[2][2], mix_seg_pred[j], f'Mix_Pred, cls: {num_class}', cmap='cityscapes')
        
        # mix weight
        # mix_w_str = 'Mix W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(mixed_seg_weight[j]))
        mix_w_str = 'Mix W'
        subplotimg(axs[2][3], mixed_seg_weight[j], mix_w_str, vmin=0, vmax=1)
        
        
        ### fourth row
        # src_init_pred
        num_class = len(torch.unique(src_seg_init_pred[j]))
        src_init_miou = calculate_iou_gpu(src_seg_init_pred[j], src_seg[j])[1]
        src_init_mpa = calculate_pa_gpu(src_seg_init_pred[j], src_seg[j])[1]
        subplotimg(axs[3][0], src_seg_init_pred[j], f'Src_Ini_Pred, mIoU: {src_init_miou:.1f}, mPA: {src_init_mpa:.1f}', cmap='cityscapes')
        
        # mask_target_pseudo_label
        mask_target_pseudo = pseudo_label[j].clone()
        mask_target_pseudo[pseudo_mask[j] == 0] = 255
        mask_target_label = tar_seg[j].clone()
        mask_target_label[pseudo_mask[j] == 0] = 255

        mpl_miou = calculate_iou_gpu(mask_target_label, mask_target_pseudo)[1]
        mpl_mpa = calculate_pa_gpu(mask_target_label, mask_target_pseudo)[1]
        subplotimg(axs[3][1], mask_target_pseudo, f'Mask Tar_PL, mIoU: {mpl_miou:.1f}, mPA: {mpl_mpa:.1f}', cmap='cityscapes')
        
        # mix init pred
        num_class = len(torch.unique(mix_seg_init_pred[j]))
        subplotimg(axs[3][2], mix_seg_init_pred[j], f'Mix_Ini_Pred, cls: {num_class}', cmap='cityscapes')
        
        # plot the confidence mask target image and label in the sixth column
        # mask_target_image = vis_tar_img[j].clone()
        # mask_target_image[:, pseudo_mask[j] == 0] = 0
        # subplotimg(axs[0][6], mask_target_image, 'Mask Tar_Img')
        # mask_target_label
        subplotimg(axs[3][3], mask_target_label, 'Mask Tar_L', cmap='cityscapes')

        
        ### fifth row
        # src dep
        num_dep, max_src_dep, min_src_dep = len(np.unique(src_dep[j])), int(np.max(src_dep[j])), int(np.min(src_dep[j]))
        # str_show = f'Src_Dep: {num_dep} {min_src_dep}-{max_src_dep}'
        str_show = 'Src_Dep:'
        visualize_depth(axs[4][0], src_dep[j], str_show, gt_range=(min_src_dep, max_src_dep))
        
        # tar dep
        num_dep, max_tar_dep, min_tar_dep = len(np.unique(tar_dep[j])), int(np.max(tar_dep[j])), int(np.min(tar_dep[j]))
        # str_show = f'Tar_Dep: {num_dep} {min_tar_dep}-{max_tar_dep}'
        str_show = 'Tar_Dep:'
        visualize_depth(axs[4][1], tar_dep[j], str_show, gt_range=(min_tar_dep, max_tar_dep))
        
        # mix dep
        num_dep, max_mix_dep, min_mix_dep = len(np.unique(mix_dep[j])), int(np.max(mix_dep[j])), int(np.min(mix_dep[j]))
        # str_show = f'Mix_Dep: {num_dep} {min_mix_dep}-{max_mix_dep}'
        str_show = 'Mix_Dep:'
        visualize_depth(axs[4][2], mix_dep[j], str_show, gt_range=(min_mix_dep, max_mix_dep))
        
        ### sixth row
        # num_dep = len(np.unique(src_dep_pred[j]))
        # src_dep_pred
        max_dep, min_dep = int(np.max(src_dep_pred[j])), int(np.min(src_dep_pred[j]))
        # str_show = f'Src_Dep_Pred,{max_dep}-{min_dep}'
        str_show = 'Src_Dep_Pred:'
        visualize_depth(axs[5][0], src_dep_pred[j], str_show, gt_range=(min_src_dep, max_src_dep))
        # subplotimg(axs[5][0], src_dep_pred[j], str_show, vmin=min_src_dep, vmax=max_src_dep)
        
        # tar_dep_pred
        max_dep, min_dep = int(np.max(tar_dep_pred[j])), int(np.min(tar_dep_pred[j]))
        # str_show = f'Tar_Dep_Pred,{max_dep}-{min_dep}'
        str_show = 'Tar_Dep_Pred:'
        visualize_depth(axs[5][1], tar_dep_pred[j], str_show, gt_range=(min_tar_dep, max_tar_dep))
        # subplotimg(axs[5][1], tar_dep_pred[j], str_show, vmin=min_tar_dep, vmax=max_tar_dep)
        
        # mix_dep_pred
        max_dep, min_dep = int(np.max(mix_dep_pred[j])), int(np.min(mix_dep_pred[j]))
        # str_show = f'Mix_Dep_Pred,{max_dep}-{min_dep}'
        str_show = 'Mix_Dep_Pred:'
        visualize_depth(axs[5][2], mix_dep_pred[j], str_show, gt_range=(min_mix_dep, max_mix_dep))
        # subplotimg(axs[5][2], mix_dep_pred[j], str_show, vmin=min_mix_dep, vmax=max_mix_dep)
        
        # for debug
        pl_w_str = 'Pseudo W: ' + ', '.join(f'{w:.2f}' if w > 0 else '0' for w in torch.unique(pseudo_weight[j]))
        subplotimg(axs[5][3], pseudo_weight[j], pl_w_str, vmin=0, vmax=1)
        
        ### seventh row
        # Src_Dep_Ini_Pred
        max_dep, min_dep = int(np.max(src_dep_init_pred[j])), int(np.min(src_dep_init_pred[j]))
        # str_show = f'Src_Dep_Ini_Pred,{min_dep}-{max_dep}'
        str_show = 'Src_Dep_Ini_Pred:'
        visualize_depth(axs[6][0], src_dep_init_pred[j], str_show, gt_range=(min_src_dep, max_src_dep))
        # subplotimg(axs[6][0], src_dep_init_pred[j], str_show, vim=min_src_dep, vmax=max_src_dep)
        
        # Tar_Dep_Ini_Pred
        max_dep, min_dep = int(np.max(tar_dep_init_pred[j])), int(np.min(tar_dep_init_pred[j]))
        # str_show = f'Tar_Dep_Ini_Pred,{min_dep}-{max_dep}'
        str_show = 'Tar_Dep_Ini_Pred:'
        visualize_depth(axs[6][1], tar_dep_init_pred[j], str_show, gt_range=(min_tar_dep, max_tar_dep))
        # subplotimg(axs[6][1], tar_dep_init_pred[j], str_show, vmin=min_tar_dep, vmax=max_tar_dep)
        
        # Mix_Dep_Ini_Pred
        max_dep, min_dep = int(np.max(mix_dep_init_pred[j])), int(np.min(mix_dep_init_pred[j]))
        # str_show = f'Mix_Dep_Ini_Pred,{min_dep}-{max_dep}'
        str_show = 'Mix_Dep_Ini_Pred:'
        visualize_depth(axs[6][2], mix_dep_init_pred[j], str_show, gt_range=(min_mix_dep, max_mix_dep))
        # subplotimg(axs[6][2], mix_dep_init_pred[j], str_show, vmin=min_mix_dep, vmax=max_mix_dep)
        
        for ax in axs.flat:
            ax.axis('off')

        plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
        plt.close(fig)

def save_debug_sup_dep_images(self, 
                    batch_size, means, stds, img, gt_seg, gt_depth, seg_pred, dep_pred, seg_init_pred, dep_init_pred):

    out_dir = os.path.join(self.respth, 'debug')
    os.makedirs(out_dir, exist_ok=True)
    
    # denorm the images
    vis_img = torch.clamp(denorm(img, means, stds), 0, 1)

    for j in range(batch_size):
        rows, cols = 2, 4
        fig, axs = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0}
        )
        ### first row
        # plot the src and tar images in the first column
        subplotimg(axs[0][0], vis_img[j], 'Img')

        num_class = len(torch.unique(gt_seg[j]))
        if 255 in torch.unique(gt_seg[j]):
            num_class -= 1
        subplotimg(axs[0][1], gt_seg[j], f'GT_Seg, cls: {num_class}', cmap='cityscapes')
        
        num_dep, max_dep, min_dep = len(np.unique(gt_depth[j])), int(np.max(gt_depth[j])), int(np.min(gt_depth[j]))
        str_show = f'GT_Dep: {num_dep}, {min_dep}-{max_dep}'
        subplotimg(axs[1][1], gt_depth[j], str_show, vmin=min_dep, vmax=max_dep)
        
        num_class = len(torch.unique(seg_pred[j]))
        if 255 in torch.unique(seg_pred[j]):
            num_class -= 1
        seg_pred_miou = calculate_iou_gpu(gt_seg[j], seg_pred[j])[1]
        subplotimg(axs[0][2], seg_pred[j], f'Seg_Pred, cls: {num_class}, mIoU: {seg_pred_miou:.1f}', cmap='cityscapes')
        
        ### second row
        metrics_np, mean_berhu_np = calculate_iberhu_numpy(dep_pred[j], gt_depth[j])
        num_dep, max_dep, min_dep = len(np.unique(dep_pred[j])), int(np.max(dep_pred[j])), int(np.min(dep_pred[j]))
        str_show = f'Dep_Pred: {min_dep}-{max_dep}, berhu: {mean_berhu_np:.1f}'
        subplotimg(axs[1][2], dep_pred[j], str_show, vmin=min_dep, vmax=max_dep)
        
        num_class = len(torch.unique(seg_init_pred[j]))
        if 255 in torch.unique(seg_init_pred[j]):
            num_class -= 1
        seg_init_pred_miou = calculate_iou_gpu(gt_seg[j], seg_init_pred[j])[1]
        subplotimg(axs[0][3], seg_init_pred[j], f'Seg_Init_Pred, cls: {num_class}, mIoU: {seg_init_pred_miou:.1f}', cmap='cityscapes')
        
        metrics_np, mean_berhu_np = calculate_iberhu_numpy(dep_init_pred[j], gt_depth[j])
        num_dep, max_dep, min_dep = len(np.unique(dep_init_pred[j])), int(np.max(dep_init_pred[j])), int(np.min(dep_init_pred[j]))
        str_show = f'Dep_Init_Pred: {min_dep}-{max_dep}, berhu: {mean_berhu_np:.1f}'
        subplotimg(axs[1][3], dep_init_pred[j], str_show, vmin=min_dep, vmax=max_dep)
        
        for ax in axs.flat:
            ax.axis('off')

        plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
        plt.close(fig)

def save_debug_images_s(seg_debug, batch_size, local_iter, train_cfg, means, stds):
    """保存调试图片"""
    out_dir = os.path.join(train_cfg['work_dir'], 'debug')
    os.makedirs(out_dir, exist_ok=True)

    for j in range(batch_size):
        cols = len(seg_debug)
        rows = max(len(v) if isinstance(v, dict) else 0 for v in seg_debug.values())

        fig, axs = plt.subplots(
            rows,
            cols,
            figsize=(5 * cols, 5 * rows),
            gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0},
            squeeze=False,
        )

        try:
            for k1, (n1, outs) in enumerate(seg_debug.items()):
                if not isinstance(outs, dict):  # 确保 `outs` 是字典
                    continue
                for k2, (n2, out) in enumerate(outs.items()):
                    subplotimg(axs[k2][k1], **prepare_debug_out(f'{n1} {n2}', out[j], means, stds))
            
            for ax in axs.flat:
                ax.axis('off')

            plt.savefig(os.path.join(out_dir, f'{(local_iter + 1):06d}_{j}_s.png'))
            plt.close()
        except Exception as e:
            print(f"Error saving debug image at iter {local_iter}, batch {j}: {e}")

    del seg_debug  # 释放内存

'''
def visualize_multilabel_bars(ax, pred_tensor, gt_tensor, title, class_names=None, width=0.3):
    """
    优化后的可视化函数，解决标签溢出问题
    """
    with torch.no_grad():
        pred = pred_tensor.cpu().numpy().squeeze()
        gt = gt_tensor.cpu().numpy().squeeze()

    assert pred.ndim == 1 and gt.ndim == 1
    num_classes = len(pred)
    
    x = np.arange(num_classes)
    class_names = class_names or [f'Class {i}' for i in range(num_classes)]
    
    # 绘制柱状图（保持不变）
    ax.bar(x - width/2, pred, width, label='Pred', color='tab:blue', alpha=0.7)
    ax.bar(x + width/2, gt, width, label='GT', color='tab:orange', alpha=0.7)

    # 优化标签设置
    ax.set_xticks(x)
    xtick_labels = ax.set_xticklabels(
        class_names,
        rotation=90,
        ha='right',
        va='top',  # 对齐方式调整为顶部对齐
        fontsize=8,  # 减小字体大小
        rotation_mode='anchor'  # 围绕锚点旋转
    )
    
    # 设置标签与坐标轴的间距
    ax.tick_params(axis='x', which='major', pad=2)  # 减小标签与柱子的间距
    
    # 调整坐标轴范围
    ax.set_ylim(0, 1.2)  # 为顶部标签腾出空间
    ax.set_xlim(-0.5, num_classes-0.5)  # 防止左右溢出
    
    # 设置标题和图例
    ax.set_title(title, pad=15)  # 增加标题与图的间距
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.95))  # 提升图例位置
    
    # 数值标签设置
    for i in x:
        ax.text(i - width/2, pred[i] + 0.02, f'{pred[i]:.2f}', 
                ha='center', va='bottom', fontsize=6)
        ax.text(i + width/2, gt[i] + 0.02, f'{gt[i]:.0f}',
                ha='center', va='bottom', fontsize=6)

    # 智能调整布局（关键修改）
    plt.tight_layout(
        rect=[0.05, 0.05, 0.95, 0.95],  # 留出5%的边界空间
        pad=0.5,
        h_pad=0.5,
        w_pad=0.5
    )
    
    # 添加自动缩放功能（可选）
    if num_classes > 20:
        for label in xtick_labels:
            label.set_fontsize(6)  # 类别过多时进一步缩小字体
        ax.tick_params(axis='x', pad=1)
'''

        
        
if __name__ == '__main__':
    # # 模拟数据
    # num_classes = 19
    # pred = torch.sigmoid(torch.randn(1, num_classes).cuda())  # [0,1]
    # gt = torch.randint(0, 2, (1, num_classes)).float().cuda() # 0/1

    # # 创建画布
    # # fig, ax = plt.subplots(figsize=(10, 4))

    # # 调用可视化函数
    # class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',  'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
    #                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    # visualize_multilabel_bars(
    #     pred,
    #     gt,
    #     title="Multi-class Prediction Results",
    #     class_names=class_names,
    #     fig_size=(12, 6),
    #     save_path='tmp.pdf',
    #     width=0.4,
    #     show=False
    # )
    
    num_classes = 19
    pred_src = torch.rand(num_classes).cuda() 
    gt_src = torch.randint(0, 2, (num_classes,)).cuda().float()
    pred_tar = torch.rand(num_classes).cuda() 
    gt_tar = torch.randint(0, 2, (num_classes,)).cuda().float()
    
    class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',  'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
                   'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    save_path = "multilabel_bars_combined.pdf"

    # 创建包含两个子图的 Figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 绘制第一个子图（SRC）
    ax1 = axes[0]
    title1 = "MultiLabel Pred SRC, loss: 0.123"
    visualize_multilabel_bars(
        pred_src,
        gt_src,
        title1,
        class_names=class_names,
        ax=ax1,
        show=False
    )

    # 绘制第二个子图（TRG）
    ax2 = axes[1]
    title2 = "MultiLabel Pred TRG, loss: 0.456"
    visualize_multilabel_bars(
        pred_tar,
        gt_tar,
        title2,
        class_names=class_names,
        ax=ax2,
        show=False
    )

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
