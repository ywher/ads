import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().float()
        return value.numpy()
    return np.asarray(value)


def _prepare_image(image_tensor):
    image = _to_numpy(image_tensor)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    image = image.squeeze()
    image_min = float(np.nanmin(image))
    image_max = float(np.nanmax(image))
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    return np.clip(image, 0.0, 1.0)


def _prepare_mask(mask_tensor):
    mask = _to_numpy(mask_tensor)
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 3:
        mask = mask[0]
    return mask


def _prepare_vector(vector_tensor):
    vector = _to_numpy(vector_tensor).reshape(-1)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_multilabel_vector(vector_tensor):
    values = _to_numpy(vector_tensor)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim == 1:
        return values
    values = values.reshape(-1, values.shape[-1])
    return values.mean(axis=0)


def _patch_scores_to_map(scores, patch_size, image_shape):
    scores = _to_numpy(scores)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if scores.ndim > 1:
        scores = scores.reshape(-1, scores.shape[-1]).mean(axis=1)
    else:
        scores = scores.reshape(-1)
    height, width = image_shape[:2]
    if len(scores) == 0:
        return np.zeros((height, width), dtype=np.float32)

    grid_h = max(1, int(np.ceil(height / patch_size[0])))
    grid_w = max(1, int(np.ceil(width / patch_size[1])))
    n_patches = grid_h * grid_w

    if len(scores) < n_patches:
        padded = np.zeros((n_patches,), dtype=np.float32)
        padded[:len(scores)] = scores
        scores = padded
    elif len(scores) > n_patches:
        scores = scores[:n_patches]

    score_grid = scores.reshape(grid_h, grid_w)
    score_tensor = torch.from_numpy(score_grid)[None, None].float()
    score_tensor = F.interpolate(
        score_tensor,
        size=(height, width),
        mode='nearest',
    )
    return score_tensor[0, 0].numpy()


def _get_class_names(dataset_class, num_classes):
    if dataset_class is None:
        return [str(i) for i in range(num_classes)]
    classes = getattr(dataset_class, 'CLASSES', None)
    if classes is None:
        classes = getattr(dataset_class, 'classes', None)
    if classes is None:
        return [str(i) for i in range(num_classes)]
    classes = list(classes)
    if len(classes) < num_classes:
        classes.extend(str(i) for i in range(len(classes), num_classes))
    return classes[:num_classes]


def _plot_multilabel(ax, pred, gt, dataset_class):
    pred = _prepare_multilabel_vector(pred)
    gt = _prepare_multilabel_vector(gt)
    num_classes = max(len(pred), len(gt))
    pred = np.pad(pred, (0, max(0, num_classes - len(pred))))[:num_classes]
    gt = np.pad(gt, (0, max(0, num_classes - len(gt))))[:num_classes]
    names = _get_class_names(dataset_class, num_classes)
    x = np.arange(num_classes)
    ax.bar(x - 0.18, pred, width=0.36, label='pred')
    ax.bar(x + 0.18, gt, width=0.36, label='gt')
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=6)
    ax.legend(fontsize=7)


def _save_patch_level_plot(
    image_tensor,
    seg_gt_tensor,
    mlc_pred_tensor,
    mlc_gt_tensor,
    patch_size,
    dataset_class,
    save_path,
    pseudo_label_tensor=None,
):
    image = _prepare_image(image_tensor)
    seg_gt = _prepare_mask(seg_gt_tensor)
    pred = _prepare_vector(mlc_pred_tensor)
    score_map = _patch_scores_to_map(pred, patch_size, image.shape)

    n_cols = 5 if pseudo_label_tensor is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 4.0))
    axes = np.asarray(axes).reshape(-1)

    axes[0].imshow(image, cmap=None if image.ndim == 3 else 'gray')
    axes[0].set_title('image')
    axes[1].imshow(seg_gt, cmap='tab20', interpolation='nearest')
    axes[1].set_title('label')

    next_ax = 2
    if pseudo_label_tensor is not None:
        pseudo = _prepare_mask(pseudo_label_tensor)
        axes[next_ax].imshow(pseudo, cmap='tab20', interpolation='nearest')
        axes[next_ax].set_title('pseudo')
        next_ax += 1

    axes[next_ax].imshow(score_map, cmap='viridis', vmin=0, vmax=1)
    axes[next_ax].set_title('patch score')

    _plot_multilabel(axes[next_ax + 1], mlc_pred_tensor, mlc_gt_tensor, dataset_class)
    axes[next_ax + 1].set_title('multi-label')

    for ax in axes:
        ax.axis('off')
    axes[next_ax + 1].axis('on')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def visualize_patch_level_src(
    image_tensor,
    seg_gt_tensor,
    mlc_pred_tensor,
    mlc_gt_tensor,
    patch_size,
    dataset_class,
    save_path,
):
    _save_patch_level_plot(
        image_tensor=image_tensor,
        seg_gt_tensor=seg_gt_tensor,
        mlc_pred_tensor=mlc_pred_tensor,
        mlc_gt_tensor=mlc_gt_tensor,
        patch_size=patch_size,
        dataset_class=dataset_class,
        save_path=save_path,
    )


def visualize_patch_level_tar(
    image_tensor,
    seg_gt_tensor,
    pseudo_label_tensor,
    mlc_pred_tensor,
    mlc_gt_tensor,
    patch_size,
    dataset_class,
    save_path,
):
    _save_patch_level_plot(
        image_tensor=image_tensor,
        seg_gt_tensor=seg_gt_tensor,
        pseudo_label_tensor=pseudo_label_tensor,
        mlc_pred_tensor=mlc_pred_tensor,
        mlc_gt_tensor=mlc_gt_tensor,
        patch_size=patch_size,
        dataset_class=dataset_class,
        save_path=save_path,
    )
