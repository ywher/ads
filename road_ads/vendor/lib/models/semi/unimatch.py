# ---------------------------------------------------------------
# UniMatch-style semi-supervised trainer.
# ---------------------------------------------------------------
import os
import random

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from lib.loss.losses import parse_losses
from lib.models.model_utils.dacs_transforms import (
    denorm_,
    get_mean_std_self,
    renorm_,
    strong_transform_wo_mix,
)
from lib.models.model_utils.funcs import add_prefix
from lib.models.model_utils.visualization import prepare_debug_out, subplotimg

from .dacs import DACS


class UniMatchDACS(DACS):
    """UniMatchV2/SSMIS basic semi-supervised objective on DAFormer models.

    The labeled branch is the same supervised target branch as DACS. The
    unlabeled branch follows UniMatchV2: weak-view EMA pseudo labels supervise
    two strongly augmented views, with optional rectangular CutMix, confidence
    filtering, MIC, and complementary dropout.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        unimatch_cfg = self.semi_cfg.get('unimatch', {})
        self.share_src_backward = True

        self.unimatch_sup_weight = float(
            unimatch_cfg.get('sup_weight', 1.0))
        self.unimatch_unsup_weight = float(
            unimatch_cfg.get('unsup_weight', 1.0))
        self.unimatch_total_divisor = float(
            unimatch_cfg.get('total_divisor', 2.0))
        self.unimatch_conf_thresh = float(
            unimatch_cfg.get('conf_thresh', self.pseudo_threshold))
        self.unimatch_cutmix_prob = float(
            unimatch_cfg.get('cutmix_prob', 0.5))
        self.unimatch_cutmix_min = float(
            unimatch_cfg.get('cutmix_size_min', 0.02))
        self.unimatch_cutmix_max = float(
            unimatch_cfg.get('cutmix_size_max', 0.4))
        self.unimatch_cutmix_ratio_min = float(
            unimatch_cfg.get('cutmix_ratio_min', 0.3))
        self.unimatch_cutmix_ratio_max = float(
            unimatch_cfg.get('cutmix_ratio_max', 1 / 0.3))
        self.unimatch_grayscale_prob = float(
            unimatch_cfg.get('grayscale_prob', 0.2))
        self.unimatch_confidence_weight = bool(
            unimatch_cfg.get('confidence_weight', False))
        self.unimatch_use_dacs_pseudo_weight = bool(
            unimatch_cfg.get(
                'use_dacs_pseudo_weight',
                self.vfm_pl_filter_enabled))
        self.unimatch_use_mic = bool(
            unimatch_cfg.get('use_mic', self.enable_masking))
        self.unimatch_comp_drop = bool(
            unimatch_cfg.get('complementary_dropout', False))
        self.unimatch_comp_drop_kept_ratio = float(
            unimatch_cfg.get('complementary_dropout_kept_ratio', 0.5))

        if self.enable_fdist:
            self.logger.warning(
                '[UniMatch] imnet feature distillation is configured but the '
                'UniMatch trainer currently keeps only supervised + '
                'unlabeled consistency branches.')

        self.logger.info(
            '[UniMatch] sup_weight=%.3f, unsup_weight=%.3f, divisor=%.3f, '
            'conf_thresh=%.3f, cutmix_prob=%.2f, gray_prob=%.2f, '
            'use_dacs_pseudo_weight=%s, MIC=%s, comp_drop=%s',
            self.unimatch_sup_weight,
            self.unimatch_unsup_weight,
            self.unimatch_total_divisor,
            self.unimatch_conf_thresh,
            self.unimatch_cutmix_prob,
            self.unimatch_grayscale_prob,
            self.unimatch_use_dacs_pseudo_weight,
            self.unimatch_use_mic and self.enable_masking,
            self.unimatch_comp_drop,
        )

    def _make_strong_view(self, img, means, stds):
        """Create one strong view: color jitter/blur plus optional grayscale."""
        batch_size = img.shape[0]
        params = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[:batch_size],
            'std': stds[:batch_size],
        }
        strong_img, _ = strong_transform_wo_mix(params, data=img.clone())
        return self._maybe_grayscale(strong_img, means, stds)

    def _maybe_grayscale(self, img, means, stds):
        """Apply random grayscale in normalized tensor space."""
        if self.unimatch_grayscale_prob <= 0:
            return img
        if random.uniform(0, 1) >= self.unimatch_grayscale_prob:
            return img

        out = img.clone()
        mean = means[:out.shape[0]]
        std = stds[:out.shape[0]]
        denorm_(out, mean, std)
        gray = (
            0.2989 * out[:, 0:1] +
            0.5870 * out[:, 1:2] +
            0.1140 * out[:, 2:3]
        )
        out = gray.repeat(1, 3, 1, 1)
        renorm_(out, mean, std)
        return out

    def _obtain_cutmix_masks(self, batch_size, height, width, device):
        """Generate rectangular CutMix masks for each sample in a batch."""
        masks = torch.zeros(
            (batch_size, height, width), device=device, dtype=torch.float32)
        if self.unimatch_cutmix_prob <= 0 or batch_size < 2:
            return masks

        area = float(height * width)
        for idx in range(batch_size):
            if random.uniform(0, 1) > self.unimatch_cutmix_prob:
                continue

            for _ in range(10):
                target_area = random.uniform(
                    self.unimatch_cutmix_min,
                    self.unimatch_cutmix_max) * area
                ratio = random.uniform(
                    self.unimatch_cutmix_ratio_min,
                    self.unimatch_cutmix_ratio_max)
                cutmix_w = int(np.sqrt(target_area / ratio))
                cutmix_h = int(np.sqrt(target_area * ratio))
                if cutmix_w <= 0 or cutmix_h <= 0:
                    continue
                if cutmix_w <= width and cutmix_h <= height:
                    x = random.randint(0, width - cutmix_w)
                    y = random.randint(0, height - cutmix_h)
                    masks[idx, y:y + cutmix_h, x:x + cutmix_w] = 1
                    break
        return masks

    @staticmethod
    def _cutmix_map(value, mask):
        if value is None:
            return None
        mixed = value.clone()
        flipped = value.flip(0)
        mixed[mask.bool()] = flipped[mask.bool()]
        return mixed

    def _apply_cutmix(self, img, pseudo_label, pseudo_conf, pseudo_weight, mask):
        """Apply the same CutMix mask to strong image and pseudo targets."""
        if mask.sum() <= 0:
            return img, pseudo_label, pseudo_conf, pseudo_weight

        mixed_img = img.clone()
        img_mask = mask.bool().unsqueeze(1).expand_as(mixed_img)
        mixed_img[img_mask] = img.flip(0)[img_mask]
        mixed_label = self._cutmix_map(pseudo_label, mask)
        mixed_conf = self._cutmix_map(pseudo_conf, mask)
        mixed_weight = self._cutmix_map(pseudo_weight, mask)
        return mixed_img, mixed_label, mixed_conf, mixed_weight

    def _build_confident_weight(self, pseudo_weight, pseudo_conf):
        """Build a pixel mask/weight from confidence and valid pseudo regions."""
        valid_region = pseudo_weight > 0
        confident = pseudo_conf.ge(self.unimatch_conf_thresh)
        if self.unimatch_use_dacs_pseudo_weight:
            base_weight = pseudo_weight
        else:
            base_weight = valid_region.to(pseudo_weight.dtype)
        weight = base_weight * confident.to(pseudo_weight.dtype)
        if self.unimatch_confidence_weight:
            weight = weight * pseudo_conf.detach()

        valid_den = valid_region.float().sum().clamp_min(1.0)
        mask_ratio = (confident & valid_region).float().sum() / valid_den
        return weight, float(mask_ratio.detach().item())

    def _forward_unlabeled_consistency(
            self,
            img_s1,
            img_s2,
            label_s1,
            label_s2,
            weight_s1,
            weight_s2,
            seg_debug):
        """Forward two strong views and backpropagate the consistency loss."""
        strong_img = torch.cat([img_s1, img_s2], dim=0)
        strong_label = torch.cat([label_s1, label_s2], dim=0)
        strong_weight = torch.cat([weight_s1, weight_s2], dim=0)

        comp_drop = False
        if self.unimatch_comp_drop:
            comp_drop = {
                'kept_ratio': self.unimatch_comp_drop_kept_ratio,
            }

        pred_results = self.get_model().forward_train(
            (strong_img, strong_label),
            seg_weight=strong_weight,
            return_feat=False,
            loss_key='unsup',
            comp_drop=comp_drop,
        )
        seg_debug['UniMatch Strong'] = self.get_model().decode_head.debug_output
        seg_logits = pred_results.pop('seg_logits', None)
        raw_loss, log_vars = parse_losses(pred_results)

        weighted_loss = raw_loss * self.unimatch_unsup_weight / \
            max(self.unimatch_total_divisor, 1e-6)
        weighted_loss.backward()

        return seg_logits, log_vars, float(raw_loss.detach().item()), \
            float(weighted_loss.detach().item())

    def _save_unimatch_debug(
            self,
            means,
            stds,
            tar_img,
            img_s1,
            img_s2,
            pseudo_label,
            pseudo_conf,
            pseudo_weight,
            cutmix_box1,
            cutmix_box2,
            strong_logits):
        """Save compact weak/strong/CutMix debug panels."""
        out_dir = os.path.join(self.cfg.respth, 'debug', 'unimatch')
        os.makedirs(out_dir, exist_ok=True)

        if strong_logits is None:
            pred_s1 = pred_s2 = None
        else:
            logits = self._get_prediction_for_debug(
                strong_logits, pseudo_label.shape[-2:])
            if logits.shape[0] >= tar_img.shape[0] * 2:
                pred_s1, pred_s2 = logits.chunk(2, dim=0)
                pred_s1 = torch.argmax(pred_s1, dim=1)
                pred_s2 = torch.argmax(pred_s2, dim=1)
            else:
                pred_s1 = torch.argmax(logits, dim=1)
                pred_s2 = None

        max_samples = min(2, tar_img.shape[0])
        means_cpu = means.detach().cpu()
        stds_cpu = stds.detach().cpu()
        for idx in range(max_samples):
            panels = [
                ('Weak', tar_img[idx]),
                ('Strong1', img_s1[idx]),
                ('Strong2', img_s2[idx]),
                ('Pseudo', pseudo_label[idx]),
                ('Conf', pseudo_conf[idx]),
                ('Weight', pseudo_weight[idx]),
                ('CutMix1', cutmix_box1[idx]),
                ('CutMix2', cutmix_box2[idx]),
            ]
            if pred_s1 is not None:
                panels.append(('Pred S1', pred_s1[idx]))
            if pred_s2 is not None:
                panels.append(('Pred S2', pred_s2[idx]))

            fig, axs = plt.subplots(
                1, len(panels), figsize=(3 * len(panels), 3),
                gridspec_kw={
                    'hspace': 0.1,
                    'wspace': 0.02,
                    'top': 0.88,
                    'bottom': 0,
                    'right': 1,
                    'left': 0,
                },
            )
            if len(panels) == 1:
                axs = [axs]
            for ax, (title, value) in zip(axs, panels):
                if torch.is_tensor(value):
                    value = value.detach().cpu()
                debug_out = prepare_debug_out(
                    title,
                    value,
                    means_cpu[idx:idx + 1],
                    stds_cpu[idx:idx + 1],
                )
                if debug_out.get('cmap') == 'cityscapes':
                    debug_out['palette'] = getattr(
                        self, 'debug_palette', None)
                subplotimg(ax, **debug_out)
                ax.axis('off')
            fig.savefig(os.path.join(
                out_dir, f'{self.local_iter + 1:06d}_{idx}.png'))
            plt.close(fig)

    def forward_train_step(self, data_batch, valid_pseudo_mask=None):
        """Run one UniMatch-style semi-supervised training iteration."""
        if len(data_batch) == 5:
            src_img, src_seg_lbl, tar_img, tar_seg_lbl, target_img_paths = data_batch
        else:
            src_img, src_seg_lbl, tar_img, tar_seg_lbl = data_batch
            target_img_paths = None
        del tar_seg_lbl, target_img_paths

        log_vars = {}
        batch_size = src_img.shape[0]
        dev = src_img.device
        self._grad_conflict_source_grads = None
        self._adapter_grad_conflict_source_grads = None

        self._update_teacher_and_mic_state()
        self.update_debug_state()
        seg_debug = {}
        means, stds = get_mean_std_self(
            self.img_mean, self.img_std, batch_size, dev)

        source_state = self._forward_source_supervised_loss(
            src_img,
            src_seg_lbl,
            seg_debug,
        )
        log_vars.update(add_prefix(source_state['log_vars'], 'src'))

        weighted_src_loss = source_state['src_loss'] * \
            self.unimatch_sup_weight / max(self.unimatch_total_divisor, 1e-6)
        weighted_src_loss.backward()
        src_loss_value = float(weighted_src_loss.detach().item())

        total_loss_value = src_loss_value
        semi_enabled = not self.source_only and \
            self.local_iter >= self.semi_begin_iter

        if not semi_enabled:
            self.local_iter += 1
            log_vars['total_loss'] = total_loss_value
            log_vars['mix_loss'] = 0.0
            log_vars['mix_seg_loss'] = 0.0
            log_vars['src_mix_ratio'] = 0.0
            return log_vars

        pseudo_label, pseudo_weight, pseudo_mask, pseudo_conf, _, vplf_log_vars = \
            self._generate_target_pseudo_state(
                tar_img,
                valid_pseudo_mask,
                seg_debug,
            )
        del pseudo_mask
        log_vars.update(vplf_log_vars)

        pseudo_weight, mask_ratio = self._build_confident_weight(
            pseudo_weight, pseudo_conf)
        img_s1 = self._make_strong_view(tar_img, means, stds)
        img_s2 = self._make_strong_view(tar_img, means, stds)
        _, _, height, width = img_s1.shape
        cutmix_box1 = self._obtain_cutmix_masks(
            batch_size, height, width, dev)
        cutmix_box2 = self._obtain_cutmix_masks(
            batch_size, height, width, dev)

        img_s1, label_s1, conf_s1, weight_s1 = self._apply_cutmix(
            img_s1, pseudo_label, pseudo_conf, pseudo_weight, cutmix_box1)
        img_s2, label_s2, conf_s2, weight_s2 = self._apply_cutmix(
            img_s2, pseudo_label, pseudo_conf, pseudo_weight, cutmix_box2)
        del conf_s1, conf_s2

        strong_logits, unsup_log_vars, raw_unsup, weighted_unsup = \
            self._forward_unlabeled_consistency(
                img_s1,
                img_s2,
                label_s1,
                label_s2,
                weight_s1,
                weight_s2,
                seg_debug,
            )
        log_vars.update(add_prefix(unsup_log_vars, 'mix'))
        total_loss_value += weighted_unsup

        log_vars.update({
            'mix_loss': weighted_unsup,
            'mix_seg_loss': raw_unsup,
            'src_mix_ratio': 0.0,
            'unimatch_mask_ratio': mask_ratio,
            'unimatch_cutmix1_ratio': float(cutmix_box1.mean().detach().item()),
            'unimatch_cutmix2_ratio': float(cutmix_box2.mean().detach().item()),
            'unimatch_unsup_weight': self.unimatch_unsup_weight,
            'unimatch_conf_thresh': self.unimatch_conf_thresh,
        })

        if self.unimatch_use_mic and self.enable_masking:
            masked_log_vars, masked_loss_value = self._forward_mic_loss(
                src_img,
                src_seg_lbl,
                tar_img,
                valid_pseudo_mask,
                pseudo_label,
                pseudo_weight,
                seg_debug,
            )
            log_vars.update(add_prefix(masked_log_vars, 'masked'))
            total_loss_value += masked_loss_value

        log_vars['total_loss'] = total_loss_value

        if self._is_master_process() and self.debug_img_interval > 0 and \
                (self.local_iter + 1) % self.debug_img_interval == 0:
            self._save_unimatch_debug(
                means,
                stds,
                tar_img,
                img_s1,
                img_s2,
                pseudo_label,
                pseudo_conf,
                pseudo_weight,
                cutmix_box1,
                cutmix_box2,
                strong_logits,
            )
            self._save_hrda_debug_images(seg_debug, batch_size, means, stds,
                                         pseudo_weight)

        self.local_iter += 1
        return log_vars
