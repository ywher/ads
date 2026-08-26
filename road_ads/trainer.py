from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from deeplab.discriminator import discriminator
from deeplab.tps import sparse_image_warp
from perturbations import DGW
from ramps import sigmoid_rampup

from .config import dump_config, resolve_repo_path
from .data import build_dataset, build_loader, infinite_loader
from .models import build_segmentor, trainable_parameter_count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _branch_config(config: Dict[str, object], name: str) -> Dict[str, object]:
    branch = dict(config["data"][name])
    branch["input"] = config["input"]
    return branch


def _make_train_loader(config: Dict[str, object], name: str, labeled: bool):
    dataset = build_dataset(
        _branch_config(config, name), train=True, labeled=labeled)
    return build_loader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        workers=int(config["training"].get("workers", 4)),
        shuffle=True,
        drop_last=True,
    )


def _make_val_loader(config: Dict[str, object]):
    branch = _branch_config(config, "target_val")
    dataset = build_dataset(branch, train=False, labeled=True)
    return build_loader(
        dataset, batch_size=1,
        workers=int(config["evaluation"].get("workers", 2)),
        shuffle=False, drop_last=False)


def _to_device(batch: Dict[str, object], device: torch.device):
    return (
        batch["image"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


def _minmax_image(image: torch.Tensor) -> torch.Tensor:
    low = image.amin(dim=(1, 2, 3), keepdim=True)
    high = image.amax(dim=(1, 2, 3), keepdim=True)
    return (image - low) / (high - low).clamp_min(1e-6)


def _one_hot(label: torch.Tensor, classes: int) -> torch.Tensor:
    valid = label != 255
    safe = label.masked_fill(~valid, 0)
    result = F.one_hot(safe, num_classes=classes).permute(0, 3, 1, 2).float()
    return result * valid.unsqueeze(1)


def _warp_nchw(
    tensor: torch.Tensor,
    source_locs: torch.Tensor,
    dest_locs: torch.Tensor,
) -> torch.Tensor:
    warped, _ = sparse_image_warp(
        tensor.permute(0, 2, 3, 1), source_locs, dest_locs,
        interpolation_order=1, regularization_weight=0.0,
        num_boundaries_points=0)
    return warped.permute(0, 3, 1, 2)


@torch.no_grad()
def _stabilization_targets(
    left_warped: torch.Tensor,
    right_warped: torch.Tensor,
    left_aligned: torch.Tensor,
    right_aligned: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lp, rp = left_warped.softmax(1), right_warped.softmax(1)
    la, ra = left_aligned.softmax(1), right_aligned.softmax(1)
    l_conf, l_cls = lp.max(1)
    r_conf, r_cls = rp.max(1)
    la_conf, la_cls = la.max(1)
    ra_conf, ra_cls = ra.max(1)

    l_stable = (l_conf > threshold) & (la_conf > threshold) & (l_cls == la_cls)
    r_stable = (r_conf > threshold) & (ra_conf > threshold) & (r_cls == ra_cls)
    l_error = (left_warped - left_aligned).square().mean(1)
    r_error = (right_warped - right_aligned).square().mean(1)

    # Target for the left model: use the right student where left is unstable,
    # or where both are stable and right is more warp-consistent.
    use_right = (~l_stable) | (l_stable & r_stable & (r_error < l_error))
    use_left = (~r_stable) | (l_stable & r_stable & (l_error < r_error))
    target_left = torch.where(use_right.unsqueeze(1), right_warped, left_warped)
    target_right = torch.where(use_left.unsqueeze(1), left_warped, right_warped)
    return target_left.detach(), target_right.detach()


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _make_optimizer(model: nn.Module, config: Dict[str, object]):
    opt_cfg = config["optimizer"]
    lr = float(opt_cfg["lr"])
    groups = model.optimizer_groups(lr)
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    for group in groups:
        if group.get("weight_decay") is None:
            group["weight_decay"] = weight_decay
    if str(opt_cfg["type"]).lower() == "sgd":
        return torch.optim.SGD(
            groups, lr=lr, momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay)
    if str(opt_cfg["type"]).lower() == "adamw":
        return torch.optim.AdamW(
            groups, lr=lr, betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {opt_cfg['type']}")


def _adjust_lr(
    optimizer, base_lr: float, step: int, max_steps: int, power: float,
    warmup_iters: int = 0, warmup_ratio: float = 1e-6,
):
    current = base_lr * max(1.0 - step / max_steps, 0.0) ** power
    if warmup_iters > 0 and step < warmup_iters:
        alpha = step / warmup_iters
        current *= warmup_ratio + alpha * (1.0 - warmup_ratio)
    for group in optimizer.param_groups:
        group["lr"] = current * float(group.get("lr_scale", 1.0))


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _student_step(
    model: nn.Module,
    disc: nn.Module,
    optimizer,
    scaler,
    labeled_image: torch.Tensor,
    labeled_mask: torch.Tensor,
    unlabeled_image: torch.Tensor,
    warped_image: torch.Tensor,
    aligned_logits: torch.Tensor,
    stabilization_target: torch.Tensor,
    real_disc_features: torch.Tensor,
    source_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
    config: Dict[str, object],
    consistency_weight: float,
    stabilization_weight: float,
    amp: bool,
) -> Tuple[Dict[str, float], torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    losses_cfg = config["ads"]

    with _autocast(labeled_image.device, amp):
        supervised_logits = model(labeled_image)
        supervised = F.cross_entropy(
            supervised_logits, labeled_mask, ignore_index=255)
    scaler.scale(supervised).backward()

    source_loss = supervised.new_zeros(())
    if source_batch is not None:
        with _autocast(labeled_image.device, amp):
            source_logits = model(source_batch[0])
            source_loss = F.cross_entropy(
                source_logits, source_batch[1], ignore_index=255)
            weighted_source = float(losses_cfg.get("source_weight", 1.0)) * source_loss
        scaler.scale(weighted_source).backward()

    with _autocast(labeled_image.device, amp):
        warped_logits = model(warped_image)
        consistency = F.mse_loss(
            warped_logits.softmax(1), aligned_logits.softmax(1))
        stabilization = F.mse_loss(
            warped_logits.softmax(1), stabilization_target.softmax(1))
        warp_loss = (
            consistency_weight * consistency
            + stabilization_weight * stabilization
        )
    scaler.scale(warp_loss).backward()

    with _autocast(labeled_image.device, amp):
        unlabeled_logits = model(unlabeled_image)
        image_disc = _minmax_image(unlabeled_image)
        fake_input = torch.cat((unlabeled_logits.softmax(1), image_disc), dim=1)
        disc_score, fake_features = disc(fake_input)
        selected = disc_score.detach().flatten() > float(losses_cfg["self_training_threshold"])
        self_training = supervised.new_zeros(())
        if selected.any():
            pseudo = unlabeled_logits.detach().argmax(1)
            self_training = F.cross_entropy(
                unlabeled_logits[selected], pseudo[selected], ignore_index=255)
        feature_matching = (
            fake_features.mean(0) - real_disc_features.mean(0)
        ).abs().mean()
        adversarial_aux = (
            float(losses_cfg["self_training_weight"]) * self_training
            + float(losses_cfg["feature_matching_weight"]) * feature_matching
        )
    scaler.scale(adversarial_aux).backward()
    scaler.step(optimizer)

    total_value = (
        float(supervised.detach())
        + float(losses_cfg.get("source_weight", 1.0)) * float(source_loss.detach())
        + consistency_weight * float(consistency.detach())
        + stabilization_weight * float(stabilization.detach())
        + float(losses_cfg["self_training_weight"]) * float(self_training.detach())
        + float(losses_cfg["feature_matching_weight"]) * float(feature_matching.detach())
    )
    return {
        "total": total_value,
        "sup": float(supervised.detach()),
        "source": float(source_loss.detach()),
        "cons": float(consistency.detach()),
        "stable": float(stabilization.detach()),
        "self": float(self_training.detach()),
        "fm": float(feature_matching.detach()),
        "selected": float(selected.float().mean()),
    }, fake_input.detach()


def _discriminator_step(
    disc: nn.Module,
    optimizer,
    fake_input: torch.Tensor,
    real_input: torch.Tensor,
) -> float:
    _set_requires_grad(disc, True)
    optimizer.zero_grad(set_to_none=True)
    fake_score, _ = disc(fake_input)
    real_score, _ = disc(real_input)
    loss = 0.5 * (
        F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))
        + F.binary_cross_entropy(real_score, torch.ones_like(real_score))
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, classes: int, device: torch.device):
    model.eval()
    confusion = torch.zeros((classes, classes), dtype=torch.float64)
    for batch in loader:
        image, label = _to_device(batch, device)
        logits = model(image)
        prediction = logits.argmax(1)
        valid = (label != 255) & (label >= 0) & (label < classes)
        encoded = classes * label[valid] + prediction[valid]
        confusion += torch.bincount(
            encoded.cpu(), minlength=classes * classes).reshape(classes, classes)
    intersection = confusion.diag()
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = intersection / union.clamp_min(1)
    return float(iou.mean() * 100.0), [float(value * 100.0) for value in iou]


def _save_checkpoint(
    output: Path, step: int, left, right, left_opt, right_opt,
    left_disc, right_disc, best: float,
) -> Path:
    path = output / f"checkpoint_{step:06d}.pth"
    torch.save({
        "step": step,
        "left": left.state_dict(),
        "right": right.state_dict(),
        "left_optimizer": left_opt.state_dict(),
        "right_optimizer": right_opt.state_dict(),
        "left_discriminator": left_disc.state_dict(),
        "right_discriminator": right_disc.state_dict(),
        "best_miou": best,
    }, path)
    return path


def train(config: Dict[str, object]) -> None:
    device = torch.device(config["runtime"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but unavailable")
    set_seed(int(config["runtime"].get("seed", 0)))
    torch.backends.cudnn.benchmark = bool(config["runtime"].get("benchmark", True))

    output = resolve_repo_path(config["output_dir"])
    assert output is not None
    output.mkdir(parents=True, exist_ok=True)
    dump_config(config, output / "resolved_config.yaml")

    target_labeled = infinite_loader(_make_train_loader(config, "target_labeled", True))
    target_unlabeled = infinite_loader(_make_train_loader(config, "target_unlabeled", False))
    source = None
    if str(config["protocol"]).lower() == "ssda":
        source = infinite_loader(_make_train_loader(config, "source", True))
    val_loader = _make_val_loader(config)

    left = build_segmentor(config["model"]).to(device)
    right = build_segmentor(config["model"]).to(device)
    total, trainable = trainable_parameter_count(left)
    print(f"Segmentor parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    classes = int(config["model"]["num_classes"])
    left_disc = discriminator(classes, dataset="cityscapes").to(device)
    right_disc = discriminator(classes, dataset="cityscapes").to(device)
    left_opt = _make_optimizer(left, config)
    right_opt = _make_optimizer(right, config)
    disc_lr = float(config["ads"]["discriminator_lr"])
    left_disc_opt = torch.optim.Adam(left_disc.parameters(), lr=disc_lr, betas=(0.9, 0.99))
    right_disc_opt = torch.optim.Adam(right_disc.parameters(), lr=disc_lr, betas=(0.9, 0.99))

    amp = bool(config["training"].get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    max_steps = int(config["training"]["max_iters"])
    base_lr = float(config["optimizer"]["lr"])
    power = float(config["optimizer"].get("power", 0.9))
    warmup_iters = int(config["optimizer"].get("warmup_iters", 0))
    warmup_ratio = float(config["optimizer"].get("warmup_ratio", 1e-6))
    crop_size = tuple(int(value) for value in config["input"]["crop_size"])
    dgw = DGW(img_h=crop_size[0], img_w=crop_size[1])
    eval_interval = int(config["evaluation"].get("interval", 10000))
    save_interval = int(config["training"].get("save_interval", eval_interval))
    log_interval = int(config["training"].get("log_interval", 50))
    best = -1.0
    start = time.time()

    for step in range(1, max_steps + 1):
        left.train()
        right.train()
        left_disc.train()
        right_disc.train()
        _adjust_lr(
            left_opt, base_lr, step - 1, max_steps, power,
            warmup_iters, warmup_ratio)
        _adjust_lr(
            right_opt, base_lr, step - 1, max_steps, power,
            warmup_iters, warmup_ratio)
        _adjust_lr(left_disc_opt, disc_lr, step - 1, max_steps, power)
        _adjust_lr(right_disc_opt, disc_lr, step - 1, max_steps, power)

        labeled_image, labeled_mask = _to_device(next(target_labeled), device)
        unlabeled_image, _ = _to_device(next(target_unlabeled), device)
        source_batch = _to_device(next(source), device) if source is not None else None
        warped_image, source_locs, dest_locs = dgw.warp(
            unlabeled_image.permute(0, 2, 3, 1))

        with torch.no_grad(), _autocast(device, amp):
            left_u_ref = left(unlabeled_image)
            right_u_ref = right(unlabeled_image)
            left_w_ref = left(warped_image)
            right_w_ref = right(warped_image)
            left_aligned = _warp_nchw(left_u_ref, source_locs, dest_locs)
            right_aligned = _warp_nchw(right_u_ref, source_locs, dest_locs)
            target_left, target_right = _stabilization_targets(
                left_w_ref, right_w_ref, left_aligned, right_aligned,
                float(config["ads"]["stable_threshold"]))

            real_input = torch.cat(
                (_one_hot(labeled_mask, classes), _minmax_image(labeled_image)), dim=1)

        consistency_weight = float(config["ads"]["consistency_weight"]) * sigmoid_rampup(
            step, int(config["ads"]["consistency_rampup"]))
        stabilization_weight = float(config["ads"]["stabilization_weight"]) * sigmoid_rampup(
            step, int(config["ads"]["stabilization_rampup"]))

        _set_requires_grad(left_disc, False)
        _set_requires_grad(right_disc, False)
        with torch.no_grad():
            _, left_real_features = left_disc(real_input)
            _, right_real_features = right_disc(real_input)

        left_stats, left_fake = _student_step(
            left, left_disc, left_opt, scaler, labeled_image, labeled_mask,
            unlabeled_image, warped_image, left_aligned, target_left,
            left_real_features, source_batch, config, consistency_weight,
            stabilization_weight, amp)
        right_stats, right_fake = _student_step(
            right, right_disc, right_opt, scaler, labeled_image, labeled_mask,
            unlabeled_image, warped_image, right_aligned, target_right,
            right_real_features, source_batch, config, consistency_weight,
            stabilization_weight, amp)
        scaler.update()

        left_disc_loss = _discriminator_step(
            left_disc, left_disc_opt, left_fake, real_input)
        right_disc_loss = _discriminator_step(
            right_disc, right_disc_opt, right_fake, real_input)

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - start
            eta = elapsed / step * (max_steps - step)
            print(
                f"iter {step:06d}/{max_steps:06d} "
                f"left={left_stats['total']:.4f} right={right_stats['total']:.4f} "
                f"sup={0.5 * (left_stats['sup'] + right_stats['sup']):.4f} "
                f"src={0.5 * (left_stats['source'] + right_stats['source']):.4f} "
                f"D={0.5 * (left_disc_loss + right_disc_loss):.4f} "
                f"elapsed={elapsed / 3600:.2f}h eta={eta / 3600:.2f}h",
                flush=True)

        if step % eval_interval == 0 or step == max_steps:
            miou, class_iou = evaluate(left, val_loader, classes, device)
            record = {"step": step, "miou": miou, "class_iou": class_iou}
            with (output / "evaluation.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"validation iter={step}: left mIoU={miou:.2f}", flush=True)
            if miou > best:
                best = miou
                torch.save(left.state_dict(), output / "best_left.pth")

        if step % save_interval == 0 or step == max_steps:
            _save_checkpoint(
                output, step, left, right, left_opt, right_opt,
                left_disc, right_disc, best)


def evaluate_checkpoint(
    config: Dict[str, object], checkpoint_path: str, student: str = "left"
) -> float:
    device = torch.device(config["runtime"].get("device", "cuda"))
    model = build_segmentor(config["model"]).to(device)
    checkpoint = torch.load(resolve_repo_path(checkpoint_path), map_location="cpu")
    state = checkpoint.get(student, checkpoint)
    model.load_state_dict(state, strict=True)
    loader = _make_val_loader(config)
    miou, class_iou = evaluate(
        model, loader, int(config["model"]["num_classes"]), device)
    print(json.dumps({"miou": miou, "class_iou": class_iou}, indent=2))
    return miou
