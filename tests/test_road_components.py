from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deeplab.tps import sparse_image_warp
from perturbations import DGW
from road_ads.config import load_config
from road_ads.models import NativeDeepLabV2


def test_config_inheritance():
    config = load_config("configs/road/vfm/ssda/gta2cityscapes.yaml")
    assert config["protocol"] == "ssda"
    assert config["training"]["max_iters"] == 40000
    assert config["model"]["num_classes"] == 19


def test_dgw_batch_and_backward():
    image = torch.randn(2, 3, 32, 64, requires_grad=True)
    dgw = DGW(num_split=2, img_h=32, img_w=64)
    warped, source, destination = dgw.warp(image.permute(0, 2, 3, 1))
    assert warped.shape == image.shape
    logits = torch.randn(2, 5, 32, 64, requires_grad=True)
    warped_logits, _ = sparse_image_warp(
        logits.permute(0, 2, 3, 1), source, destination,
        interpolation_order=1)
    warped_logits.mean().backward()
    assert logits.grad is not None


def test_native_forward_backward_without_pretrain():
    model = NativeDeepLabV2(num_classes=19)
    image = torch.randn(1, 3, 64, 128)
    logits = model(image)
    assert logits.shape[:2] == (1, 19)
    logits.mean().backward()


if __name__ == "__main__":
    test_config_inheritance()
    test_dgw_batch_and_backward()
    test_native_forward_backward_without_pretrain()
    print("road ADS component tests passed")
