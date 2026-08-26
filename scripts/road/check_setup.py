#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from road_ads.config import load_config, resolve_repo_path
from road_ads.data import read_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one road ADS config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-missing-pretrained", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    required = ["target_labeled", "target_unlabeled", "target_val"]
    if config["protocol"] == "ssda":
        required.append("source")
    for name in required:
        branch = config["data"][name]
        root = resolve_repo_path(branch["root"])
        split = resolve_repo_path(branch["split"])
        if root is None or not root.exists():
            raise FileNotFoundError(f"{name} root is missing: {root}")
        if split is None or not split.is_file():
            raise FileNotFoundError(f"{name} split is missing: {split}")
        samples = read_split(split)
        first_image = Path(samples[0][0])
        first_image = first_image if first_image.is_absolute() else root / first_image
        if not first_image.is_file():
            raise FileNotFoundError(f"{name} first image is missing: {first_image}")
        print(f"{name:18s}: {len(samples):6d} samples")

    pretrained = resolve_repo_path(config["model"].get("pretrained"))
    if pretrained is None or not pretrained.is_file():
        message = f"Pretrained checkpoint is missing: {pretrained}"
        if not args.allow_missing_pretrained:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}")
    else:
        print(f"pretrained         : {pretrained}")
    print(f"OK: {args.config}")


if __name__ == "__main__":
    main()
