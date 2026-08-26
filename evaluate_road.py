#!/usr/bin/env python3
import argparse

from road_ads.config import load_config
from road_ads.trainer import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a road ADS checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--student", choices=["left", "right"], default="left")
    args = parser.parse_args()
    evaluate_checkpoint(load_config(args.config), args.checkpoint, args.student)


if __name__ == "__main__":
    main()
