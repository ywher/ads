#!/usr/bin/env python3
import argparse

from road_ads.config import load_config
from road_ads.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ADS-DGW on driving-scene Semi or SSDA protocols")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
