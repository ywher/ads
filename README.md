# Adversarial Dual-Student with Differentiable Spatial Warping for Semi-Supervised Semantic Segmentation (ADS-SemiSeg)
This repository contains official implementation of Adversarial Dual-Student with Differentiable Spatial Warping for Semi-Supervised Semantic Segmentation in TCSVT 2022, by Cong Cao, Tianwei Lin, Dongliang He, Fu Li, Huanjing Yue, Jingyu Yang, and Errui Ding. [[arxiv]](https://arxiv.org/abs/2203.02792) [[journal]](https://ieeexplore.ieee.org/abstract/document/9889741)

<p align="center">
  <img width="800" src="https://github.com/cao-cong/ADS-SemiSeg/blob/main/images/framework.png">
</p>

## Driving-scene adaptation

The `road_ads/` extension evaluates ADS-DGW on five driving transfers with two
explicit protocols and two segmentor families:

- `semi`: labeled and unlabeled target images only.
- `ssda`: the same ADS objective plus one independently sampled supervised
  source batch per iteration. No TC-ADA initialization, mixing, or loss schedule
  is transferred to ADS.
- `native`: the official DeepLabV2-ResNet101 setup (`80k`, batch 4,
  `256x512`, SGD) and the published ADS Cityscapes hyperparameters.
- `vfm`: DINOv3-B + ReIN + HRDA under the common comparison setup (`40k`,
  batch 2, `1024x1024`, AdamW).

The five tasks are GTA5-to-Cityscapes, SYNTHIA-to-Cityscapes,
Cityscapes-to-ACDC, Cityscapes-to-MUSES, and Cityscapes-to-Mapillary. The first
four use the `1/64` target split; Mapillary uses `1/128`.

### Environment and assets

The extension is tested in the existing `reinpy10` Conda environment. After
cloning this repository next to the SSDA project, link the datasets, splits,
and DINOv3 checkpoint:

```bash
conda activate reinpy10
bash scripts/road/setup_assets.sh /path/to/SSDA
python scripts/road/check_setup.py \
  --config configs/road/vfm/ssda/gta2cityscapes.yaml
```

Native experiments additionally require the official ADS checkpoint at
`pretrained/MS_DeepLab_resnet_pretrained_COCO_init.pth`. Dataset links,
pretrained weights, logs, checkpoints, and outputs are ignored by Git.

### Training and evaluation

Run one experiment in a detached `screen` session:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/road/train.sh \
  configs/road/vfm/semi/gta2cityscapes.yaml ads_vfm_semi_g2c

CUDA_VISIBLE_DEVICES=1 bash scripts/road/train.sh \
  configs/road/vfm/ssda/gta2cityscapes.yaml ads_vfm_ssda_g2c
```

Launch all five tasks on five GPUs:

```bash
bash scripts/road/train_five.sh vfm ssda 0,1,2,3,4
```

Evaluate a saved student checkpoint:

```bash
python evaluate_road.py \
  --config configs/road/vfm/ssda/gta2cityscapes.yaml \
  --checkpoint outputs/road/vfm/ssda/gta2cityscapes_1_64/checkpoint_040000.pth
```

Resolved configs, 10k-interval validation records, both student states, and the
best left-student state are stored in each output directory.


## Code

### Environment

- Python >= 3.5
- Pytorch >= 1.1
- NVIDIA Tesla V100

### Test

You can download pretrained weights from [here](https://drive.google.com/drive/folders/1Ch9bUbqToN2hisl3afnCW32qhP12p9SB?usp=sharing) (ADS-DGW_Dataset_SemiRatio_iterXXXXX.pth), then run:
```
bash run_scripts/test_VOC2012.sh
```
### Train

Train baseline:
```
bash run_scripts/train_baseline_VOC2012.sh
```
Train Mean-Teacher with DGW augmentation:
```
bash run_scripts/train_MT_DGW_VOC2012.sh
```
Train ADS with DGW augmentation:
```
bash run_scripts/train_ADS_DGW_VOC2012.sh
```

## Citation

If you find our paper or code helpful in your research or work, please cite our paper:
```
@article{cao2022adversarial,
  title={Adversarial dual-student with differentiable spatial warping for semi-supervised semantic segmentation},
  author={Cao, Cong and Lin, Tianwei and He, Dongliang and Li, Fu and Yue, Huanjing and Yang, Jingyu and Ding, Errui},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={33},
  number={2},
  pages={793--803},
  year={2022},
  publisher={IEEE}
}
```
