#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class BaseDepthDataset(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train', norm={'mean':(123.675, 116.28, 103.53), 'std':(58.395, 57.12, 57.375)}, return_img_name=False):
        super(BaseDepthDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode  # 
        self.trans_func = trans_func

        self.ignore_index = 255
        self.lb_map = None
        self.norm_cfg = norm

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths, self.de_paths = [], [], []
        for pair in pairs:
            imgpth, lbpth, depth = pair.split(',')
            self.img_paths.append(os.path.join(dataroot, imgpth))
            self.lb_paths.append(os.path.join(dataroot, lbpth))
            self.de_paths.append(os.path.join(dataroot, depth))

        assert len(self.img_paths) == len(self.lb_paths)
        assert len(self.img_paths) == len(self.de_paths)
        
        self.len = len(self.img_paths)
        self.return_img_name = return_img_name

    def __getitem__(self, idx):
        impth, lbpth, depth = self.img_paths[idx], self.lb_paths[idx], self.de_paths[idx]
        img, label, depth = self.get_img_lb_dep(impth, lbpth, depth)
        if self.lb_map is not None:
            label = self.lb_map[label]
        im_lb_dep = dict(im=img, lb=label, dep=depth)
        if self.trans_func is not None:  # augment the image and label
            im_lb_dep = self.trans_func(im_lb_dep)

        im, lb, dep = im_lb_dep['im'], im_lb_dep['lb'], im_lb_dep['dep']
        im = im[:, :, ::-1].copy()  # BGR to RGB
        im = self.norm_img(im)  # normalize the image
        im = torch.from_numpy(im.transpose(2, 0, 1).copy()).clone()  # HWC to CHW
        lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        dep = torch.from_numpy(dep.astype(np.float32).copy()).clone()
        
        if self.return_img_name:
            im_name = os.path.basename(impth)
            return im, lb, dep, im_name
        else:
            return im, lb, dep
        
    def norm_img(self, img):
        img = img.astype(np.float32)
        mean = np.float64(np.array(self.norm_cfg['mean']).reshape(1, -1))
        stdinv = 1 / np.float64(np.array(self.norm_cfg['std']).reshape(1, -1))
        cv2.subtract(img, mean, img)
        cv2.multiply(img, stdinv, img)
        # img -= self.norm_cfg['mean']
        # img /= self.norm_cfg['std']
        return img

    def get_img_lb_dep(self, impth, lbpth, depth):
        img = cv2.imread(impth)[:, :, :].copy() 
        # img = cv2.imread(impth)[:, :, ::-1].copy()  # BGR to RGB
        label = cv2.imread(lbpth, 0)  # read in gray mode
        depth = cv2.imread(depth, 0)  # read in gray mode
        return img, label, depth

    def __len__(self):
        return self.len


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = CityScapes('./data/', mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
