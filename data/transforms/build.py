# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing, LocalCat


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        size = cfg.INPUT.SIZE_TRAIN
        nsize = (size[0],size[1])
        print("nsize:",nsize)
        transform = T.Compose([
            T.Resize(nsize),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(nsize),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
#            LocalCat(2)
        ])
    else:
        size = cfg.INPUT.SIZE_TEST
        nsize = (size[0],size[1])
        transform = T.Compose([
            T.Resize(nsize),
            T.ToTensor(),
            normalize_transform
#            LocalCat(2)
        ])

    return transform
