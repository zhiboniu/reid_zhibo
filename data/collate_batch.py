# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, lines, _ = zip(*batch)
    lines = torch.tensor(lines, dtype=torch.int64)
    return torch.stack(imgs, dim=0), lines


def val_collate_fn(batch):
    imgs, lines, _ = zip(*batch)
    return torch.stack(imgs, dim=0), lines
