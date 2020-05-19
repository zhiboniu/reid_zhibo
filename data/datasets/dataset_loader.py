# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset

def pucutimg(img, maxs = 2, linenum = 3):
    lines = [x*1.0/(linenum+1) for x in range(1, linenum+1)]
    scale = random.uniform(1, self.maxs)
    c,h,w = img.size()
    nh = h*scale
    lines = [x*scale for x in lines if x*scale<=1.0 else -1]
    img = img.resize(w,nh)
    nimg = img.crop(0,0,w,h)
    return nimg, lines

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
        img,lines = pucutimg(img)
    return img,lines


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img,lines = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, lines, img_path
