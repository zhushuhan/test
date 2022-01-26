from PIL import Image

import os
import numpy as np
import torch 
from torch.utils.data import Dataset

class BrainDataset_slice(Dataset):
    """ define dataset """
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self,item):
        img = Image.open(self.images_path[item]) # (H, W, C)
        label = self.images_class[item]
        if self.transform is not None :
            img = self.transform(img)
        
        return img, label

class BrainDataset_sample(Dataset):
    """ define dataset """
    def __init__(self, samples_path: list, samples_class: list, mode=None, transform=None):
        self.samples_path = samples_path
        self.samples_class = samples_class
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.samples_path)

    def __getitem__(self,item):
        sample_path = self.samples_path[item]
        imgs_path = os.listdir(sample_path) # 获取当前sample内的slice名称
        slice_num = len(imgs_path) # 获取当前sample的slice数目
        if self.mode == 'train': # 如果是train，随机取sample中的slice
            x = np.random.randint(slice_num) # 产生[0, slice_num)的随机数
            img_path = os.path.join(sample_path, imgs_path[x])
            img = Image.open(img_path) # (H, W, C)
            label = self.samples_class[item]
        else: #如果是val，取sample中的16.jpg，因为16.jpg是标注的slice
            img_path = os.path.join(sample_path, '16.jpg')
            img = Image.open(img_path) # (H, W, C)
            label = self.samples_class[item]
        if self.transform is not None :
            img = self.transform(img)
        return img, label

    # def __getitem__(self,item):
    #     sample_path = self.samples_path[item]
    #     imgs_path = os.listdir(sample_path) # 获取当前sample内的slice名称
    #     slice_num = len(imgs_path) # 获取当前sample的slice数目
    #     if self.mode == 'train': # 如果是train，随机取sample中的slice
    #         x = np.random.randint(slice_num) # 产生[0, slice_num)的随机数
    #         img_path = os.path.join(sample_path, imgs_path[x])
    #         img = Image.open(img_path) # (H, W, C)
    #         label = self.samples_class[item]
    #     else: #如果是val，取sample中的16.jpg，因为16.jpg是标注的slice
    #         img_path = os.path.join(sample_path, '16.jpg')
    #         img = Image.open(img_path) # (H, W, C)
    #         label = self.samples_class[item]
    #     if self.transform is not None :
    #         img = self.transform(img)
    #     return img, label
