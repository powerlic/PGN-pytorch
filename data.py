from torchvision import datasets, transforms
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import custom_transforms as trans

img_transform = transforms.Compose([
    trans.RandomHorizontalFlip(),
    trans.RandomGaussianBlur(),
    trans.RandomScaleCrop(700,512),
    trans.Normalize(),

    trans.ToTensor()
])


class TrainImageFolder(data.Dataset):
    def __init__(self,data_dir):
        self.f = open(os.path.join(data_dir,'train_id.txt'))
        self.file_list=self.f.readlines()
        self.data_dir=data_dir

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir,'train_images',self.file_list[index][:-1]+'.jpg')).convert('RGB')
        parse = Image.open(os.path.join(self.data_dir, 'train_segmentations', self.file_list[index][:-1]) + '.png')
        sample={'image':img,'label':parse}
        sample=img_transform(sample)
        return sample['image'],sample['label']

    def __len__(self):
        return len(self.file_list)

class ValImageFolder(data.Dataset):
    def __init__(self,data_dir):
        self.f = open(os.path.join(data_dir,'val_id.txt'))
        self.file_list=self.f.readlines()
        self.data_dir=data_dir

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir,'val_images',self.file_list[index]+'.png'))
        parse = Image.open(os.path.join(self.data_dir, 'val_segmentations', self.file_list[index] + '.png'))
        sample = {'image': img, 'label': parse}
        sample = img_transform(sample)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.file_list)

