import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random
import glob
import os
from PIL import Image

from utils import NIPS_DATA, load_ground_truth, IMAGENET_VAL

class NIPSDataset(torch.utils.data.Dataset):
    '''
    Randomly sample 15000 examples from ImageNet Validation Dataset.
    They are used as attacked examples.
    '''
    def __init__(self, subset=False, data_path=NIPS_DATA):
        self.subset = subset
        image_id_list, f_label_ori_list, f_label_tar_list = load_ground_truth()
        f_image_paths = []
        for image_id in image_id_list:
            path = os.path.join(data_path, 'images', '{}.png'.format(image_id))
            f_image_paths.append(path)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            ])        
        
        if self.subset:
            inds = [i for i in range(len(f_image_paths))]
            random.seed(1024)
            random.shuffle(inds)
            used_inds = inds[:200]
            self.label_ori_list = []
            self.label_tar_list = []
            self.image_paths = []
            for ind in used_inds:
                self.label_ori_list.append(f_label_ori_list[ind])
                self.label_tar_list.append(f_label_tar_list[ind])
                self.image_paths.append(f_image_paths[ind])
        else:
            self.label_ori_list = f_label_ori_list
            self.label_tar_list = f_label_tar_list
            self.image_paths = f_image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)
        image = self.transforms(image)
        return image, self.label_ori_list[idx], self.label_tar_list[idx]

class ClassSamples5000(torch.utils.data.Dataset):
    '''
    Randomly sample 5000 examples from ImageNet Validation Dataset.
    They are used to evaluate the IoUs between images with different augmentation.
    '''
    def __init__(self, valdir=IMAGENET_VAL):
        normalize = transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1., 1., 1.])
        self.val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Scale(256),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize,]))
        inds = [i for i in range(len(self.val_dataset))]
        random.seed(1024)
        random.shuffle(inds)
        self.inds = inds[:5000]
        
    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        image, label = self.val_dataset[self.inds[idx]]
        target_label = [i for i in range(1000) if i!=label]
        random.seed(idx)
        random.shuffle(target_label)
        target_label = target_label[0]
        return image, label, target_label