import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random
import glob
import os
from PIL import Image

from utils import NIPS_DATA, load_ground_truth

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path=NIPS_DATA, part=1, part_index=1):
        image_id_list, self.label_ori_list, self.label_tar_list = load_ground_truth()
        self.image_paths = []
        for image_id in image_id_list:
            path = os.path.join(data_path, 'images', '{}.png'.format(image_id))
            self.image_paths.append(path)
        if part == 1:
            pass
        else:
            length = len(self.image_paths)
            part_len = int(length / part)
            self.image_paths = self.image_paths[(part_index-1)*part_len:part_index*part_len]
            self.label_ori_list = self.label_ori_list[(part_index-1)*part_len:part_index*part_len]
            self.label_tar_list = self.label_tar_list[(part_index-1)*part_len:part_index*part_len]

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            ])        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)
        image = self.transforms(image)
        return image, self.label_ori_list[idx], self.label_tar_list[idx]