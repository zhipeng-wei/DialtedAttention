import torch
import torch.nn as nn
import csv
import os

# these two variables need to be specified.
OPT_PATH = '/pscratch/sd/z/zpwei/TargetedAttack/output'
NIPS_DATA = '/pscratch/sd/z/zpwei/TargetedAttack/nips-dataset'

class Normalize(nn.Module):
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

    def forward(self, x):
        return (x - self.mean.to(x.device)[None, :, None, None]
                ) / self.std.to(x.device)[None, :, None, None]

def load_ground_truth(csv_filename=os.path.join(NIPS_DATA, 'images.csv')):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_ori_list.append( int(row['TrueLabel']) - 1 )
            label_tar_list.append( int(row['TargetClass']) - 1 )

    return image_id_list, label_ori_list, label_tar_list