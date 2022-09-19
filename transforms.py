import math
from enum import Enum
from typing import List, Tuple, Optional, Dict

import torch
import torchvision
import copy
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import numpy as np

# refer to randaugment in pytorch

def augmentation_space(num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
    return {
        # op_name: (magnitudes, signed)
        # RandAugment
        "ShearX": (torch.linspace(0.0, 180.0, num_bins), True),
        "ShearY": (torch.linspace(0.0, 180.0, num_bins), True),
        "TranslateX": (torch.linspace(0.0, 1.0 * image_size[1], num_bins), True),
        "TranslateY": (torch.linspace(0.0, 1.0 * image_size[0], num_bins), True),
        "Rotate": (torch.linspace(0.0, 180.0, num_bins), True),
        "Brightness": (torch.linspace(0.0, 5.0, num_bins), False),
        "Color": (torch.linspace(0.0, 5.0, num_bins), False),
        "Contrast": (torch.linspace(0.0, 5.0, num_bins), False),
        "Sharpness": (torch.linspace(0.0, 5.0, num_bins), False),
        "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
         
        "Crop": (torch.linspace(1.0, 0.0, num_bins), False),
        "Flip": (torch.linspace(0.0, 1.0, num_bins), False),
        'DI': (torch.linspace(299, 700, num_bins), False)
        }

def ColorRelated(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
    ):
    '''
    Brightness, Color, Contrast, Sharpness, Solarize, AutoContrast, Scale (SI attack)
    refer to https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#RandAugment
    '''
    if op_name == 'Brightness':
        img_out = F.adjust_brightness(img, magnitude)
    elif op_name == "Color":
        img_out = F.adjust_saturation(img, magnitude)
    elif op_name == "Contrast":
        img_out = F.adjust_contrast(img, magnitude)
    elif op_name == "Sharpness":
        img_out = F.adjust_sharpness(img, magnitude)
    elif op_name == "Solarize":
        img_out = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img_out = F.autocontrast(img)
    elif op_name == 'Identity':
        img_out = img
    return img_out

def PositionRelated(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
    ):
    '''
    ShearX, ShearY, TranslateX, TranslateY, Rotate, Crop, Cutout, Flip
    '''
    if op_name == "ShearX":
        img_out = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[magnitude, 0.0],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img_out = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, magnitude],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "TranslateX":
        img_out = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img_out = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img_out = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == 'Crop':
        crop = torchvision.transforms.RandomResizedCrop(299, scale=(magnitude, 1.0), ratio=(1.,1.))
        img_out = crop(img)
    elif op_name == 'Flip':
        c = np.random.rand(1)[0]
        if c <= magnitude:
            img_out = F.hflip(img)
        else:
            img_out = img.clone()
        c = np.random.rand(1)[0]
        if c <= magnitude:
            img_out = F.vflip(img_out)
    elif op_name == 'DI':
        if int(magnitude) == 299:
            img_out = img.clone()
        else:
            rnd = np.random.randint(299, int(magnitude),size=1)[0]
            h_rem = int(magnitude) - rnd
            w_rem = int(magnitude) - rnd
            pad_top = np.random.randint(0, h_rem,size=1)[0]
            pad_bottom = h_rem - pad_top
            pad_left = np.random.randint(0, w_rem,size=1)[0]
            pad_right = w_rem - pad_left
            img_out = torch.nn.functional.pad(torch.nn.functional.interpolate(img, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
    return img_out

class SingleTransforms(torch.nn.Module):
    def __init__(
        self,
        opt_name,
        magnitude: int = 4,
        num_magnitude_bins: int = 11,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.opt_name = opt_name
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.image_size = (299, 299)
        self.magnitudes, self.signed = augmentation_space(self.num_magnitude_bins, self.image_size)[self.opt_name]

        self.left = float(self.magnitudes[0].item()) if self.magnitudes.ndim > 0 else 0.0
        self.right = float(self.magnitudes[self.magnitude].item()) if self.magnitudes.ndim > 0 else 0.0

    def _random_magnitude(self):                
        if self.opt_name in ['Solarize', 'Crop']:
            diff = self.left - self.right
            rand_para = diff * np.random.rand(1)[0] + self.right
        else:
            diff = self.right - self.left
            rand_para = diff * np.random.rand(1)[0] + self.left

        if self.signed and torch.randint(2, (1,)):
            rand_para *= -1.0
        return rand_para

    def forward(self, img:Tensor) -> Tensor:
        if self.opt_name in ['Brightness', 'Color', 'Contrast', 'Sharpness', 'Solarize', 'AutoContrast', 'Scale']:
            rand_mag = self._random_magnitude()
            img_out = ColorRelated(img, self.opt_name, rand_mag, interpolation=self.interpolation, fill=self.fill)
        else:
            if self.opt_name in ['Crop', 'Flip', 'DI']:
                img_out = PositionRelated(img, self.opt_name, self.right, interpolation=self.interpolation, fill=self.fill)
            else:
                rand_mag = self._random_magnitude()
                img_out = PositionRelated(img, self.opt_name, rand_mag, interpolation=self.interpolation, fill=self.fill)
        return img_out

def attack_augmentation_space(num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
    return {
        # op_name: (magnitudes, signed)
        # RandAugment
        "ShearX": (torch.linspace(0.0, 180.0, num_bins), True),
        "ShearY": (torch.linspace(0.0, 180.0, num_bins), True),
        "TranslateX": (torch.linspace(0.0, 1.0 * image_size[1], num_bins), True),
        "Flip": (torch.linspace(0.0, 1.0, num_bins), False),
        "TranslateY": (torch.linspace(0.0, 1.0 * image_size[0], num_bins), True),

        "Rotate": (torch.linspace(0.0, 180.0, num_bins), True),
        "Crop": (torch.linspace(1.0, 0.01, num_bins), False),        
        'DI': (torch.linspace(299, 700, num_bins), False),
        }

class TargetAdvAugment(torch.nn.Module):
    def __init__(
        self,
        opt_nums: int = 2,
        magnitude: int = 4,
        pop_opts: list = [],
        num_magnitude_bins: int = 11,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.opt_nums = opt_nums
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.image_size = (299, 299)
        self.op_meta = attack_augmentation_space(self.num_magnitude_bins, self.image_size)

        # delete opts
        for pop_opt in pop_opts:
            self.op_meta.pop(pop_opt)
        print ('Used Augmentation:', self.op_meta.keys())

    def _mag_para(self, opt_name, magnitudes, signed):
        left = float(magnitudes[0].item()) if magnitudes.ndim > 0 else 0.0
        right = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
        if opt_name in ['Solarize', 'Crop']:
            diff = left - right
            rand_para = diff * np.random.rand(1)[0] + right
        else:
            diff = right - left
            rand_para = diff * np.random.rand(1)[0] + left

        if signed and torch.randint(2, (1,)):
            rand_para *= -1.0
        
        return rand_para, right

    def forward(self, img:Tensor) -> Tensor:
        forward_img = img.clone()
        opt_info = {}
        for _ in range(self.opt_nums):
            op_index = int(torch.randint(len(self.op_meta), (1,)).item())
            op_name = list(self.op_meta.keys())[op_index]
            magnitudes, signed = self.op_meta[op_name]
            para, right = self._mag_para(op_name, magnitudes, signed)
            if op_name in ['Brightness', 'Color', 'Contrast', 'Sharpness', 'Solarize', 'AutoContrast', 'Scale', 'Identity']:
                forward_img = ColorRelated(forward_img, op_name, para, interpolation=self.interpolation, fill=self.fill)
                opt_info[op_name] = para
            else:
                if op_name in ['Crop', 'Flip', 'DI']:
                    forward_img = PositionRelated(forward_img, op_name, right, interpolation=self.interpolation, fill=self.fill)
                    opt_info[op_name] = right
                else:
                    forward_img = PositionRelated(forward_img, op_name, para, interpolation=self.interpolation, fill=self.fill)
                    opt_info[op_name] = para
        return forward_img