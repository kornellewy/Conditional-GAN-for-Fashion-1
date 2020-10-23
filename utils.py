"""
code made by Korneliusz Lewczuk korneliuszlewczuk@gmail.com
sorces: 
https://github.com/cheind/py-thin-plate-spline/blob/master/TPS.ipynb
"""

import os
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable


def load_images(path):
    images = []
    valid_images = [".jpeg", ".jpg", '.png']
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(os.path.join(path, f))
    return images

def denormalize_tensor(normalize_tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    return inv_normalize(normalize_tensor)

