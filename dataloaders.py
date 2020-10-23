"""
code made by Korneliusz Lewczuk korneliuszlewczuk@gmail.com
sorces: 
https://arxiv.org/pdf/1709.04695.pdf
"""

import torch
import os
from PIL import Image
import numpy as np
import random
from torchvision import transforms

from utils import load_images 


class DataloaderGenerator(object):
    def __init__(self, dataset_path, shape=(128,96)):
        super().__init__()
        self.dataset_path = dataset_path

        self.cloth_dir_name = 'cloth'
        self.cloth_dir_path = os.path.join(self.dataset_path, self.cloth_dir_name)
        self.cloth_paths = load_images(self.cloth_dir_path)
        # print("len(self.cloth_paths): ", len(self.cloth_paths))

        self.cloth_mask_dir_name = 'tshirt-mask'
        self.cloth_mask_dir_path = os.path.join(self.dataset_path, self.cloth_mask_dir_name)
        self.cloth_mask_paths = load_images(self.cloth_mask_dir_path)

        self.person_with_cloth_dir_name = 'image'
        self.person_with_cloth_dir_path = os.path.join(self.dataset_path,
                                                        self.person_with_cloth_dir_name)
        self.person_with_cloth_paths = load_images(self.person_with_cloth_dir_path)
        # print("len(self.person_with_cloth_paths): ", len(self.person_with_cloth_paths))

        self.normalize_img = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
            )

        self.normalize_mask = transforms.Normalize(
            [0.5], [0.5]
            )

        self.transform_img = transforms.Compose([
            transforms.Resize(shape),
            transforms.CenterCrop(shape),
            transforms.ToTensor(),
            self.normalize_img])

        self.transform_mask = transforms.Compose([
            transforms.Resize(shape),
            transforms.CenterCrop(shape),
            transforms.ToTensor(),
            self.normalize_mask])

    def __len__(self):
        return len(self.cloth_paths)

    def __getitem__(self, idx):
        cloth_path = self.cloth_paths[idx]
        person_with_cloth_path = self.person_with_cloth_paths[idx]
        person_with_cloth_mask_path = self.cloth_mask_paths[idx]

        target_cloth_idx = random.randint(0, len(self.cloth_paths)-1)
        target_cloth_path = self.cloth_paths[target_cloth_idx]

        cloth = Image.open(cloth_path)
        person_with_cloth = Image.open(person_with_cloth_path)
        person_with_cloth_mask = Image.open(person_with_cloth_mask_path)
        target_cloth = Image.open(target_cloth_path)
        
        cloth = self.transform_img(cloth)
        person_with_cloth = self.transform_img(person_with_cloth)
        target_cloth = self.transform_img(target_cloth)
        person_with_cloth_mask = self.transform_mask(person_with_cloth_mask)

        return cloth, person_with_cloth, person_with_cloth_mask, target_cloth


class DataloaderDiscriminator(object):
    def __init__(self, dataset_path, shape=(128,96)):
        super().__init__()
        self.dataset_path = dataset_path
        self.cloth_dir_name = 'cloth'
        self.cloth_dir_path = os.path.join(self.dataset_path, self.cloth_dir_name)
        self.cloth_paths = load_images(self.cloth_dir_path)
        # print("len(self.cloth_paths): ", len(self.cloth_paths))

        self.person_with_cloth_dir_name = 'image'
        self.person_with_cloth_dir_path = os.path.join(self.dataset_path,
                                                        self.person_with_cloth_dir_name)
        self.person_with_cloth_paths = load_images(self.person_with_cloth_dir_path)
        # print("len(self.person_with_cloth_paths): ", len(self.person_with_cloth_paths))

        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
            )
        self.transform_img = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(shape),
            transforms.CenterCrop(shape),
            transforms.ToTensor(),
            self.normalize])

    def __len__(self):
        return len(self.cloth_paths)

    def __getitem__(self, idx):
        # only return real pair
        cloth_path = self.cloth_paths[idx]
        person_with_cloth_path = self.person_with_cloth_paths[idx]

        cloth = Image.open(cloth_path)
        person_with_cloth = Image.open(person_with_cloth_path)
    
        cloth = self.transform_img(cloth)
        person_with_cloth = self.transform_img(person_with_cloth)

        return cloth, person_with_cloth