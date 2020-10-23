"""
code made by Korneliusz Lewczuk korneliuszlewczuk@gmail.com
sorces:
https://arxiv.org/pdf/1709.04695.pdf
"""

import unittest
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
import torch.utils.data as data

from utils import load_images, denormalize_tensor
from dataloaders import DataloaderGenerator, DataloaderDiscriminator
from models import Generator, Discriminator


class TestDataloaderGenerator(unittest.TestCase):
    def setUp(self):
        self.dataset_path = ''
        self.shape=(128,96)

    def test__len__(self):
        test_path = os.path.join(self.dataset_path, '')
        self.assertEqual(len(load_images(test_path)), len(DataloaderGenerator(self.dataset_path)))

    def test__getitem__(self):
        dataloader = DataloaderGenerator(self.dataset_path)
        cloth, person_with_cloth, person_with_cloth_mask, target_cloth = dataloader[len(dataloader)-1]

        self.assertEqual(cloth.size(), (3, self.shape[0], self.shape[1]))
        self.assertEqual(person_with_cloth.size(), (3, self.shape[0], self.shape[1]))
        self.assertEqual(target_cloth.size(), (3, self.shape[0], self.shape[1]))
        self.assertEqual(person_with_cloth_mask.size(), (1, self.shape[0], self.shape[1]))

        cloth = denormalize_tensor(cloth)
        person_with_cloth = denormalize_tensor(person_with_cloth)
        target_cloth = denormalize_tensor(target_cloth)

        save_image(cloth, "")
        save_image(person_with_cloth, "")
        save_image(target_cloth, "")
        save_image(person_with_cloth_mask, "")


class TestDataloaderDiscriminator(unittest.TestCase):
    def setUp(self):
        self.dataset_path = ''
        self.shape=(128,96)

    def test__len__(self):
        test_path = os.path.join(self.dataset_path, '')
        self.assertEqual(len(load_images(test_path)), len(DataloaderDiscriminator(self.dataset_path)))

    def test__getitem__(self):
        dataloader = DataloaderDiscriminator(self.dataset_path)
        cloth, person_with_cloth = dataloader[5]

        self.assertEqual(cloth.size(), (3, self.shape[0], self.shape[1]))
        self.assertEqual(person_with_cloth.size(), (3, self.shape[0], self.shape[1]))

        cloth = denormalize_tensor(cloth)
        person_with_cloth = denormalize_tensor(person_with_cloth)

        save_image(cloth, "")
        save_image(person_with_cloth, "")


class TestGenerator(unittest.TestCase):
    def test_forward(self):
        sample = torch.randn(1, 9, 128, 96)
        model = Generator()
        sample_output = model.forward(sample)
        self.assertEqual(sample_output.size(), (1, 4, 128, 96))


class TestDiscriminator(unittest.TestCase):
    def test_forward(self):
        sample = torch.randn(1, 6, 128, 96)
        model = Discriminator()
        sample_output = model.forward(sample)
        self.assertEqual(sample_output.size(), (1, 1))

if __name__ == '__main__':
    unittest.main()
