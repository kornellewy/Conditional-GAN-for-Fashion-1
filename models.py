"""
code made by Korneliusz Lewczuk korneliuszlewczuk@gmail.com
sorces: 
https://arxiv.org/pdf/1709.04695.pdf
https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/

TODO: uzyc nn.LeakyReLU(0.2, True)
TODO: uzyc resneta zamiast unet
TODO: wiecej warst po drodze
TODO: zobaczyc batchnorm
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        # 9x128x96
        self.conv1 = nn.Conv2d(9, 64, kernel_size=4, stride=2, padding=1)
        # 64x64x48
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.instancenorm1 = nn.InstanceNorm2d(num_features=128)
        # 128x32x24
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.instancenorm2 = nn.InstanceNorm2d(num_features=256)
        # 256x16x12
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.instancenorm3 = nn.InstanceNorm2d(num_features=512)
        # half of nnet in midle we have 512x8x6 

        # 512x8x6
        self.convtrans1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.instancenorm4 = nn.InstanceNorm2d(num_features=256)
        # now we stack 256x16x12 with output form conv4 256x16x12 to 512x16x12
        self.convtrans2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.instancenorm5 = nn.InstanceNorm2d(num_features=128)

        self.convtrans3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.instancenorm6 = nn.InstanceNorm2d(num_features=64)

        self.convtrans4 = nn.ConvTranspose2d(128, 4, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x_2 = x
        x = self.conv2(x)
        x = self.instancenorm1(x)
        x = self.relu(x)
        
        x_3 = x
        x = self.conv3(x)
        x = self.instancenorm2(x)
        x = self.relu(x)

        x_4 = x
        x = self.conv4(x)
        x = self.instancenorm3(x)
        x = self.relu(x)

        x = self.convtrans1(x)
        x = self.instancenorm4(x)
        x = self.relu(x)
        x = torch.cat([x, x_4], 1)

        x = self.convtrans2(x)
        x = self.instancenorm5(x)
        x = self.relu(x)
        x = torch.cat([x, x_3], 1)

        x = self.convtrans3(x)
        x = self.instancenorm6(x)
        x = self.relu(x)
        x = torch.cat([x, x_2], 1)

        x = self.convtrans4(x)
        x = self.relu(x)

        return x

"""
We mast use PatchGAN so i use the info from:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L318
https://arxiv.org/ftp/arxiv/papers/1902/1902.00536.pdf
I jast use conv and mean as patchgan.
"""

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        # last activation sigmoid
        self.sigmoid = nn.Sigmoid()
        # 9x128x96
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1)
        # 64x64x48
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.instancenorm1 = nn.InstanceNorm2d(num_features=128)
        # 128x32x24
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.instancenorm2 = nn.InstanceNorm2d(num_features=256)
        # 256x16x12
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.instancenorm3 = nn.InstanceNorm2d(num_features=512)
        # half of nnet in midle we have 512x8x6 
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)
        self.instancenorm4 = nn.InstanceNorm2d(num_features=512)
        # 1x4x3 
        self.conv6 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.instancenorm5 = nn.InstanceNorm2d(num_features=512)
        # 1x2x2 
        self.conv7 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1)
        # 1x1x1 

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.instancenorm1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.instancenorm2(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.instancenorm3(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.instancenorm4(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.instancenorm5(x)
        x = self.relu(x)
        
        x = self.conv7(x)
        x = self.sigmoid(x)
        # x = torch.mean(x)
        # if x > 0.5:
        #     x = torch.Tensor([1])
        # else:
        #     x = torch.Tensor([0])
        x = torch.squeeze(x, -1)
        x = torch.squeeze(x, -1)
        return x