# ESRGANでDIV2Kデータを学習させる
# DIV2Kの教師データ数は800、検証データ数は100

#
# https://qiita.com/pacifinapacific/items/ec338a500015ae8c33fe
# https://qiita.com/AokiMasataka/items/bfb5e338079f01bfc996


import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
# import matplotlib.pyplot as plt

class PairImage(ImageFolder):
    def __init__(self, root, transform = None, large_size = 256, small_size = 64, **kwds):
        super().__init__(root, transform=transform, **kwds)
        self.large_resizer = transforms.Scale(large_size)
        self.small_resizer = transforms.Scale(small_size)

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loder(path)
        large_img = self.large_resizer(img)
        small_img = self.small_resizer(img)
        return large_img, small_img

class SR_ADV():
    def __init__(self):
        train_data = PairImage('G:/DIV2K/DIV2K_train_HR', transforms = transforms.ToTensor())
        test_data = PairImage('G:/DIV2K/DIV2K_valid_HR', transforms = transforms.ToTensor())
        batch_size = 8
        train_loader = DataLoader(train_data, batch_size, shuffle = True)
        test_loader = DataLoader(train_data, batch_size, shuffle = True)

class ResidualBlock(nn.Module):
    def __init__(self, nf = 64):
        super(ResidualBlock, self).__init__()
        self.Block = nn.Sequentia(
            nn.Conv2d(nf, nf, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(nf),
            nn.PReLU(),
            nn.Conv2d(nf, nf, kernel_ize = 3, padding = 1),
            nn.BatchNorm2d(nf),
        )
    def forward(self, x):
        return x + self.Block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.COnv2d(3, 64, kernel_size = 9, padding = 4)
        self.relu = nn.PReLU()

        self.residualLayer = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

        self.pixelShuffle = nn.Swquential(
            nn.Conv2d(64, 64*4, kernel_size = 3, padding  = 1),
            nn.PReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 4, kernel_size = 8, padding = 4),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        skip = self.relu(x)
        x = self.residualLayer(skip)
        x = self.pixelShuffle(x + skip)

class Discriminator(nn.Module):
    def __init__(self, size = 64):
        super(Discriminator, self).__init__()
        size = int(size / 8) ** 2

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.2)
        )

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg16(pretrained = True)
