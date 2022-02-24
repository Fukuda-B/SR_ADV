'''
    SR_ADV | train_set.py

    image set

-----
    memo

    https://chowdera.com/2021/12/202112300421271916.html
'''

# ----- module
import os
from pathlib import Path
import random
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import InterpolationMode

# ----- config
import set_e
settings = set_e.Settings()
import train
opt = train.Param()
random.seed(settings.seed)
torch.manual_seed(seed=settings.seed)
torch.cuda.manual_seed_all(settings.seed)

# ----- data load
class ImageDataset(Dataset):
    def __init__(self, dataset_dir, hr_shape):
        hr_height, hr_width = hr_shape

        self.lr_transform = transforms.Compose([
            transforms.Resize((hr_height // 4, hr_height // 4), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(opt.random_flip),
            # transforms.GaussianBlur(kernel_size=opt.random_blur_kernel, sigma=opt.random_blur_sigma),
            transforms.ToTensor(),
            transforms.Normalize(opt.mean, opt.std),
            ])

        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_height, hr_height), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(opt.mean, opt.std),
            ])

        p = Path(settings.image_dir_save)
        self.files = sorted(p.glob('*.'+settings.save_img_format))
        # print(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index%len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self): return len(self.files)

class AsImageDataset(Dataset):
    def __init__(self, dataset_dir):
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(opt.mean, opt.std),
            ])
        p = Path(dataset_dir)
        self.files = sorted([i for i in p.glob('*') if os.path.isfile(i)])

    def lr_transform(self, img, img_size):
        img_width, img_height = img_size

        self.__lr_transform = transforms.Compose([
            transforms.Resize((img_height, img_width), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(opt.mean, opt.std),
            ])
        img = self.__lr_transform(img)
        return img

    def __getitem__(self, index):
        img = Image.open(self.files[index%len(self.files)])
        img_size = img.size
        img_lr = self.lr_transform(img, img_size)
        img_hr = self.hr_transform(img)
        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self): return len(self.files)

class TestImageDataset(Dataset):
    def __init__(self, dataset_dir):
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(opt.mean, opt.std),
            ])
        p = Path(dataset_dir)
        self.files = sorted([i for i in p.glob('*') if os.path.isfile(i)])
        # self.files = sorted(p.glob('*'))
        # self.files = sorted(glob(Path(dataset_dir).joinpath('*')))

    def lr_transform(self, img, img_size):
        img_width, img_height = img_size

        self.__lr_transform = transforms.Compose([
            transforms.Resize((img_height // 4, img_width // 4), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(opt.mean, opt.std),
            ])
        img = self.__lr_transform(img)
        return img

    def __getitem__(self, index):
        img = Image.open(self.files[index%len(self.files)])
        img_size = img.size
        img_lr = self.lr_transform(img, img_size)
        img_hr = self.hr_transform(img)
        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self): return len(self.files)

# ----- test
if __name__ == '__main__':
    train_dataloader = ImageDataset(settings.image_dir_save, (128, 128))
    img = train_dataloader[0]
    print(img['lr'])
