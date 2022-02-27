'''
    SR_ADV | train_set.py

    image set

-----
    memo

    https://chowdera.com/2021/12/202112300421271916.html
'''

# ----- module
import os
import cv2
from pathlib import Path
import random
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import InterpolationMode
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ----- config
import set_e
settings = set_e.Settings()
opt = set_e.Param()
random.seed(settings.seed)
torch.manual_seed(seed=settings.seed)
torch.cuda.manual_seed_all(settings.seed)

# ----- data load
class ImageDataset(Dataset):
    def __init__(self, dataset_dir, hr_shape):
        hr_height, hr_width = hr_shape

        # self.lr_transform = transforms.Compose([
        #     transforms.Resize((hr_height // 4, hr_height // 4), interpolation=InterpolationMode.BICUBIC),
        #     transforms.RandomHorizontalFlip(opt.random_flip),
        #     transforms.GaussianBlur(kernel_size=opt.random_blur_kernel, sigma=opt.random_blur_sigma),
        #     transforms.ToTensor(),
        #     transforms.Normalize(opt.mean, opt.std),
        #     ])
        self.lr_transform = A.Compose([
            A.Resize(hr_height // 4, hr_width // 4, interpolation=cv2.INTER_CUBIC),
            A.JpegCompression(
                quality_lower=opt.random_jpg_noise_quality[0],
                quality_upper=opt.random_jpg_noise_quality[1],
                p=opt.random_jpg_noise_p),
            A.GaussianBlur(p=opt.random_blur_p),
            A.HorizontalFlip(p=opt.random_flip),
            A.Normalize(mean=opt.mean, std=opt.std),
            ToTensorV2(),
        ])

        # self.hr_transform = transforms.Compose([
        #     transforms.Resize((hr_height, hr_height), interpolation=InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(opt.mean, opt.std),
        #     ])
        self.hr_transform = A.Compose([
            A.Resize(hr_height, hr_width, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=opt.mean, std=opt.std),
            ToTensorV2(),
        ])

        p = Path(settings.image_dir_save)
        self.files = sorted(p.glob('*.'+settings.save_img_format))
        # print(self.files)

    def __getitem__(self, index):
        # img = Image.open(self.files[index%len(self.files)])
        img = cv2.imread(str(self.files[index%len(self.files)]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lr = self.lr_transform(image=img)['image']
        img_hr = self.hr_transform(image=img)['image']
        # print(type(img_lr))
        # print(img_lr.shape)
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
        # self.hr_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(opt.mean, opt.std),
        #     ])
        self.hr_transform = A.Compose([
            A.Normalize(mean=opt.mean, std=opt.std),
            ToTensorV2(),
        ])
        p = Path(dataset_dir)
        self.files = sorted([i for i in p.glob('*') if os.path.isfile(i)])
        # self.files = sorted(p.glob('*'))
        # self.files = sorted(glob(Path(dataset_dir).joinpath('*')))

    def lr_transform(self, image, img_size):
        (img_width, img_height) = map(int, img_size)

        # self.__lr_transform = transforms.Compose([
        #     transforms.Resize((img_height // 4, img_width // 4), interpolation=InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(opt.mean, opt.std),
        #     ])
        self.__lr_transform = A.Compose([
            A.Resize(img_height // 4, img_width // 4, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=opt.mean, std=opt.std),
            ToTensorV2(),
        ])
        img = self.__lr_transform(image=image)
        return img

    def __getitem__(self, index):
        # img = Image.open(self.files[index%len(self.files)])
        # img_size = img.size
        # img_lr = self.lr_transform(img, img_size)
        # img_hr = self.hr_transform(img)
        img = cv2.imread(str(self.files[index%len(self.files)]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        img_lr = self.lr_transform(image=img, img_size=(w, h))['image']
        img_hr = self.hr_transform(image=img)['image']
        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self): return len(self.files)

# ----- test
if __name__ == '__main__':
    train_dataloader = ImageDataset(settings.image_dir_save, (128, 128))
    img = train_dataloader[0]
    print(type(img['lr']))
    # print(img['lr'])
