'''
    SR_ADV | sr_adv.py

    generate images using weights
'''

# ---- module
import os
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from train_set import TestImageDataset
from train_set import AsImageDataset as TestImageDataset # input without shrinking
from torchvision.utils import save_image

# ----- config
import model
import set_e
settings = set_e.Settings()
opt = set_e.Param()
torch.manual_seed(settings.seed)
torch.cuda.manual_seed_all(settings.seed)

# ----- main
# generator_weight_path = 'G:/IMG_Dataset/weight/generator_00014000.pth'
# generator_weight_path = 'G:/IMG_Dataset_/weight/generator_00008000.pth'
generator_weight_path = 'D:/IMG_Dataset_val/tmp2/weight/generator_00059000.pth'

def denormalize(t):
    for i in range(3):
        t[:, i].mul_(opt.std[i]).add_(opt.mean[i])
    return torch.clamp(t, 0, 255)

# ----- test

if __name__ == '__main__':
    demo_dataloader = DataLoader(
        TestImageDataset(settings.image_dir_demo), batch_size=1, shuffle=False, num_workers=opt.num_workers,)
    generator = model.GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(opt.device)
    load_model = torch.load(generator_weight_path)
    generator.load_state_dict(load_model)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # print(load_model.keys())
    # print(type(demo_dataloader))
    with torch.no_grad():
        for i, imgs in enumerate(demo_dataloader):
            print(i)
            imgs_lr = Variable(imgs['lr'].type(Tensor))

            gen_hr = generator(imgs_lr)
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

            imgs_lr = denormalize(imgs_lr)
            gen_hr = denormalize(gen_hr)
            os.makedirs(settings.image_dir_demo, exist_ok=True)

            save_image(imgs_lr, Path(settings.image_dir_demo).joinpath('low_{:01}.{}'.format(i, settings.demo_img_format)), normalize=False)
            save_image(gen_hr, Path(settings.image_dir_demo).joinpath('gen_hr_{:01}.{}'.format(i, settings.demo_img_format)), normalize=False)
