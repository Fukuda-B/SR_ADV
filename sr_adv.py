'''
    SR_ADV | sr_adv.py
'''

# ---- module
import os
import torch
from torch.autograd import Variable
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from train_set import TestImageDataset

# ----- config
import model
import set_e
settings = set_e.Settings()
import train
opt = train.Param()

# ----- main
generator_weight_path = ''

def denormalize(t):
    for i in range(3):
        t[:, i].mul_(opt.std[i]).add_(opt.mean[i])
    return torch.clamp(t, 0, 255)

demo_dataloader = DataLoader(
    TestImageDataset(settings.image_dir_demo),
    batch_size=1,
    shuffle=False,
    num_workers=opt.num_workers,
)

generator = model.GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(opt.device)
generator.load_state_dict(torch.load(generator_weight_path))
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

with torch.no_grad():
    for i, imgs in enumerate(demo_dataloader):
        imgs_lr = Variable(imgs['lr'].type(Tensor))

        gen_hr = generator(imgs_lr)
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

        imgs_lr = denormalize(imgs_lr)
        gen_hr = denormalize(gen_hr)
        os.mkdirs(settings.image_dir_demo, exist_ok=True)

        model.save_tmp_image(imgs_lr, Path(settings.image_dir_demo).joinpath('low_{:01}.{}'.format(i, settings.demo_img_format), nrow=1, normalize=False))
        model.save_tmp_image(gen_hr, Path(settings.image_dir_demo).joinpath('gen_hr_{:01}.{}'.format(i, settings.demo_img_format), nrow=1, normalize=False))
