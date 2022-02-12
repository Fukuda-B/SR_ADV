'''
    SR_ADV | sr_adv.py
'''

# ---- module
import torch

# ----- config
import set_e
settings = set_e.Settings()

# ----- main
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
with torch.no_grad():
    for i, imgs in enumerate(demo_dataloader):
        imgs_lr = Variable(imgs['lr'].type(Tensor))

        gen_hr = generator(imgs_lr)
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

        imgs_lr = denormalize(imgs_lr)
        gen_hr = denormalize(gen_hr)
        os.mkdirs(settings.image_dir_demo, exist_ok=True)

        save_image(imgs_lr, Path(settings.image_dir_demo).joinpath('low_{:01}.{}'.format(i, settings.demo_img_format), nrow=1, normalize=False))
        save_image(imgs_hr, Path(settings.image_dir_demo).joinpath('gen_hr_{:01}.{}'.format(i, settings.demo_img_format), nrow=1, normalize=False))
