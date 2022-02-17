'''
    SR_ADV | train.py

    train

-----
    memo

    https://discuss.pytorch.org/t/mean-and-std-values-for-transforms-normalize/70458
    https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
'''

# ----- module
import os
import torch
import numpy as np
import train_set
import model
from torch.utils.data import DataLoader

# ----- config
import set_e
settings = set_e.Settings()

class Param:
    def __init__(self):
        self.n_epoch = 50
        # self.batch_size = 16
        self.batch_size = 8
        # self.warmup_batches = 500
        self.warmup_batches = 5
        self.sample_interval = 100
        self.checkpoint_interval = 1000
        self.num_workers = os.cpu_count()
        self.hr_height = 128
        self.hr_width = 128
        self.hr_shape = (self.hr_height, self.hr_width)
        self.channels = 3
        self.residual_blocks = 23
        self.lr = 0.0002
        self.b1 = 0.9
        self.b2 = 0.999
        self.lambda_adv = 5.00E-03
        self.lambda_pixel = 1.00E-02
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

# ----- main
if __name__ == '__main__':
    opt = Param()
    gan = model.MODEL(opt)
    train_dataloader = DataLoader(
        train_set.ImageDataset(
            settings.image_dir_save,
            opt.hr_shape,
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        train_set.TestImageDataset(
            settings.image_dir_test,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    for epoch in range(1, opt.n_epoch+1):
        for batch_num, imgs in enumerate(train_dataloader):
            batches_done=(epoch-1)*len(train_dataloader)+batch_num

            if batches_done <= opt.warmup_batches:
                gan.pre_train(imgs, batches_done, epoch, batch_num) # pre train
            else:
                gan.train(imgs, batches_done, epoch, batch_num, opt) # train

            # save sample
            if batches_done%opt.sample_interval==0:
                for i, imgs in enumerate(test_dataloader):
                    gan.save_image(imgs, batches_done, i, opt)

            # save weight
            if batches_done%opt.checkpoint_interval==0:
                gan.save_weight(batches_done)
