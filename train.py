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
import datetime
import torch
import numpy as np
import train_set
import model
# import model_deepspeed as model # if use deepspeed
from torch.utils.data import DataLoader
import gc

# ----- config
import set_e
settings = set_e.Settings()
opt = set_e.Param()

torch.manual_seed(settings.seed)
torch.cuda.manual_seed_all(settings.seed)

load_gen_model_name = False
load_dis_model_name = False
# load_gen_model_name = 'G:/IMG_Dataset_gen/generator_00014000.pth' # 読み込む重みが保存されたファイルの名前 (generator)
# load_dis_model_name = 'G:/IMG_Dataset_gen/discriminator_00014000.pth' # (discriminator)

# ----- main
if __name__ == '__main__':
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

    if load_gen_model_name and load_dis_model_name: # 学習済み重みを読み込む場合
        # generator = model.GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(opt.device)
        # discriminator = model.Discriminator((opt.channels, *opt.hr_shape))
        load_gen_model = torch.load(load_gen_model_name)
        load_dis_model = torch.load(load_dis_model_name)
        gan.generator.load_state_dict(load_gen_model)
        gan.discriminator.load_state_dict(load_dis_model)
        # gan.generator.eval()
        # gan.discriminator.eval()
        # del load_gen_model
        # del load_dis_model

        load_batch_num = int(load_gen_model_name[len(load_gen_model_name)-12:][:8])

        start_epoch = load_batch_num//len(train_dataloader)
        start_batch = load_batch_num%len(train_dataloader)
        print(f'load batches_num : {load_batch_num}, start_epoch : {start_epoch}, start_batch : {start_batch}')

    else: start_epoch = 1

    gc.collect()
    torch.cuda.empty_cache()

    print(f'start : {datetime.datetime.now()}')
    for epoch in range(start_epoch, opt.n_epoch+1):
        for batch_num, imgs in enumerate(train_dataloader):
            if load_gen_model_name and epoch==start_epoch and batch_num<=start_batch:
                continue

            batches_done=(epoch-1)*len(train_dataloader)+batch_num

            if batches_done <= opt.warmup_batches:
                gan.pre_train(imgs, batches_done, epoch, batch_num) # pre train
            else:
                gan.train(imgs, batches_done, epoch, batch_num, opt) # train

            # save sample
            if batches_done%opt.sample_interval==0:
                for i, imgs in enumerate(test_dataloader):
                    gan.save_tmp_image(imgs, batches_done, i, opt)

            # save weight
            if batches_done%opt.checkpoint_interval==0:
                gan.save_weight(batches_done)
    print(f'\nend : {datetime.datetime.now()}')
