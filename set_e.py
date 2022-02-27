'''
    SR_ADV | set_e.py

    environmental settings
'''

# ----- module
import os
import torch
import numpy as np

# ----- settings
class Settings:
    def __init__(self):
        host_dir = 'G:/IMG_Dataset'
        # host_dir = 'G:/IMG_Dataset_gen'
        # host_dir = '/content/drive/My Drive/SR_ADV/2022_2_24_4'
        # host_dir = '../img-dataset'
        # host_dir = '../img-dataset-gen'

        save_dir = host_dir
        # save_dir = '../../working'

        # crop
        self.seed = 1011
        self.image_dir_load = f'{host_dir}/all' # not cropped image
        self.image_dir_save = f'{host_dir}/crop' # cropped image & train image
        self.load_img_format = 'png'
        self.save_img_format = 'jpg'
        # self.random_crop_cnt = 3  # Number of times to crop from a single image
        # self.random_crop_cnt = 6  # Number of times to crop from a single image
        self.random_crop_cnt = 7  # Number of times to crop from a single image
        self.crop_size = (128, 128)

        # train
        self.image_dir_test =  f'{host_dir}/test' # for check learning
        self.image_dir_demo = f'{host_dir}/demo' # SR_ADV input
        self.test_img_format = 'jpg'
        self.demo_img_format = 'png'
        self.image_dir_proc = f'{save_dir}/proc' # save temp image dir
        self.weight_dir_save =  f'{save_dir}/weight' # save weight dir
        self.log_dir = f'{save_dir}/logs' # save logs dir

# ----- opt
class Param:
    def __init__(self):
        self.n_epoch = 50
        # self.n_epoch = 100
        self.batch_size = 16
        # self.batch_size = 8
        self.warmup_batches = 500
        # self.warmup_batches = 5
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
        self.random_flip = 0.3
        self.random_blur_p = 0.3
        self.random_jpg_noise_p = 0.5
        self.random_jpg_noise_quality = (0, 30)

