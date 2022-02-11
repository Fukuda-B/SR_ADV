'''
    SR_ADV | make_d.py

    make random crop data
'''
import os
import re
import glob
import cv2
import numpy as np
from pathlib import Path
from matplotlib import image
from tqdm import tqdm

# -----
settings = {
    'image_dir_load' : 'G:/IMG_Dataset/all',
    'image_dir_save' : 'G:/IMG_Dataset/crop',
    'seed' : 1011,
    'random_crop_cnt' : 8, # Number of times to crop from a single image
    'crop_size' : (128, 128),
    'load_img_format' : 'png',
    'save_img_format' : 'jpg',
}
np.random.seed(seed=settings['seed'])

# -----
def random_crop(image_p, crop_size):
    '''
        image           : np.array
        crop_size tuple : (h, w)
    '''
    # print(type(image_p))
    image = cv2.imread(image_p)
    h, w, c = image.shape
    # if (h < crop_size[0] or w < crop_size[1]):
    #     raise Exception('image is smaller then the crop size')
    pos_top = np.random.randint(0, h-crop_size[0])
    pos_left = np.random.randint(0, h-crop_size[1])
    res = image[pos_top:pos_top+crop_size[0], pos_left:pos_left+crop_size[1]]
    return res

def run_crop(settings):
    '''
        settings        : {settings}
    '''
    load_path = Path(settings['image_dir_load'])
    save_path = Path(settings['image_dir_save'])
    img_f = list(load_path.glob('*.'+settings['load_img_format']))
    cc = 0
    for i in tqdm(img_f):
        i = str(i)
        for j in range(settings['random_crop_cnt']):
            try:
                proc_img = random_crop(i, settings['crop_size'])
            except Exception as e:
                continue
            save_name = f"{cc}.{settings['save_img_format']}"
            s_path = str(save_path.joinpath(save_name))
            cv2.imwrite(s_path, proc_img)
            cc += 1

if __name__ == '__main__':
    run_crop(settings)
