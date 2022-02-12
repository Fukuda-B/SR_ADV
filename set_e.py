'''
    SR_ADV | set_e.py

    environmental settings
'''

settings = {
    'seed' : 1011,
    'image_dir_load' : 'G:/IMG_Dataset/all', # not cropped image
    'image_dir_save' : 'G:/IMG_Dataset/crop', # cropped image
    'load_img_format' : 'png',
    'save_img_format' : 'jpg',
    'random_crop_cnt' : 4, # Number of times to crop from a single image
    'crop_size' : (128, 128),
}
