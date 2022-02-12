'''
    SR_ADV | set_e.py

    environmental settings
'''

# ----- config
class Settings:
    def __init__(self):
        # crop
        self.seed = 1011
        self.image_dir_load = 'G:/IMG_Dataset/all' # not cropped image
        self.image_dir_save = 'G:/IMG_Dataset/crop' # cropped image
        self.load_img_format = 'png'
        self.save_img_format = 'jpg'
        self.random_crop_cnt = 4  # Number of times to crop from a single image
        self.crop_size = (128, 128)

        # train
        self.image_dir_test =  'G:/IMG_Dataset/test'
        self.image_dir_demo = 'G:/IMG_Dataset/demo'
        self.test_img_format = 'jpg'
        self.demo_img_format = 'png'
        self.weight_dir_save =  'G:/IMG_Dataset/weight'