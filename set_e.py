'''
    SR_ADV | set_e.py

    environmental settings
'''

# ----- config
class Settings:
    def __init__(self):
        host_dir = 'G:/IMG_Dataset'
        # host_dir = '/content/drive/My Drive/SR_ADV/2022_2_24_4'
        # host_dir = '../img-dataset'

        save_dir = host_dir
        # save_dir = '../../working'

        # crop
        self.seed = 1011
        self.image_dir_load = f'{host_dir}/all' # not cropped image
        self.image_dir_save = f'{host_dir}/crop' # cropped image & train image
        self.load_img_format = 'png'
        self.save_img_format = 'jpg'
        # self.random_crop_cnt = 3  # Number of times to crop from a single image
        self.random_crop_cnt = 6  # Number of times to crop from a single image
        self.crop_size = (128, 128)

        # train
        self.image_dir_test =  f'{host_dir}/test' # for check learning
        self.image_dir_demo = f'{host_dir}/demo' # SR_ADV input
        self.test_img_format = 'jpg'
        self.demo_img_format = 'png'
        self.image_dir_proc = f'{save_dir}/proc' # save temp image dir
        self.weight_dir_save =  f'{save_dir}/weight' # save weight dir
        self.log_dir = f'{save_dir}/logs' # save logs dir
