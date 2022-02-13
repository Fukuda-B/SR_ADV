'''
    SR_ADV | train.py

    train
'''

# ----- module
import torch
import train_set
import model

# ---- config
class Param:
    def __init__(self):
        self.n_epoch = 50
        self.batch_size = 16
        self.warmup_batches = 500
        self.sample_interval = 100
        self.checkpoint_interval = 1000
        self.n_cpu = 8
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
        self.log_dir = 'G:/IMG_Dataset/logs'

# ----- main
if __name__ == '__main__':
    opt = Param()
    gan = model.ESRGAN(opt)
    for epoch in range(1, opt.n_epoch+1):
        for batch_num, imgs in enumerate(train_dataloader):
            batches_done=(epoch-1)*len(train_dataloader)+batch_num

            if batches_done <= opt.warmup_batches:
                gan.pre_train(imgs, batches_done) # pre train
            else:
                gan.train(imgs, batches_done) # train

            # save sample
            if batches_done%opt.sample_interval==0:
                for i, imgs in enumerate(test_dataloader):
                    gan.save_image(imgs, batches_done)

            # save weight
            if batches_done%opt.checkpoint_interval==0:
                gan.save_weight(batches_done)
