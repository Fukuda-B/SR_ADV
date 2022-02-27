# SR_ADV
Super-resolution.

### Usage
Put images to be used for training in the directory [set_e.Settings.image_dir_load](https://github.com/Fukuda-B/SR_ADV/blob/226ce008ef2f0d884c5c55f7acf2726af8dfa68a/set_e.py#L12).  
Execute `make_d.py` to cut out images randomly.  
(If you want to check the training status, put the images in the directory set_e.Settings.image_dir_test, and the output for each set sample interval will be saved in the directory set_e.Settings.image_dir_proc.)  
Run `train.py` to train.  
The weights are saved in the directory [set_e.Settings.weight_dir_save](https://github.com/Fukuda-B/SR_ADV/blob/226ce008ef2f0d884c5c55f7acf2726af8dfa68a/set_e.py#L25).  
(Settings.image_dir_demo directory if you want to check the learned weights.)  
When the training is complete, set the variable ([generator_weight_path](https://github.com/Fukuda-B/SR_ADV/blob/226ce008ef2f0d884c5c55f7acf2726af8dfa68a/sr_adv.py#L27)) to the name of the file where the weights are saved in and run `sr_adv.py`.  

---

`make_d.py` : Make croped images  
`train_set.py` : Dataset  
`set_e.py` : Settings (directory)  
`train.py` : Option (training) / Training  
`model.py` : GAN model  
`sr_adv.py` : Generate images using trained weight  
  
`model_deepspeed.py` : GAN model (use deepspeed)  
