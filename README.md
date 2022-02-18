# SR_ADV
Super-resolution.

### Usage
Put images to be used for training in the directory [set_e.Settings.image_dir_load](https://github.com/Fukuda-B/SR_ADV/blob/0ce3b01708db0eaf6a06076790b8adf99ee22dd9/set_e.py#L12).  
Execute `make_d.py` to cut out images randomly.  
(If you want to check the training status, put the images in the directory set_e.Settings.image_dir_test, and the output for each set sample interval will be saved in the directory set_e.Settings.image_dir_proc.)  
Run `train.py` to train.  
The weights are saved in the directory [set_e.Settings.weight_dir_save](https://github.com/Fukuda-B/SR_ADV/blob/0ce3b01708db0eaf6a06076790b8adf99ee22dd9/set_e.py#L25).  
(Settings.image_dir_demo directory if you want to check the learned weights.)  
When the training is complete, set the variable ([generator_weight_path](https://github.com/Fukuda-B/SR_ADV/blob/0ce3b01708db0eaf6a06076790b8adf99ee22dd9/sr_adv.py#L25)) to the name of the file where the weights are saved in and run `sr_adv.py`.  



