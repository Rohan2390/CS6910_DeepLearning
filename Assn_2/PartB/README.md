# CS6910 DL Assignment 2

### Part B Files
- model.py => This file is used to create Transfer Learning Model for Part B
- main.py => This file performs single run for given config
- wandbmain.py => This file is used to perform runs that are logged in wandb, both for new sweeps as well as running sweeps

### For running normal training on a config using cmd
Use following command for running main.py

```commandline
python3 main.py
```

Arguments that can be passed to this file are as given below.
```commandline
--baseModel Base Model for Transfer Learning
--lr Learning rate
--rotation_range Rotation Augmentation
--shifting_range Hieght,Width and Zoom Shift Augmentations
--flip Horizontal and Vetical Flip
--imageSize Image Size
--bs Batch Size
--epochs Epochs
--epochUpdate Epochs to update training layers for updating
--pTrainLayers Percentage of Train layers to train
--denseNeurons Neurons in Dense Layer
```

### For running wandb sweeps
Use following command for running wandbmain.py

```commandline
python3 wandbmain.py
```
Arguments that can be passed to this file are as given below.
```commandline
--path  Path to the config.json used for starting new sweep using that config.
--sweepId Sweep Id of a running sweep to continue new runs in that sweep. 
```
Only 1 argument is needed for running, if both given new sweep using config.json is created.

Config.json should have same keys as arguments given to the main.py mentioned above.