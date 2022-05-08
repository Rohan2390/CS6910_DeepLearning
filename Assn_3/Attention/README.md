# CS6910 DL Assignment 3

### Attention RNN
- model.py => This file is used to create RNN Model
- Attention.py => This file has Bahadnau attention layer implemented
- main.py => This file performs single run for given config
- wandbmain.py => This file is used to perform runs that are logged in wandb, both for new sweeps as well as running sweeps
- test.py => Perform test on test data and creates prediction files and connectivity and attention map
### For running normal training on a config using cmd
Use following command for running main.py

```commandline
python3 main.py
```

Arguments that can be passed to this file are as given below.
```commandline
--lr Learning rate
--bs Batch Size
--epochs Epochs
--embeddingDims Output Dimension of Embedding
--RNNLayer LSTM,GRU,RNN
--RNNLayerDims Number of neurons in RNN Layers
--numEncoderLayers Number of Encoder Layers
--numDEcoderLayers Number of Decoder Layers
--dropout Dropout layer probability
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