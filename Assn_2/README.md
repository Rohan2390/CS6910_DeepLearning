# CS6910 DL Assignment 2

### Common Files for Both Part of assignments:
- dataPreparation.py => This file takes zip as input and splits train data into train and valid, also prepares data in file structure used for further training.
- test.py => This file performs evaluation of the given model on the test data and perform visualization of predictions, feature maps and images that excite neurons using Guided Back Prop.

### For Data Preparation
Use following command for running dataPreparation.py

```commandline
python3 dataPreparation.py
```

Arguments that can be passed to this file are as given below.
```commandline
--path Path to the .zip file containing data
```

### For testing
Use following command for running test.py
```commandline
python3 test.py
```
Arguments that can be passed to this file are as given below.
```commandline
--path  Path to the model used for testing
--imageSize Size of image
--bs Batch Size
--visualizeFilters Boolean to control visualizing of the filter only used if model trained is of PartA
--baseModel Base Model if preprocessing is used for Part B
```
