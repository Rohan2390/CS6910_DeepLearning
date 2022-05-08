# CS6910 DL Assignment 3

### Common Files for Both Part of assignments:
- dataPreparation.py => This file takes language as input and train,valid and test csv with preprocessing, and also outputs JSON of which character is mapped to which int for both languages.
- visualization.ipynb => Prediction analysis and plotting notebook
### For Data Preparation
Use following command for running dataPreparation.py

```commandline
python3 dataPreparation.py --lg lg
```

Arguments that can be passed to this file are as given below.
```commandline
--lg langauge to use other than english.
```