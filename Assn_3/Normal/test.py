from keras.preprocessing.sequence import pad_sequences
from main import convertToArray
import pandas as pd


def test(model,testPath='test.csv'):

    test = pd.read(testPath)

    lg = test.columns[0]
    en = 'en'

    test[lg] = test[lg].apply(convertToArray).apply(lambda x: x[1:])
    test[en] = test[en].apply(convertToArray)  # .apply(lambda x: x[1:] if config['padding'] == 'post' else x)


    testX = pad_sequences(test[en].values, padding='pre',
                           value=1.0, maxlen=model.maxLen)
    testY = pad_sequences(test[lg].values, padding='post',
                           value=0.0, maxlen=model.maxLen)

    print("Loading Best Model")
    model.loadTestModel()
    print("For Test")
    model.evaluate(testX, testY)

