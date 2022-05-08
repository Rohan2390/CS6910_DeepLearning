from tensorflow.keras.preprocessing.sequence import pad_sequences
import main as convert
import pandas as pd
import numpy as np
import json
import os

#Decode output vectors to wors
def decodeWords(encoding, mapping, lg):
    l = [0, 1]

    output = []

    for word in encoding:
        string = ''
        for c in word:
            if c not in l:
                string += mapping[lg][int(c)]

            if c == 0:
                break
        output.append(string)

    return output

#Test on model
def test(model, testPath='test.csv'):
    test = pd.read_csv(testPath)

    lg = test.columns[0]
    en = 'en'

    test[lg] = test[lg].apply(convert.convertToArray).apply(lambda x: x[1:])
    test[en] = test[en].apply(convert.convertToArray)  # .apply(lambda x: x[1:] if config['padding'] == 'post' else x)

    testX = pad_sequences(test[en].values, padding='pre',
                          value=1.0, maxlen=model.maxLen)
    testY = pad_sequences(test[lg].values, padding='post',
                          value=0.0, maxlen=model.maxLen)

    print("Loading Best Model")
    model.loadTestModel()
    print("For Test")
    _, predictions, scores = model.evaluate(testX, testY)

    with open('dict.json', 'r') as f:
        outputDict = json.load(f)
        outputDict[en] = {outputDict[en][i]: i for i in outputDict[en]}
        outputDict[lg] = {outputDict[lg][i]: i for i in outputDict[lg]}

    preds = pd.DataFrame()
    preds[en] = decodeWords(test[en], outputDict, en)
    preds[lg] = decodeWords(test[lg], outputDict, lg)
    preds[lg + '_preds'] = decodeWords(predictions, outputDict, lg)

    shiftedPreds = np.concatenate([
        np.concatenate([np.ones(1, ), i[1:]]).reshape(1, -1) for i in predictions
    ], axis=0)
    connectivity = model.getConnectivity(testX, shiftedPreds)

    #Save output
    if not os.path.exists('predictions_attention'):
        os.mkdir('predictions_attention')

    with open('predictions_attention/attention_score.npy', 'wb') as f:
        np.save(f, scores)
    with open('predictions_attention/connectivity.npy', 'wb') as f:
        np.save(f, connectivity)
    preds.to_csv('predictions_attention/predictions.csv', index=False)

