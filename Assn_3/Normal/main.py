from model import RNNModel
import test
from keras.optimizer_v2.adam import Adam
import wandb
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import History
import pandas as pd
import numpy as np
import argparse


def convertToArray(string):
    string = string.strip('[]')
    output = [int(i) for i in string.split(', ')]
    return output


def train(config, trainPath='train.csv', validPath='valid.csv', wandbLog=False):
    train = pd.read_csv(trainPath)
    valid = pd.read_csv(validPath)

    lg = train.columns[0]
    en = 'en'

    train[lg] = train[lg].apply(convertToArray).apply(lambda x: x[1:])
    train[en] = train[en].apply(convertToArray)#.apply(lambda x: x[1:] if config['padding'] == 'post' else x)

    valid[lg] = valid[lg].apply(convertToArray).apply(lambda x: x[1:])
    valid[en] = valid[en].apply(convertToArray)#.apply(lambda x: x[1:] if config['padding'] == 'post' else x)

    maxLen = max(
        train[lg].apply(len).max(),
        train[en].apply(len).max(),
        valid[lg].apply(len).max(),
        valid[en].apply(len).max()
    )

    trainX = pad_sequences(train[en].values, padding='pre',
                           value=1.0, maxlen=maxLen)
    trainY = pad_sequences(train[lg].values, padding='post',
                           value=0.0, maxlen=maxLen)

    trainYInput = pad_sequences(train[lg].apply(lambda x: [1] + x[:-1]).values,
                                padding='post',
                                value=0.0, maxlen=maxLen)

    validX = pad_sequences(valid[en].values, padding='pre',
                           value=1.0, maxlen=maxLen)
    validY = pad_sequences(valid[lg].values, padding='post',
                           value=0.0, maxlen=maxLen)

    inputVocabSize = np.max(trainX) + 1
    outputVocabSize = np.max(trainY) + 1

    model = RNNModel(config, maxLen, inputVocabSize, outputVocabSize)
    optimizer = Adam(lr=config['lr'])

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = History()

    epochs = config['epochs']
    oldAcc = float('-inf')

    for epoch in range(epochs):

        model.fit(
            x=[trainX, trainYInput],
            y=trainY,
            batch_size=config['bs'],
            epochs=1,
            shuffle=True,
            callbacks=[history]
        )

        print("#########################")
        print("For Valid")
        valAcc, _ = model.evaluate(validX, validY)

        if valAcc > oldAcc:
            print("Saving Model")
            oldAcc = valAcc
            model.saveTestModel()

        if wandbLog:
            wandb.log({
                "Train Loss": history.history['loss'][-1],
                "Train Acc": history.history['accuracy'][-1],
                "Val Acc": valAcc,
                "epoch": epoch
            })

    if not wandbLog:
        test.test(model)

#Updating Config to new args from command line
def updateConfig(args, config):

    if args.lr:
        config['lr']=args.lr

    if args.epochs:
        config['epochs']=args.epochs

    if args.bs:
        config['bs']=args.bs

    if args.embeddingDims:
        config['embeddingDims']=args.embeddingDims

    if args.RNNLayer:
        config['RNNLayer']=args.RNNLayer

    if args.RNNLayerDims:
        config['RNNLayerDims']=args.RNNLayerDims

    if args.numEncoderLayers:
        config['numEncoderLayers']=args.numEncoderLayers

    if args.numDecoderLayers:
        config['numDecoderLayers']=args.numDecoderLayers

    if args.dropout:
        config['dropout']=args.dropout

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Part A training')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', dest='epochs', type=int, help='Number of epochs')
    parser.add_argument('--bs', dest='bs', type=int, help='Batch Size')
    parser.add_argument('--embeddingDims', dest='embeddingDims', type=int, help='Output Dimension of Embedding')
    parser.add_argument('--RNNLayer', dest='RNNLayer', type=str, help='LSTM,GRU,RNN')
    parser.add_argument('--RNNLayerDims', dest='RNNLayerDims', type=int, help='Number of neurons in RNN Layers')
    parser.add_argument('--numEncoderLayers', dest='numEncoderLayers', type=int, help='Number of Encoder Layers')
    parser.add_argument('--numDEcoderLayers', dest='numDecoderLayers', type=int, help='Number of Decoder Layers')
    parser.add_argument('--dropout', dest='dropout', type=float, help='Dropout layer probability')

    config = {
        'lr': 0.001,
        'epochs': 10,
        'bs': 64,
        'embeddingDims': 256,
        'RNNLayer': 'LSTM',
        'RNNLayerDims': 256,
        'numEncoderLayers': 3,
        'numDecoderLayers': 3,
        'dropout': 0.3,
    }

    config = updateConfig(parser.parse_args(), config)
    train(config)