from keras.layers import Input, Embedding, LSTM, SimpleRNN, GRU, Dense
from keras.models import Model
import numpy as np
from tqdm import tqdm

RNNLayer = {'rnn': SimpleRNN, 'LSTM': LSTM, 'GRU': GRU}


class RNNModel:

    def __init__(self, config, maxLen, inputVocabSize, outputVocabSize):

        self.config = config
        self.maxLen = maxLen
        self.inputVocabSize = inputVocabSize
        self.outputVocabSize = outputVocabSize

        self.encoderInput = Input(shape=(self.maxLen,))
        self.decoderInput = Input(shape=(self.maxLen,))

        self.encoderEmbedding = Embedding(input_dim=inputVocabSize,
                                          output_dim=config['embeddingDims'],
                                          input_length=maxLen
                                          )
        self.decoderEmbedding = Embedding(input_dim=outputVocabSize,
                                          output_dim=config['embeddingDims'],
                                          input_length=maxLen
                                          )

        self.encoderLayers = []

        for layer in range(config['numEncoderLayers']):
            self.encoderLayers.append(RNNLayer[config['RNNLayer']](
                config['RNNLayerDims'],
                dropout=config['dropout'],
                return_state=True,
                return_sequences=layer != config['numEncoderLayers'] - 1
            ))

        self.decoderLayers = []

        for layer in range(config['numDecoderLayers']):
            self.decoderLayers.append(RNNLayer[config['RNNLayer']](
                config['RNNLayerDims'],
                dropout=config['dropout'],
                return_state=True,
                return_sequences=True
            ))

        self.finalDense = Dense(outputVocabSize, activation='softmax')

        self.createTrainModel()
        self.createInferenceModels()

    def createTrainModel(self):

        encoderEmbeds = self.encoderEmbedding(self.encoderInput)
        encoderOutput = self.encoderLayers[0](encoderEmbeds)

        for layer in self.encoderLayers[1:]:
            encoderOutput = layer(encoderOutput[0])

        decoderEmbeds = self.decoderEmbedding(self.decoderInput)
        decoderOutput = self.decoderLayers[0](decoderEmbeds, initial_state=encoderOutput[1:])

        for layer in self.decoderLayers[1:]:
            decoderOutput = layer(decoderOutput[0], initial_state=encoderOutput[1:])

        finalOutput = self.finalDense(decoderOutput[0])

        self.trainModel = Model([self.encoderInput, self.decoderInput], finalOutput)
        self.trainModel.summary()

    def createInferenceModels(self):

        encoderInput = Input(shape=(self.maxLen,))
        encoderEmbeds = self.encoderEmbedding(encoderInput)
        encoderOutput = self.encoderLayers[0](encoderEmbeds)

        for layer in self.encoderLayers[1:]:
            encoderOutput = layer(encoderOutput[0])

        self.encoder = Model(
            encoderInput, encoderOutput
        )

        self.encoder.summary()

        decoderInput = Input(shape=(None,))
        decoderCurrentOutput = [self.decoderEmbedding(decoderInput)]
        decoderInputs = [decoderInput]
        decoderOutputs = []

        for layer in self.decoderLayers:

            if self.config['RNNLayer'] == 'LSTM':
                decoderHStateInput = Input(shape=(self.config['RNNLayerDims'],))
                decoderCStateInput = Input(shape=(self.config['RNNLayerDims'],))
                layerStateInput = [decoderHStateInput, decoderCStateInput]
                decoderInputs += layerStateInput

            else:
                decoderStateInput = Input(shape=(self.config['RNNLayerDims'],))
                layerStateInput = [decoderStateInput]
                decoderInputs += layerStateInput

            decoderCurrentOutput = layer(decoderCurrentOutput[0], initial_state=layerStateInput)
            decoderOutputs += list(decoderCurrentOutput[1:])

        finalOutput = self.finalDense(decoderCurrentOutput[0])

        self.decoder = Model(decoderInputs, [finalOutput] + decoderOutputs)

        self.decoder.summary()

    def compile(self, **kwargs):
        self.trainModel.compile(**kwargs)

    def fit(self, **kwargs):
        self.trainModel.fit(**kwargs)

    def predict(self, x):

        predictions = []
        encoderOutput = self.encoder.predict(x,batch_size=self.config['bs'])

        for j,example in enumerate(x):

            state = [s[j:i+1] for s in encoderOutput[1:]] * self.config['numDecoderLayers']

            output = [np.ones((1, 1))]
            prediction = []
            outputC = -1
            i = 0

            while(outputC!=0 and i<self.maxLen):

                output = self.decoder.predict(output + state)
                state = output[1:]
                output = [np.argmax(output[0], axis=2)]
                outputC = output[0][0][0]
                prediction.append(outputC)

            predictions.append(prediction)

        return np.concatenate(predictions, axis=1)

    def evaluate(self, x, y):

        preds = self.predict(x)
        correct = 0
        print(y[69], preds[69])
        for iY, iPred in zip(y, preds):

            if np.all(iY == iPred):
                correct += 1

        print(f'Word level Accuracy is {correct / len(y):0.4f}')
        return correct / len(y), preds


if __name__ == '__main__':
    m = RNNModel({
        'embeddingDims': 256,
        'RNNLayer': 'LSTM',
        'RNNLayerDims': 256,
        'dropout': 0.3,
        'numDecoderLayers': 3,
        'numEncoderLayers': 3
    }, 20, 35, 26)

    m.predict(np.zeros((100, 20)))




