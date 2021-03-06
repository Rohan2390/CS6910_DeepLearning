from tensorflow.keras.layers import Input, Embedding, LSTM, SimpleRNN, GRU, Dense, Concatenate, Activation
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import Attention
import numpy as np
from tqdm import tqdm
import os

RNNLayer = {'RNN': SimpleRNN, 'LSTM': LSTM, 'GRU': GRU}


class RNNModel:

    def __init__(self, config, maxLen, inputVocabSize, outputVocabSize):

        self.config = config
        self.maxLen = maxLen
        self.inputVocabSize = inputVocabSize
        self.outputVocabSize = outputVocabSize

        #Creating layers
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
                recurrent_dropout=config['dropout'],
                return_state=True,
                return_sequences=True
            ))

        self.attentionLayer = Attention.BahdanauAttentionLayer(config['RNNLayerDims'])

        self.decoderLayers = []

        for layer in range(config['numDecoderLayers']):
            self.decoderLayers.append(RNNLayer[config['RNNLayer']](
                config['RNNLayerDims'],
                dropout=config['dropout'],
                recurrent_dropout=config['dropout'],
                return_state=True,
                return_sequences=True
            ))

        self.finalDense = Dense(outputVocabSize)
        self.finalSoftmax = Activation("softmax")

        self.createTrainModel()
        self.createInferenceModels()

    #Function to create Training model
    def createTrainModel(self):

        encoderEmbeds = self.encoderEmbedding(self.encoderInput)
        encoderOutput = self.encoderLayers[0](encoderEmbeds)

        for layer in self.encoderLayers[1:]:
            encoderOutput = layer(encoderOutput[0])

        decoderEmbeds = self.decoderEmbedding(self.decoderInput)
        decoderOutput = self.decoderLayers[0](decoderEmbeds, initial_state=encoderOutput[1:])

        for layer in self.decoderLayers[1:]:
            decoderOutput = layer(decoderOutput[0], initial_state=encoderOutput[1:])

        attentionOutput, _ = self.attentionLayer(decoderOutput[0], encoderOutput[0])
        concatOutput = Concatenate(axis=-1)([decoderOutput[0], attentionOutput])

        finalOutput = self.finalDense(concatOutput)
        finalOutput = self.finalSoftmax(finalOutput)

        self.trainModel = Model([self.encoderInput, self.decoderInput], finalOutput)
        self.trainModel.summary()

    #Function to create Inference Model
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
        encoderOutputDecoderInput = Input(shape=(None, self.config['RNNLayerDims']))
        decoderCurrentOutput = [self.decoderEmbedding(decoderInput)]
        decoderInputs = [decoderInput, encoderOutputDecoderInput]
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

        attentionOutput, attentionScores = self.attentionLayer(decoderCurrentOutput[0], encoderOutputDecoderInput)
        concatOutput = Concatenate(axis=-1)([decoderCurrentOutput[0], attentionOutput])

        finalOutput = self.finalDense(concatOutput)
        finalOutput = self.finalSoftmax(finalOutput)

        self.decoder = Model(decoderInputs, [finalOutput, attentionScores] + decoderOutputs)

        self.decoder.summary()

    #Comiple train Model
    def compile(self, **kwargs):
        self.trainModel.compile(**kwargs)

    #Fit on train Moadel
    def fit(self, **kwargs):
        self.trainModel.fit(**kwargs)

    #Predict using input at word level
    def predict(self, x):

        predictions = []
        scores = []
        print("Calculating Encoder Output")
        encoderOutput = self.encoder.predict(x, batch_size=self.config['bs'])
        state = encoderOutput[1:] * self.config['numDecoderLayers']
        encoderOutput = [encoderOutput[0]]
        output = [np.ones((len(x), 1))]

        print("Calculating Decoder Output")
        for t in tqdm(range(self.maxLen)):
            output = self.decoder.predict(output + encoderOutput + state, batch_size=self.config['bs'])
            state = output[2:]
            scores.append(output[1])
            output = [np.argmax(output[0], axis=2)]
            predictions += output

        return np.concatenate(predictions, axis=1), np.concatenate(scores, axis=1)
    #Evaluate Word level accuracy
    def evaluate(self, x, y):

        preds, scores = self.predict(x)
        correct = 0

        for iY, iPred in zip(y, preds):

            if np.all(iY == iPred):
                correct += 1

        print(f'Word level Accuracy is {correct / len(y):0.4f}')
        return correct / len(y), preds, scores

    #Save model
    def saveTestModel(self):

        if not os.path.exists("model"):
            os.mkdir("model")

        self.encoder.save("model/encoder")
        self.decoder.save("model/decoder")
    #Load model
    def loadTestModel(self):
        self.encoder = load_model("model/encoder")
        self.decoder = load_model("model/decoder")

    #Create model for Connectivity calculation
    def createGradientModel(self):

        encoderEmbeds = Input(shape=(self.maxLen, self.config['embeddingDims']))
        encoderOutput = self.encoderLayers[0](encoderEmbeds)

        for layer in self.encoderLayers[1:]:
            encoderOutput = layer(encoderOutput[0])

        decoderEmbeds = self.decoderEmbedding(self.decoderInput)
        decoderOutput = self.decoderLayers[0](decoderEmbeds, initial_state=encoderOutput[1:])

        for layer in self.decoderLayers[1:]:
            decoderOutput = layer(decoderOutput[0], initial_state=encoderOutput[1:])

        attentionOutput, _ = self.attentionLayer(decoderOutput[0], encoderOutput[0])
        concatOutput = Concatenate(axis=-1)([decoderOutput[0], attentionOutput])

        finalOutput = self.finalDense(concatOutput)

        self.gradientModel = Model([encoderEmbeds, self.decoderInput], finalOutput)

    #Perform back propogations to get connectivity
    def getConnectivity(self, x, shiftedPreds):

        gradientBatch = 1024

        self.createGradientModel()

        connectivity = []

        for t in tqdm(range(self.maxLen)):

            batchConnectivity = []

            for i in range(0,len(x),gradientBatch):
                batchX = tf.convert_to_tensor(x[i:i+gradientBatch], dtype=tf.float32)
                batchShiftedPreds = tf.convert_to_tensor(shiftedPreds[i:i+gradientBatch])

                with tf.GradientTape() as tape:
                    tape.watch(batchX)
                    embedOutput = self.encoderEmbedding(batchX)
                    tape.watch(embedOutput)
                    output = self.gradientModel([embedOutput, batchShiftedPreds])
                    batchConnectivity.append(np.linalg.norm(tape.gradient(output[:, t, :], embedOutput).numpy(),axis=2))

            connectivity.append(np.concatenate(batchConnectivity,axis=0))

        connectivity = np.stack(connectivity)

        return connectivity