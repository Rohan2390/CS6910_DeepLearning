import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,AdditiveAttention

@tf.keras.utils.register_keras_serializable()
class BahdanauAttentionLayer(Layer):

    def __init__(self, dims):
        super().__init__()
        #Dense before attentions
        self.wQuery = Dense(dims, use_bias=False)
        self.wValue = Dense(dims, use_bias=False)
        #Actual Attention
        self.attentionLayer = AdditiveAttention()

    def call(self, query,value):
        #Perform layer on input
        queryOutput = self.wQuery(query)
        valueOutput = self.wValue(value)

        queryMask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        valueMask = tf.ones(tf.shape(value)[:-1], dtype=bool)

        attentionOutput,attentionScore = self.attentionLayer(
          [queryOutput,value,valueOutput],
          mask=[queryMask, valueMask],
          return_attention_scores=True
        )

        return attentionOutput,attentionScore
