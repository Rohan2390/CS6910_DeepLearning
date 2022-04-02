import tensorflow as tf
from keras.models import Model
from matplotlib import pyplot as plt
import numpy as np

@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  return tf.nn.relu(x), grad


def main(model,xBatch):

    gbpen_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(index=20).output]
    )

    layer_dict = [layer for layer in gbpen_model.layers if hasattr(layer, 'activation')]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guidedRelu

    fig = plt.figure(figsize=(20, 4))
    fig.subplots_adjust(bottom=0.025, left=0.025, top=0.975, right=0.975)

    axs = fig.add_subplot(2,10,(1,10))
    axs.imshow(xBatch[0]/255,aspect=1.0)
    axs.set_title('Original Image')
    axs.set_xticks([])
    axs.set_yticks([])

    op_shape = model.layers[20].output.shape[1:]

    for i in range(10):

        neuron = (np.random.randint(0, op_shape[0]),
                  np.random.randint(0, op_shape[1]),
                  np.random.randint(0, op_shape[2]))

        mask = np.zeros((1,*op_shape),dtype=np.float)
        mask[0,neuron[0],neuron[1],neuron[2]] = 1

        with tf.GradientTape() as tape:
            inputs = tf.cast(xBatch[0:1],tf.float32)
            tape.watch(inputs)
            outputs = gbpen_model(inputs) * mask

        grads = tape.gradient(outputs, inputs)[0]

        img_gb = np.dstack((grads[:, :, 0], grads[:, :, 1], grads[:, :, 2],))
        m,s = img_gb.mean(),img_gb.std()
        img_gb -= m
        img_gb /= s
        img_gb *= 0.25
        img_gb += 0.5
        img_gb = np.clip(img_gb,0,1)

        axs = fig.add_subplot(2,10,11+i)
        axs.imshow(img_gb,aspect='auto')
        axs.set_xticklabels([])
        axs.set_yticklabels([])
        axs.set_title("Excited Neurons")

    plt.savefig('GBPEN.png')


