from model import TLModel
from keras.optimizers.adam_v2 import Adam
from keras.preprocessing.image_dataset import image_dataset_from_directory

def train(config):

    model = TLModel()
    optimizer = Adam()

    train_ds = image_dataset_from_directory(
    directory='inaturalist_12K/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256)
    )

    validation_ds = image_dataset_from_directory(
        directory='inaturlist_12K/test',
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(256, 256)
    )

    model.compile(optimizer=optimizer,loss='categroical_crossentropy')
    model.fit(x=train_ds,epochs=10,validation_ds=validation_ds)

if __name__=='__main__':
    train(None)