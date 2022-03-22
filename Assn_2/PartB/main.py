from model import TLModel
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.image import ImageDataGenerator
import os

def train(config):

    model = TLModel()
    optimizer = Adam()

    train_ds = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zca_whitening=True,
    )

    train_gen = train_ds.flow_from_directory(
      os.path.join('inaturalist_12K','train'),
      target_size=(256,256),
      batch_size=32,
      class_mode='categorical'
    )

    validation_ds = ImageDataGenerator(
    )

    valid_gen = validation_ds.flow_from_directory(
      os.path.join('inaturalist_12K','valid'),
      target_size=(256,256),
      batch_size=32,
      class_mode='categorical'
    )

    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(
      x=train_gen,
      steps_per_epoch=9000//32,
      epochs=10,
      validation_data=valid_gen,
      validation_steps=1000//32
    )

if __name__=='__main__':
    train(None)