from model import CNNModel
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
import os
import argparse
import wandb

def train(wandbLog=False):
    model = CNNModel()
    optimizer = Adam(lr=0.001)

    train_ds = ImageDataGenerator(
        rotation_range = 15,
        width_shift_range =0.1,
        height_shift_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True,
        vertical_flip = True,
    )

    train_gen = train_ds.flow_from_directory(
        os.path.join('inaturalist_12K', 'train'),
        target_size=(config['imageSize'], config['imageSize']),
        batch_size=config['bs'],
        class_mode='categorical'
    )

    validation_ds = ImageDataGenerator(
    )

    valid_gen = validation_ds.flow_from_directory(
        os.path.join('inaturalist_12K', 'valid'),
        shuffle=False,
        target_size=(256,256),
        batch_size=32,
        class_mode='categorical'
    )

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = History()

    epochs = 2
    oldAcc = 0

    for epoch in range(epochs):

        model.fit(
            x=train_gen,
            steps_per_epoch=9000 // 32,
            epochs=1,
            validation_data=valid_gen,
            validation_steps=1000 // 32,
            callbacks=[history]
        )

        if history.history['val_accuracy'][-1] > oldAcc:
            print('Saving Model')
            model.model.save("BestModel")
            oldAcc = history.history['val_accuracy'][-1]

        if wandbLog:
            wandb.log({"Train Loss": history.history['loss'][-1], "Val Loss": history.history['val_loss'][-1],
                       "Train Acc": history.history['accuracy'][-1],
                       "Val Acc": history.history['val_accuracy'][-1],
                       "epoch": epoch
                       })
    train(None)