from model import TLModel
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
import os
import argparse
import wandb

def train(config,wandbLog=False):
    model = TLModel(baseModel=config['baseModel'],
                    epochs=config['epochs'],
                    pTrainLayers=config['pTrainLayers'],
                    denseNeurons=config['denseNeurons']
                    )
    optimizer = Adam(lr=config['lr'])

    train_ds = ImageDataGenerator(
        rotation_range=config['rotation_range'],
        width_shift_range=config['shifting_range'],
        height_shift_range=config['shifting_range'],
        zoom_range=config['shifting_range'],
        horizontal_flip=config['flip'],
        vertical_flip=config['flip'],
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
        target_size=(config['imageSize'], config['imageSize']),
        batch_size=config['bs'],
        class_mode='categorical'
    )

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = History()

    epochs = config['epochs']
    epochUpdate = config['epochUpdate']
    oldAcc = 0

    for epoch in range(epochs):

        if epoch % epochUpdate == 0:
            model.startLayers(epoch)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
            x=train_gen,
            steps_per_epoch=9000 // config['bs'],
            epochs=1,
            validation_data=valid_gen,
            validation_steps=1000 // config['bs'],
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


def updateConfig(args, config):
    if args.baseModel:
        config['baseModel'] = args.baseModel

    if args.lr:
        config['lr'] = args.lr

    if args.rotation_range:
        config['rotation_range'] = args.rotation_range

    if args.shifting_range:
        config['shifting_range'] = args.shifting_range

    if args.flip:
        config['flip'] = args.flip

    if args.imageSize:
        config['imageSize'] = args.imageSize

    if args.bs:
        config['bs'] = args.bs

    if args.epochs:
        config['epochs'] = args.epochs

    if args.epochUpdate:
        config['epochUpdate'] = args.epochUpdate

    if args.pTrainLayers:
        config['pTrainLayers'] = args.pTrainLayers

    if args.denseNeurons:
        config['denseNeurons'] = args.denseNeurons

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preparation')
    parser.add_argument('--baseModel', dest='baseModel', type=str, help='Base Model for Transfer Learning')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate')
    parser.add_argument('--rotation_range', dest='rotation_range', type=int, help='Rotation Augmentation')
    parser.add_argument('--shifting_range', dest='shifting_range', type=float, help='Hieght,Width and Zoom Shift Augmentations')
    parser.add_argument('--flip', dest='flip', type=bool, help='Horizontal and Vetical Flip')
    parser.add_argument('--imageSize', dest='imageSize', type=int, help='Image Size')
    parser.add_argument('--bs', dest='bs', type=int, help='Batch Size')
    parser.add_argument('--epochs', dest='epochs', type=int, help='Epochs')
    parser.add_argument('--epochUpdate', dest='epochUpdate', type=int,
                        help='Epochs to update training layers for updating')
    parser.add_argument('--pTrainLayers', dest='pTrainLayers', type=float, help='Percentage of Train layers to train')
    parser.add_argument('--denseNeurons', dest='denseNeurons', type=int, help='Neurons in Dense Layer')

    config = {
        'baseModel': 'EffnetV2B0',
        'lr': 1e-3,
        'rotation_range': 15,
        'shifting_range': 0.1,
        'flip': True,
        'imageSize': 256,
        'bs': 32,
        'epochs': 10,
        'epochUpdate': 2,
        'pTrainLayers': 0.1,
        'denseNeurons': 1000,
    }

    config = updateConfig(parser.parse_args(), config)
    train(config)