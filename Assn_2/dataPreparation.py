from sklearn.model_selection import train_test_split
import os
import zipfile
from tqdm import tqdm
import argparse


def main(datasetPath):
    trainPath = os.path.join('inaturalist_12K', 'train')
    validPath = os.path.join('inaturalist_12K', 'valid')
    testPath = os.path.join('inaturalist_12K', 'val')

    if not os.path.exists(datasetPath) and not (os.path.exists(trainPath) and os.path.exists(testPath)):
        print('Zip file not found\nExiting Program')
        return

    if not (os.path.exists(trainPath) and os.path.exists(testPath)):
        print('Extracting Zip')

        with zipfile.ZipFile(datasetPath, 'r') as z1:
            z1.extractall('')

    classes = [i for i in os.listdir(trainPath) if i[0] != '.']

    if not os.path.exists(validPath):
        print('Creating Validation Set')
        os.makedirs(validPath)

        x, y = [], []

        for c in tqdm(classes):
            os.makedirs(os.path.join(validPath, c))

            names = os.listdir(os.path.join(trainPath, c))
            x += names
            y += [c] * len(names)

        trainX, valX, trainY, valY = train_test_split(x, y, stratify=y, test_size=0.1)

        for i, j in zip(valX, valY):
            os.rename(os.path.join(trainPath, j, i), os.path.join(validPath, j, i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preparation')
    parser.add_argument('--path', dest='path', type=str, help='Path to the dataset')
    main(parser.parse_args().path)