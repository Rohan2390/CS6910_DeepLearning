from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import argparse
import os

def main(args):

    model = load_model(args.modelPath)

    test_ds = ImageDataGenerator(
    )

    test_gen = test_ds.flow_from_directory(
        os.path.join('inaturalist_12K', 'val'),
        shuffle=True,
        target_size=(args.imageSize, args.imageSize),
        batch_size=args.bs,
        class_mode='categorical'
    )

    model.evaluate(test_gen)
    yPreds = model.predict(test_gen)
    print(len(yPreds))

    #xBatch,yBatch = test_gen.flow()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument('--path',dest='path',type=str,help="Path to the model used for testing",default='BestModel')
    parser.add_argument('--imageSize',dest='imageSize',type=int,help="Size of image",default="256")
    parser.add_argument('--bs',dest='bs',type=int,help="Batch Size",default=64)
    parser.add_argument('--visualizeFilters',dest='visualizeFilters',type=bool,help="Boolean to control visualizing of the filter",default=False)

    args = parser.parse_args()
    main(args)
