from keras.models import load_model,Model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import argparse
import os
import numpy as np
from PartA.guidedBackProp import main as gbpen

classes = ['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptillia']

#Evaluate Model on Test Data
def main(args):

    #Load Model
    model = load_model(args.path)

    #Dataloader
    test_ds = ImageDataGenerator(
    )

    test_gen = test_ds.flow_from_directory(
        os.path.join('inaturalist_12K', 'val'),
        shuffle=True,
        target_size=(args.imageSize, args.imageSize),
        batch_size=args.bs,
        class_mode='categorical'
    )

    #Evaluate
    metric = model.evaluate(test_gen,return_dict=True)

    #Load 1 batch for plots
    xBatch,yBatch = next(test_gen)
    yPreds = model.predict(xBatch)

    fig,axs = plt.subplots(3,10,figsize=(50,15))
    fig.suptitle(f"Overall Accuracy:{metric['accuracy']}",fontsize=25)

    for i in range(3):
        for j in range(10):

            axs[i][j].imshow(xBatch[i*10+j]/255,aspect='auto')
            axs[i][j].set_xticklabels([])
            axs[i][j].set_yticklabels([])

            axs[i][j].set_title(f"Actual:{classes[np.argmax(yBatch[i*10+j])]}, Predicted:{classes[np.argmax(yPreds[i*10+j])]}",fontsize=15)


    plt.savefig('TestOutput.png')

    #Visualize filters and Guided BackProp
    if args.visualizeFilters:
        #Filter visualization
        filters, bias = model.layers[0].get_weights()

        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        fig, axs = plt.subplots(8, 8, figsize=(20, 20))
        fig.suptitle("Filters of First Layer", fontsize=20)

        for i in range(8):
            for j in range(8):
                axs[i][j].imshow(filters[:, :, :, i * 8 + j], aspect='auto')
                axs[i][j].set_xticklabels([])
                axs[i][j].set_yticklabels([])

        plt.savefig('Filters.png')

        #Feature Map output of first conv layer visualization
        newModel = Model(inputs=model.inputs, outputs=model.layers[1].output)

        featureMap = newModel.predict(xBatch)

        fig, axs = plt.subplots(8, 8, figsize=(20, 20))
        fig.suptitle("Feature Map of First Layer", fontsize=20)

        for i in range(8):
            for j in range(8):
                axs[i][j].imshow(featureMap[0, :, :, i * 8 + j], aspect='auto')
                axs[i][j].set_xticklabels([])
                axs[i][j].set_yticklabels([])

        plt.savefig('FeatureMap.png')

        #Guided Back Prop Visualization
        gbpen(model,xBatch)




if __name__=='__main__':
    #Parse cmd args
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument('--path',dest='path',type=str,help="Path to the model used for testing",default='BestModel')
    parser.add_argument('--imageSize',dest='imageSize',type=int,help="Size of image",default="256")
    parser.add_argument('--bs',dest='bs',type=int,help="Batch Size",default=64)
    parser.add_argument('--visualizeFilters',dest='visualizeFilters',type=bool,help="Boolean to control visualizing of the filter",default=False)

    args = parser.parse_args()
    main(args)