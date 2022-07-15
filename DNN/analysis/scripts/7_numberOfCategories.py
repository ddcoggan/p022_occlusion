import os
import glob
import sys
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import pickle
import time
#time.sleep(72000)

sys.path.append('/mnt/HDD12TB/masterScripts')
from DNN.analysis.scripts.test import test

overwrite = False
models = ['alexnet']#['alexnet']#, 'CORnet_S']
datasets = ['imagenet16', 'imagenet1000']#, 'places365_standard']
occlusionMethodsTrain = ['naturalTextured2', 'unoccluded']
occlusionMethodsTest = [os.path.basename(x) for x in sorted(glob.glob('DNN/images/occluders/*'))]
occlusionMethodsNoLevels = ['naturalTextured','naturalTextured2']
for o in occlusionMethodsNoLevels:
    occlusionMethodsTest.remove(o)
coveragesTest = [0.,.1,.2,.3,.4,.5,.6,.7,.8,.9]
batchSize = 32
workers = 8
invert = False
colours = [(0,0,0),(127,127,127),(255,255,255)]
lineFigSize = (5,4)
matFigSizeType = (5.5,5.5)
matFigSizeLevel = (7,8)

# accuracy across coverages and occlusion Type
for model in models:

    for dataset in datasets:

        datasetPath = f'/home/dave/Datasets/{dataset}'

        for occlusionMethodTrain in occlusionMethodsTrain:

            modelDir = os.path.join(os.getcwd(), 'DNN/data', model, dataset, 'fromPretrained', occlusionMethodTrain)

            weightFiles = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))
            weightsPath = weightFiles[-1]

            outDir = f'DNN/analysis/results/numberOfCategories/{model}/{dataset}/{occlusionMethodTrain}/acrossOcclusionTypesLevels'
            os.makedirs(outDir, exist_ok=True)

            accuracyPath = os.path.join(outDir, 'accuracy.pkl')
            if not os.path.isfile(accuracyPath) or overwrite:

                accuracyMats = {'acc1': np.empty(shape=(len(occlusionMethodsTest), len(coveragesTest))),
                                'acc5': np.empty(shape=(len(occlusionMethodsTest), len(coveragesTest)))}

                for ote, occlusionMethodTest in enumerate(occlusionMethodsTest):

                    for cte, coverageTest in enumerate(coveragesTest):

                        # call script
                        acc1, acc5 = test(model, datasetPath, batchSize, weightsPath, workers, occlusionMethodTest, coverageTest, colours, invert)
                        accuracyMats['acc1'][ote, cte] = acc1
                        accuracyMats['acc5'][ote, cte] = acc5

                        print(f'{model} | dataset: {dataset} | trained on {occlusionMethodTrain} | tested on {occlusionMethodTest} at {coveragesTest[cte]} | acc1 {acc1} | acc5 {acc5}')

                # save accuracyMats
                pickle.dump(accuracyMats, open(accuracyPath, 'wb'))

        for occlusionMethodTrain in occlusionMethodsTrain:

            outDir = f'DNN/analysis/results/numberOfCategories/{model}/{dataset}/{occlusionMethodTrain}/acrossOcclusionTypesLevels'

            modelDir = os.path.join(os.getcwd(), 'DNN/data', model, dataset, 'fromPretrained', occlusionMethodTrain)
            accuracyPath = os.path.join(outDir, 'accuracy.pkl')
            accuracyMats = pickle.load(open(accuracyPath, 'rb'))

            # plots
            for acc in accuracyMats.keys():

                # matrix plot
                fig, ax = plt.subplots(figsize = matFigSizeLevel)
                im = ax.imshow(accuracyMats[acc]*100, vmin = 0, vmax = 100, cmap='rainbow')
                ax.set_yticks(np.arange(len(occlusionMethodsTest)))
                ax.set_xticks(np.arange(len(coveragesTest)))
                ax.set_yticklabels(occlusionMethodsTest)
                ax.set_xticklabels(coveragesTest)
                ax.tick_params(direction='in')
                ax.set_xlabel('occlusion level tested on')
                ax.set_ylabel('occlusion type tested on')
                fig.colorbar(im, fraction = .0231)
                for i in range(len(coveragesTest)):
                    for j in range(len(occlusionMethodsTest)):
                        text = ax.text(i, j, f'{int(accuracyMats[acc][j, i]*100)}',
                                       ha="center", va="center", color="k")
                ax.set_title(f'{dataset}, {occlusionMethodTrain}, top {acc[3]} accuracy')
                #fig.tight_layout()
                plt.savefig(f'{outDir}/{acc}_mat.png')
                plt.show()


