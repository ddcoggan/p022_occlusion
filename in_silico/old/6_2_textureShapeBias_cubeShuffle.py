'''
uses test.py from masterScripts/DNN to train different DNNs on different datasets
if script is interrupted or additional epochs are required at a later time, script will continue from last recorded epoch
'''

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
import pandas as pd
#time.sleep(72000)

sys.path.append('/mnt/HDD12TB/masterScripts')
from DNN.analysis.scripts.test import test

overwrite = True
models = ['alexnet']#['alexnet']#, 'CORnet_S']
datasets = ['imagenet16']#, 'places365_standard']
datasetsTest = {'original': 'imagenet16',
                'cubeShuffle': 'imagenet16_cubeShuffle'}
occlusionMethods = [os.path.basename(x) for x in sorted(glob.glob('DNN/images/occluders/*'))]
occlusionMethods.append('unoccluded')
occludersNoLevels = ['unoccluded','naturalTextured','naturalTextured2']

coveragesTrain = [.1,.2,.4,.8]
batchSize = 128
workers = 8
invert = False


# accuracy across occlusion methods / within coverage
for model in models:

    for dataset in datasets:

        outDir = f'DNN/analysis/results/shapeTextureBias/{model}/{dataset}/cubeShuffle'
        os.makedirs(outDir, exist_ok=True)
        accuracyPath = os.path.join(outDir, 'shapeTextureBias.pkl')
        if os.path.isfile(accuracyPath) and not overwrite:
            accuracies = pickle.load(open(accuracyPath, 'rb'))
        else:
            accuracies = {'occType': [], 'occLevel': [], 'shuffledOriginal': [], 'acc1': [], 'acc5': []}
            for coverage in coveragesTrain:

                for otr, occTypeTrain in enumerate(occlusionMethods):
                    if occTypeTrain in occludersNoLevels:
                        modelDir = os.path.join(os.getcwd(), 'DNN/data', model, dataset, 'fromPretrained/unoccluded')
                    else:
                        modelDir = os.path.join(os.getcwd(), 'DNN/data', model, dataset, 'fromPretrained', occTypeTrain, f'{int(coverage * 100)}')


                    weightFiles = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))
                    weightsPath = weightFiles[-1]

                    for d, datasetTest in enumerate(datasetsTest.keys()):

                        datasetTestName = datasetsTest[datasetTest]
                        datasetPath = f'/home/dave/Datasets/{datasetTestName}'

                        # call script
                        acc1, acc5 = test(model, datasetPath, batchSize, weightsPath, workers, 'unoccluded', 0, [], invert)

                        accuracies['occType'].append(occTypeTrain)
                        accuracies['occLevel'].append(coverage)
                        accuracies['shuffledOriginal'].append(datasetTest)
                        accuracies['acc1'].append(acc1)
                        accuracies['acc5'].append(acc5)

                        print(f'{model} | coverage: {coverage} | trained on {occlusionMethods[otr]} | tested on {datasetTestName} | acc1 {acc1} | acc5 {acc5}')

            # save accuracies
            accuracies = pd.DataFrame(accuracies)
            pickle.dump(accuracies, open(accuracyPath, 'wb'))

        # plot by coverage
        for coverage in coveragesTrain:

            df = accuracies[accuracies['occLevel'] == coverage]

            for acc in ['acc1','acc5']:

                # plot
                fig, ax = plt.subplots(figsize = (6,3))
                nBars = len(datasetsTest.keys())
                barWidth = 1 / (nBars + 1)  # add 1 for gap

                for d, datasetTest in enumerate(datasetsTest.keys()):
                    values = df[acc][df['shuffledOriginal'] == datasetTest]
                    xoffset = -(barWidth/2)*(nBars/2) + (d*barWidth)
                    plt.bar(np.arange(len(occlusionMethods))+xoffset, values, label=datasetTest, width=barWidth)
                plt.tick_params(direction='in')
                plt.xticks(np.arange(len(occlusionMethods)), occlusionMethods, rotation=45, ha="right", rotation_mode="anchor")
                ax.set_xlabel('occlusion type trained on')
                ax.set_ylabel('accuracy')
                plt.ylim(0,1)
                plt.title(f'accuracy at {int(coverage*100)}% coverage', size=8)
                plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                plt.tight_layout()
                fig.savefig(os.path.join(outDir, f'barPlot_{int(coverage*100)}_{acc}.png'))
                plt.show()
                plt.close()

        # plot by occlusion type
        for occTypeTrain in occlusionMethods[:-1]:

            df = accuracies[accuracies['occType'] == occTypeTrain]

            for acc in ['acc1','acc5']:

                # plot
                fig, ax = plt.subplots(figsize=(4, 3))
                nBars = len(datasetsTest.keys())
                barWidth = 1 / (nBars + 1)  # add 1 for gap

                for d, datasetTest in enumerate(datasetsTest.keys()):
                    values = df[acc][df['shuffledOriginal'] == datasetTest]
                    xoffset = -(barWidth / 2) * (nBars / 2) + (d * barWidth)
                    plt.bar(np.arange(len(coveragesTrain)) + xoffset, values, label=datasetTest, width=barWidth)
                plt.tick_params(direction='in')
                plt.xticks(np.arange(len(coveragesTrain)), coveragesTrain, rotation=0, ha="right", rotation_mode="anchor")
                ax.set_xlabel('occlusion level trained on')
                ax.set_ylabel('accuracy')
                plt.ylim(0, 1)
                plt.title(f'accuracy for {occTypeTrain}', size=8)
                plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                plt.tight_layout()
                fig.savefig(os.path.join(outDir, f'barPlot_{occTypeTrain}_{acc}.png'))
                plt.show()
                plt.close()

        # shape texture index
        for occTypeTrain in occlusionMethods[:-1]:

            df = accuracies[accuracies['occType'] == occTypeTrain]

            for acc in ['acc1', 'acc5']:

                # plot
                fig, ax = plt.subplots(figsize=(3, 3))
                values = []
                for c in coveragesTrain:
                    originalValue = df[acc][(df['shuffledOriginal'] == 'original') &
                                            (df['occLevel'] == c)].item()
                    shuffledValue = df[acc][(df['shuffledOriginal'] == 'cubeShuffle') &
                                            (df['occLevel'] == c)].item()
                    biasIdx = (originalValue - shuffledValue) / (originalValue + shuffledValue)
                    values.append(biasIdx)
                plt.bar(np.arange(len(coveragesTrain)), values)
                plt.tick_params(direction='in')
                plt.xticks(np.arange(len(coveragesTrain)), coveragesTrain, rotation=0, ha="right", rotation_mode="anchor")
                ax.set_xlabel('occlusion level trained on')
                ax.set_ylabel('shape bias index')
                plt.ylim(0, 1)
                plt.title(f'accuracy for {occTypeTrain}', size=8)
                plt.tight_layout()
                fig.savefig(os.path.join(outDir, f'barPlot_biasIndex_{occTypeTrain}_{acc}.png'))
                plt.show()
                plt.close()



