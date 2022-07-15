'''
uses test.py from masterScripts/DNN to train different DNNs on different datasets
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
import datetime
import pandas as pd

sys.path.append('/mnt/HDD12TB/masterScripts')
from DNN.test import test

overwrite = False
models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'alexnet', 'cornet_s', 'inception_v3', 'vgg19', 'PredNetImageNet']
dataset = 'imagenet16'#, 'places365_standard']
datasetPath = f'/home/dave/Datasets/{dataset}'
occTypesTest = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 'polkadot','polkasquare','crossBarOblique','crossBarCardinal', 'naturalUntexturedCropped2']
occLevelsTest = [.2,.4,.6,.8,.9]
visibilities = (1-np.array(occLevelsTest)) * 100
visibilities = [int(np.round(x,1)) for x in visibilities]
visibilities.reverse()
occlusionMethodsNoLevels = ['unoccluded','naturalTextured','naturalTextured2']
coverage = .5
batchSize = 8
workers = 8
invert = False
occColour = [(0,0,0),(255,255,255)]
lineFigSize = (7,4)

outDir = f'DNN/analysis/results/behaviouralStim'
os.makedirs(outDir, exist_ok=True)
accuracyPath = os.path.join(outDir, 'accuracy.pkl')
if os.path.isfile(accuracyPath) and not overwrite:
    accuracies = pickle.load(open(accuracyPath, 'rb'))
else:
    accuracies = {}

for modelName in models:

    if not modelName in accuracies:
        accuracies[modelName] = {}

    for occTypeTest in occTypesTest:

        if not occTypeTest in accuracies[modelName]:
            accuracies[modelName][occTypeTest] = {}

        occTypesTrain = ['unoccluded', 'naturalTextured', occTypeTest]
        if modelName == 'vgg19':
            occTypesTrain += ['mixedOccluders']

        for otr, occTypeTrain in enumerate(occTypesTrain):

            if not occTypeTrain in accuracies[modelName][occTypeTest]:
                accuracies[modelName][occTypeTest][occTypeTrain] = {'acc1': np.empty(len(occLevelsTest)),
                                                                    'acc5': np.empty(len(occLevelsTest))}

                modelDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occTypeTrain)

                if occTypeTrain not in occlusionMethodsNoLevels:
                    modelDir = os.path.join(modelDir, 'mixedLevels')

                weightFiles = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))
                weightsPath = weightFiles[-1]

                for olt, occLevelTest in enumerate(occLevelsTest):

                    # call script
                    acc1, acc5 = test(modelName, datasetPath, batchSize, weightsPath, workers, occTypeTest, occLevelTest, 1, occColour, invert)

                    # store accuracies
                    accuracies[modelName][occTypeTest][occTypeTrain]['acc1'][olt] = acc1
                    accuracies[modelName][occTypeTest][occTypeTrain]['acc5'][olt] = acc5

                    print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | model: {modelName} | trained on: {occTypeTrain} | testing on: {occTypeTest} at {occLevelTest} occluded |  acc1: {acc1} | acc5: {acc5}')

                pickle.dump(accuracies, open(accuracyPath, 'wb'))

    # get performance on unoccluded images
    if not 'unoccluded' in accuracies[modelName]:
        accuracies[modelName]['unoccluded'] = {}
    occTypesTrainAll = occTypesTrain + occTypesTest # need unoccluded performance for models trained on each occluder
    if modelName == 'vgg19':
        occTypesTrain += ['mixedOccluders']
    for otr, occTypeTrain in enumerate(occTypesTrainAll):
        if not occTypeTrain in accuracies[modelName]['unoccluded']:

            modelDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occTypeTrain)

            if occTypeTrain not in occlusionMethodsNoLevels:
                modelDir = os.path.join(modelDir, 'mixedLevels')

            weightFiles = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))
            weightsPath = weightFiles[-1]

            acc1, acc5 = test(modelName, datasetPath, batchSize, weightsPath, workers, 'unoccluded', 0, occColour, invert)

            accuracies[modelName]['unoccluded'][occTypeTrain] = {'acc1': acc1, 'acc5': acc5}

            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | model: {modelName} | trained on: {occTypeTrain} | testing on: unoccluded |  acc1: {acc1} | acc5: {acc5}')

            pickle.dump(accuracies, open(accuracyPath, 'wb'))


    # plots
    occTypesTrain = ['unoccluded', 'naturalTextured', 'sameAsTest']
    if modelName == 'vgg19':
        occTypesTrain += ['mixedOccluders']
    for occTypeTrain in occTypesTrain:
        for acc in ['acc1']:

            # accuracy plots

            occPerf = np.empty((len(occTypesTest), len(occLevelsTest)))

            if occTypeTrain != 'sameAsTest':
                unoccPerf = accuracies[modelName]['unoccluded'][occTypeTrain][acc]
                for o, occluder in enumerate(occTypesTest):
                    occPerf[o, :] = accuracies[modelName][occluder][occTypeTrain][acc]
            else:
                unoccPerf = np.empty((len(occTypesTest)))
                for o, occluder in enumerate(occTypesTest):
                    unoccPerf[o] = accuracies[modelName]['unoccluded'][occluder][acc]
                    occPerf[o,:] = accuracies[modelName][occluder][occluder][acc]
                unoccPerf = np.mean(unoccPerf, axis=0)

            unoccPerf *= 100
            occPerf *= 100

            plt.figure(figsize=(10, 3))
            colours = list(mcolors.TABLEAU_COLORS.keys())
            for v, visibility in enumerate(visibilities):
                barWidth = 1 / 6
                xshift = (v * barWidth) - (2 * barWidth)
                plt.bar(np.arange(len(occTypesTest)) + xshift, occPerf[:,len(visibilities)-(v+1)], width=barWidth, color=mcolors.TABLEAU_COLORS[colours[0]], alpha=1-(visibility/100), label=visibility)
            plt.xticks(np.arange(len(occTypesTest)), labels=occTypesTest, rotation=25, ha='right')
            plt.tick_params(direction='in')
            plt.axhline(y=unoccPerf, color=mcolors.TABLEAU_COLORS[colours[1]], linestyle='dashed')
            plt.axhline(y=100 / 16, color='k', linestyle='dotted')
            plt.ylim(0, 100)
            plt.title(f'{acc}, {modelName} trained on {occTypeTrain}')
            plt.ylabel('accuracy (%)')
            plt.legend(title='visibility (%)', bbox_to_anchor=(1.04, 1), loc='upper left', frameon=False)
            plt.tight_layout()
            outDir = f'DNN/analysis/results/behaviouralStim/{modelName}'
            os.makedirs(outDir, exist_ok=True)
            plt.savefig(f'{outDir}/accuracy_{occTypeTrain}_{acc}.png')
            plt.show()

            # plot mean across all occluder types for each occluder level
            occPerfMean = np.mean(occPerf, axis=0)

            plt.figure(figsize=(2.5, 3))
            colours = list(mcolors.TABLEAU_COLORS.keys())
            for v, visibility in enumerate(visibilities):
                x_pos = len(occPerfMean) - (v + 1)
                plt.bar(x_pos, occPerfMean[v], width=1, color=mcolors.TABLEAU_COLORS[colours[0]], alpha=(visibility / 100), label=visibilities[x_pos])
            plt.xticks(np.arange(len(occPerfMean)), labels=visibilities)
            plt.tick_params(direction='in')
            plt.axhline(y=np.mean(unoccPerf), color=mcolors.TABLEAU_COLORS[colours[1]], linestyle='dashed')
            plt.axhline(y=100 / 16, color='k', linestyle='dotted')
            plt.ylim(0, 100)
            plt.title(f'{acc} mean x occluders', size=10)
            plt.xlabel('visibility (%)')
            plt.ylabel('accuracy (%)')
            plt.tight_layout()
            plt.savefig(f'{outDir}/accMeanXocc_{occTypeTrain}_{acc}.png')
            plt.show()

            # plot performance after subtracting mean accuracy for each occlusion level
            occPerfMeanOcc = np.mean(occPerf, axis=0)
            occPerfSub = occPerf - np.tile(np.expand_dims(occPerfMeanOcc, axis=0), (9, 1))
            plt.figure(figsize=(10, 3))
            colours = list(mcolors.TABLEAU_COLORS.keys())
            for v, visibility in enumerate(visibilities):
                means = occPerfSub[:, len(visibilities)-(v+1)]
                barWidth = 1 / 6
                xshift = (v * barWidth) - (2 * barWidth)
                plt.bar(np.arange(len(occTypesTest)) + xshift, means, width=barWidth, color=mcolors.TABLEAU_COLORS[colours[0]], alpha=1-(visibility / 100), label=visibility)
            plt.xticks(np.arange(len(occTypesTest)), labels=occTypesTest, rotation=25, ha='right')
            plt.tick_params(direction='in')
            plt.title(f'{acc}, {modelName} trained on {occTypeTrain}, subtracting mean of each occlusion level')
            plt.legend(title='visibility (%)', bbox_to_anchor=(1.04, 1), loc='upper left', frameon=False)
            plt.ylabel('accuracy (%)')
            plt.tight_layout()
            plt.savefig(f'{outDir}/accSubMeanOcc_{occTypeTrain}_{acc}.png')
            plt.show()

            # compare with human performance
            if acc == 'acc1':

                # raw scores
                humanAccData = pickle.load(open(f'behavioural/v3_variousTypesLevels/accuracies.pkl', 'rb'))
                CNNacc = np.empty_like(occPerf)
                fig, ax = plt.subplots(figsize=(6.7, 4))
                for o in range(len(occTypesTest)):
                    humanAcc = np.mean(humanAccData['occ'], axis=0)[o,:]
                    CNNacc[o,:] = occPerf[o,::-1]
                    ax.scatter(humanAcc, CNNacc[o,:], color=mcolors.TABLEAU_COLORS[colours[o]], label = occTypesTest[o])

                humanAcc = np.mean(humanAccData['occ'], axis=0).flatten()
                CNNacc = CNNacc.flatten()
                R = np.corrcoef(humanAcc, CNNacc)[1,0]

                plt.xticks(np.arange(0,101,20))
                plt.xlim(0,100)
                plt.ylim(0,100)
                plt.yticks(np.arange(0,101,20))
                plt.tick_params(direction='in')
                plt.xlabel('human accuracy (%)')
                plt.ylabel('CNN accuracy (%)')

                # plot regression line
                b, a = np.polyfit(humanAcc, CNNacc, deg=1)
                xseq = np.linspace(0, 100, num=100)
                ax.plot(xseq, a + b * xseq, color="k", lw=2.5)

                plt.title(f'human performance versus {modelName}\ntrained on {occTypeTrain}, R = {R:.2f}')
                plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', frameon=False)
                plt.tight_layout()
                plt.savefig(f'{outDir}/corrWithHumans_{occTypeTrain}_{acc}.png')
                plt.show()

                # mean for occ level subtracted
                humanAccData = pickle.load(open(f'behavioural/v3_variousTypesLevels/accuracies.pkl', 'rb'))
                occPerfMeanOcc = np.mean(humanAccData['occ'], axis=1)
                humanAccDataMeanSub = humanAccData['occ'] - np.tile(np.expand_dims(occPerfMeanOcc, axis=1), (1, 9, 1))
                CNNacc = np.empty_like(occPerfSub)
                fig, ax = plt.subplots(figsize=(6.7,4))
                for o in range(len(occTypesTest)):
                    humanAcc = np.mean(humanAccDataMeanSub, axis=0)[o,:]
                    CNNacc[o, :] = occPerfSub[o, ::-1]
                    ax.scatter(humanAcc, CNNacc[o,:], color=mcolors.TABLEAU_COLORS[colours[o]], label = occTypesTest[o])
                humanAcc = np.mean(humanAccDataMeanSub, axis=0).flatten()
                CNNacc = CNNacc.flatten()

                R = np.corrcoef(humanAcc, CNNacc)[1,0]

                plt.tick_params(direction='in')
                plt.xlabel('human accuracy (%)')
                plt.ylabel('CNN accuracy (%)')
                plt.xlim(-40,40)
                plt.ylim(-40, 40)

                # plot regression line
                b, a = np.polyfit(humanAcc, CNNacc, deg=1)
                xseq = np.linspace(min(min(humanAcc),min(CNNacc)), max(max(humanAcc),max(CNNacc)), num=100)
                ax.plot(xseq, a + b * xseq, color="k", lw=2.5)

                plt.title(f'human performance versus {modelName}\ntrained on {occTypeTrain}, R = {R:.2f}')
                plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', frameon=False)
                plt.tight_layout()
                plt.savefig(f'{outDir}/corrWithHumSubMeanOcc_{occTypeTrain}_{acc}.png')
                plt.show()



