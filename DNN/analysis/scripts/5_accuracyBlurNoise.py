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
import datetime
import scipy.stats

sys.path.append('/mnt/HDD12TB/masterScripts')
from DNN.analysis.scripts.blurNoiseTest import test

overwrite = False
models = ['alexnet', 'vgg19', 'cornet_s', 'resnet152']#, 'inception_v3', 'PredNetImageNet']
dataset = 'imagenet16'#, 'places365_standard']
datasetPath = f'/home/dave/Datasets/{dataset}'
occTypesTrain = ['unoccluded']
occTypesTrain += [os.path.basename(x) for x in sorted(glob.glob('DNN/images/occluders/*'))]
occTypesNoLevels = ['unoccluded','naturalTextured','naturalTextured2']
levels = {'blur': [0,1,2,4,8],
          'noise': [1,.8,.4,.2,.1]}

batchSizes = {'alexnet': 1024,
              'vgg19': 128,
              'cornet_s': 256,
              'resnet152': 64,
              'inception_v3': 32,
              'PredNetImageNet': 8}
workers = 8
lineFigSize = (5,4)

outDir = f'DNN/analysis/results/accuracyBlurNoise'
os.makedirs(outDir, exist_ok=True)
accuracyPath = os.path.join(outDir, 'accuracy.pkl')

if os.path.isfile(accuracyPath) and not overwrite:
    accuracies = pickle.load(open(accuracyPath, 'rb'))
else:
    accuracies = {}

for blurNoise in ['noise','blur']:

    theseLevels = levels[blurNoise]

    if not blurNoise in accuracies.keys():
        accuracies[blurNoise] = {}

    for modelName in models:

        batchSize = batchSizes[modelName]

        if not modelName in accuracies[blurNoise].keys():
            accuracies[blurNoise][modelName] = {}

        for occTypeTrain in occTypesTrain:

            if modelName == 'alexnet' and occTypeTrain == 'unoccluded':
                saveOutputs = True
            else:
                saveOutputs = False

            if not occTypeTrain in accuracies[blurNoise][modelName].keys():
                accuracies[blurNoise][modelName][occTypeTrain] = {'acc1': np.empty(len(theseLevels)),
                                                                  'acc5': np.empty(len(theseLevels))}

            if occTypeTrain in occTypesNoLevels:
                modelDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occTypeTrain)
            else:
                modelDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occTypeTrain, 'mixedLevels')
            weightFiles = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))
            weightsPath = weightFiles[-1]

            for l, level in enumerate(theseLevels):

                acc1, acc5 = test(modelName, datasetPath, batchSize, weightsPath, workers, blurNoise, level, saveOutputs)
                accuracies[blurNoise][modelName][occTypeTrain]['acc1'][l] = acc1
                accuracies[blurNoise][modelName][occTypeTrain]['acc5'][l] = acc5

                print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | testing on: {blurNoise} at {level} level | model: {modelName} | trained on: {occTypeTrain} | acc1: {acc1} | acc5: {acc5}')

                # save accuracyMats
                pickle.dump(accuracies, open(accuracyPath, 'wb'))


# plots per test occlusion type
for blurNoise in ['blur','noise']:
    theseLevels = levels[blurNoise]
    for occTypeTrain in occTypesTrain:
        for acc in ['acc1','acc5']:

            # line plot
            colours = list(mcolors.TABLEAU_COLORS.keys())
            xpos = np.arange(len(theseLevels))
            plt.figure(figsize=lineFigSize)

            models = ['alexnet','vgg19']
            for m, model in enumerate(models):
                yvals = np.array(accuracies[blurNoise][model][occTypeTrain][acc]) * 100
                if model.endswith('ImageNet'):
                    modelLabel = model[:-8]
                else:
                    modelLabel = model
                plt.plot(xpos, yvals, label=f'{modelLabel}', color=colours[m])

            plt.xlabel(f'{blurNoise} level tested on')
            plt.xticks(xpos, theseLevels)
            plt.tick_params(direction='in')
            plt.ylabel('accuracy (%)')
            plt.ylim((0, 100))
            plt.title(f'train: {occTypeTrain}, test: {blurNoise}, {acc}')
            plt.legend(title='occlusion type\ntrained on', bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)
            #plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{outDir}/{blurNoise}_{occTypeTrain}_{acc}_line.png')
            plt.show()

# plots of selected networks and occlusion types averaged together
lineFigSize = (10,5)
models2use = ['alexnet', 'vgg19', 'cornet_s', 'resnet152']
occTypesTrain = ['unoccluded','naturalTextured2']
occTypeClusters = {'barHorz': ['barHorz02', 'barHorz04','barHorz08', 'barHorz16'],
                   'barVert': ['barVert02', 'barVert04','barVert08', 'barVert16'],
                   'crossBar': ['crossBarCardinal', 'crossBarOblique'],
                   'polka': ['polkadot','polkasquare'],
                   'mudSplash': ['mudSplash'],
                   'natural': ['naturalTexturedCropped','naturalUntexturedCropped','naturalTexturedCropped2','naturalUntexturedCropped2']}
colours = list(mcolors.TABLEAU_COLORS.keys())
lineTypes = ['solid', 'dashed', 'dotted']
for occGroup in occTypeClusters:

    # line plot
    xpos = np.arange(len(coveragesTest))
    plt.figure(figsize=lineFigSize)


    for ote, occTypeTest in enumerate(occTypeClusters[occGroup]):

        for otr, occTypeTrain in enumerate(occTypesTrain + [occTypeTest]):

            performance = np.zeros((len(models2use), 10))
            for m, model in enumerate(models2use):

                performance[m,:] = np.array(accuracyMats[occTypeTest][model][occTypeTrain]['acc1']) * 100

            yvals = np.mean(performance, axis=0)
            ySEs = scipy.stats.sem(performance, axis=0)
            polyfit = np.polyfit(yvals, xpos/10, 3)
            p = np.poly1d(polyfit)
            plt.errorbar(xpos, yvals, yerr=ySEs, fmt='-', label=f'{occTypeTrain}_{occTypeTest}_50({p(50):.2f})_75({p(75):.2f})', color=colours[ote], linestyle=lineTypes[otr], capsize=3)

    plt.xlabel('occlusion level tested on')
    plt.xticks(xpos, coveragesTest)
    plt.tick_params(direction='in')
    plt.ylabel('accuracy (%)')
    plt.ylim((0, 100))
    plt.title(f'{occGroup}, top 1 accuracy')
    plt.legend(title='training occluder/test occluder/occlusion thresholds', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{outDir}/{occGroup}_line.png')
    plt.show()

    # bar plot
    for t, thr in enumerate([50, 75]):
        plt.figure(figsize=[3,4])
        for ote, occTypeTest in enumerate(occTypeClusters[occGroup]):
            for otr, occTypeTrain in enumerate(occTypesTrain + [occTypeTest]):
                estimates = np.zeros((len(models2use)))
                for m, model in enumerate(models2use):
                    vals = np.array(accuracyMats[occTypeTest][model][occTypeTrain]['acc1']) * 100
                    polyfit = np.polyfit(vals, xpos/10, 3)
                    p = np.poly1d(polyfit)
                    estimates[m] = np.array(p(thr))
                meanEst = np.mean(estimates)
                seEst = scipy.stats.sem(estimates)
                plt.bar(ote+(otr/3)-.167, meanEst, yerr = seEst, color='white', width=1/3, edgecolor=colours[ote], linestyle=lineTypes[otr])
        plt.xlabel('test occlusion type')
        plt.xticks(np.arange(len(occTypeClusters[occGroup])), labels=(occTypeClusters[occGroup]), rotation = 25)
        plt.tick_params(direction='in')
        plt.ylabel(f'occlusion threshold (%)')
        plt.ylim((0, 1))
        plt.title(f'{occGroup}, occlusion threshold for {thr}% accuracy')
        plt.tight_layout()
        plt.savefig(f'{outDir}/{occGroup}_thr_{thr}.png')
        plt.show()

# mean of barHorz v barVert
models2use = ['alexnet', 'vgg19', 'cornet_s', 'resnet152']
occTypesTrain = ['unoccluded','naturalTextured2']
occTypeClusters = {'barHorz': ['barHorz02', 'barHorz04','barHorz08', 'barHorz16'],
                   'barVert': ['barVert02', 'barVert04','barVert08', 'barVert16']}

# line plot
xpos = np.arange(len(coveragesTest))
plt.figure(figsize=lineFigSize)

for o, occGroup in enumerate(occTypeClusters):

    for otr, occTypeTrain in enumerate(occTypesTrain):

        performance = np.zeros((len(models2use),(len(occTypeClusters[occGroup])), 10))

        for m, model in enumerate(models2use):

            for ote, occTypeTest in enumerate(occTypeClusters[occGroup]):

                performance[m,ote,:] = np.array(accuracyMats[occTypeTest][model][occTypeTrain]['acc1']) * 100

        meanXoccTypes = np.mean(performance, axis=1)
        yvals = np.mean(meanXoccTypes, axis=0)
        yerrs = scipy.stats.sem(meanXoccTypes, axis=0)
        polyfit = np.polyfit(yvals, xpos/10, 3)
        p = np.poly1d(polyfit)
        plt.errorbar(xpos, yvals, yerr=yerrs, label=f'{occTypeTrain}_{occGroup}_50({p(50):.2f})_75({p(75):.2f})', color=colours[o], linestyle=lineTypes[otr], capsize=3)

    # add a line for trained and tested on same occlusion type
    performance = np.zeros((len(models2use), (len(occTypeClusters[occGroup])), 10))
    for m, model in enumerate(models2use):
        for ot, occType in enumerate(occTypeClusters[occGroup]):

            performance[m,ot,:] = np.array(accuracyMats[occType][model][occType]['acc1']) * 100
    meanXoccTypes = np.mean(performance, axis=1)
    yvals = np.mean(meanXoccTypes, axis=0)
    yerrs = scipy.stats.sem(meanXoccTypes, axis=0)
    polyfit = np.polyfit(yvals, xpos / 10, 3)
    p = np.poly1d(polyfit)
    plt.errorbar(xpos, yvals, yerr=yerrs, label=f'{occGroup}_{occGroup}_50({p(50):.2f})_75({p(75):.2f})', color=colours[o], linestyle='dotted', capsize=3)

plt.xlabel('occlusion level tested on')
plt.xticks(xpos, coveragesTest)
plt.tick_params(direction='in')
plt.ylabel('accuracy (%)')
plt.ylim((0, 100))
plt.title(f'bar, top 1 accuracy')
plt.legend(title='training occluder/test occluder/occlusion thresholds', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
# plt.grid(True)
plt.tight_layout()
plt.savefig(f'{outDir}/bar_line.png')
plt.show()


