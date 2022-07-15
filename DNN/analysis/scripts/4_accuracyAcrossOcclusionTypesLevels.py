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
#time.sleep(36000)

sys.path.append('/mnt/HDD12TB/masterScripts')
from DNN.analysis.scripts.test import test

overwrite = True
models = ['vgg19']#['alexnet']#, 'CORnet_S']
datasets = ['imagenet16']#, 'places365_standard']
occlusionMethods = [os.path.basename(x) for x in sorted(glob.glob('DNN/images/occluders/*'))]
occlusionMethods.append('unoccluded')
occlusionMethodsNoLevels = ['unoccluded','naturalTextured','naturalTextured2']
occludersBehavioural = occTypesTest = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 'polkadot','polkasquare','crossBarOblique','crossBarCardinal', 'naturalUntexturedCropped2']

for o in occlusionMethodsNoLevels:
    if o != 'unoccluded'
        occlusionMethods.remove(o)

coveragesTrain = [[.1,.2,.4,.8]]
coveragesTest = [0.,.1,.2,.3,.4,.5,.6,.7,.8,.9]
batchSize = 128
workers = 8
invert = False
colours = [(0,0,0),(127,127,127),(255,255,255)]
lineFigSize = (5,4)
matFigSizeType = (9,9)
matFigSizeLevel = (5,3)

# accuracy across occlusion methods / within coverage
for model in models:

    for dataset in datasets:
        datasetPath = f'/home/dave/Datasets/{dataset}'
        outDir = f'DNN/analysis/results/{model}/{dataset}/accuracy/withinOcclusionLevels'
        os.makedirs(outDir, exist_ok=True)
        accuracyPath = os.path.join(outDir, 'accuracy.pkl')
        if os.path.isfile(accuracyPath) and not overwrite:
            accuracyMats = pickle.load(open(accuracyPath, 'rb'))
        else:
            accuracyMats = {}
            for coverage in coveragesTrain:

                if type(coverage) is list:
                    coverageString = 'mixedLevels'
                else:
                    coverageString = f'{int(coverage * 100)}'

                accuracyMats[coverageString] = {'acc1': np.empty(shape=(len(occlusionMethods), len(occlusionMethods))),
                                                'acc5': np.empty(shape=(len(occlusionMethods), len(occlusionMethods)))}

                for otr, occTypeTrain in enumerate(occlusionMethods):

                    if occTypeTrain == 'unoccluded':
                        modelDir = os.path.join(os.getcwd(), 'DNN/data', model, dataset, 'fromPretrained/unoccluded')
                    else:
                        modelDir = os.path.join(os.getcwd(), 'DNN/data', model, dataset, 'fromPretrained', occTypeTrain, coverageString)

                    weightFiles = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))
                    weightsPath = weightFiles[-1]

                    for ote, occTypeTest in enumerate(occlusionMethods):

                        # call script
                        acc1, acc5 = test(model, datasetPath, batchSize, weightsPath, workers, occTypeTest, coverage, 1, colours, invert)
                        accuracyMats[coverageString]['acc1'][otr, ote] = acc1
                        accuracyMats[coverageString]['acc5'][otr, ote] = acc5

                        print(f'{model} | coverage: {coverage} | trained on {occlusionMethods[otr]} | tested on {occlusionMethods[ote]} | acc1 {acc1} | acc5 {acc5}')

            # save accuracyMats
            pickle.dump(accuracyMats, open(accuracyPath, 'wb'))

        for coverage in coveragesTrain:

            if type(coverage) is list:
                coverageString = 'mixedLevels'
            else:
                coverageString = f'{int(coverage * 100)}'

            # plots
            for acc in accuracyMats[coverageString].keys():

                # matrix plot
                fig, ax = plt.subplots(figsize = matFigSizeType)
                im = ax.imshow(accuracyMats[coverageString][acc]*100, vmin = 0, vmax = 100, cmap='rainbow')
                ax.set_xticks(np.arange(len(occlusionMethods)))
                ax.set_yticks(np.arange(len(occlusionMethods)))
                ax.set_xticklabels(occlusionMethods)
                ax.set_yticklabels(occlusionMethods)
                ax.tick_params(direction = 'in')
                ax.set_xlabel('occlusion type tested on')
                ax.set_ylabel('occlusion type trained on')
                fig.colorbar(im, fraction = 0.0453)
                for i in range(len(occlusionMethods)):
                    for j in range(len(occlusionMethods)):
                        text = ax.text(j, i, f'{int(accuracyMats[coverageString][acc][i, j]*100)}',
                                       ha="center", va="center", color="k")
                ax.set_title(f'level: {coverageString}, top {acc[3]} accuracy')
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")
                plt.tight_layout()
                plt.savefig(f'{outDir}/level{coverageString}_{acc}_mat.png')
                plt.show()

                '''
                # line plot
                plotColours = list(mcolors.TABLEAU_COLORS.keys())
                plotColours.append('black')
                xpos = np.arange(len(occlusionMethods))
                plt.figure(figsize=lineFigSize)
                for otr, occTypeTrain in enumerate(occlusionMethods):
                    plt.plot(xpos, accuracyMats[coverage][acc][otr,:]*100, label=occTypeTrain, color=plotColours[otr])
                plt.xlabel('occlusion level tested on')
                plt.xticks(xpos, occlusionMethods, rotation = 45, ha = 'right')
                plt.tick_params(direction='in')
                plt.ylabel('accuracy (%)')
                plt.ylim((0,100))
                plt.title(f'{coverage}, top {acc[3]} accuracy')
                plt.legend(title='occlusion level trained on', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                #plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{outDir}/cov{int(coverage * 100)}_{acc}_line.png')
                plt.show()
                '''


'''
# accuracy across coverage / within occlusion Type
for model in models:

    for dataset in datasets:
        datasetPath = f'/home/dave/Datasets/{dataset}'
        outDir = f'DNN/analysis/results/{model}/{dataset}/accuracy/withinOcclusionTypes'
        os.makedirs(outDir, exist_ok=True)
        accuracyPath = os.path.join(outDir, 'accuracy.pkl')
        if os.path.isfile(accuracyPath) and not overwrite:
            accuracyMats = pickle.load(open(accuracyPath, 'rb'))
        else:
            accuracyMats = {}
            for occlusionMethod in occlusionMethods:
                if occlusionMethod != 'unoccluded':
                    accuracyMats[occlusionMethod] = {'acc1': np.empty(shape=(len(coveragesTrain), len(coveragesTest))),
                                    'acc5': np.empty(shape=(len(coveragesTrain), len(coveragesTest)))}

                    for ctr, coverageTrain in enumerate(coveragesTrain):
                        if coverageTrain > 0:
                            modelDir = os.path.join(os.getcwd(), 'DNN/data', model, dataset, 'fromPretrained', occlusionMethod, f'{int(coverageTrain * 100)}')
                        else:
                            modelDir = os.path.join(os.getcwd(), 'DNN/data', model, dataset, 'fromPretrained/unoccluded')

                        weightFiles = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))
                        weightsPath = weightFiles[-1]

                        for cte, coverageTest in enumerate(coveragesTest):

                            # call script
                            acc1, acc5 = test(model, datasetPath, batchSize, weightsPath, workers, occlusionMethod, coverageTest, colours, invert)
                            accuracyMats[occlusionMethod]['acc1'][ctr, cte] = acc1
                            accuracyMats[occlusionMethod]['acc5'][ctr, cte] = acc5

                            print(f'{model} | occType: {occlusionMethod} | trained on {coveragesTrain[ctr]} | tested on {coveragesTest[cte]} | acc1 {acc1} | acc5 {acc5}')
                else:
                    modelDir = os.path.join(os.getcwd(), 'DNN/data', model, dataset, 'fromPretrained', occlusionMethod)
                    weightFiles = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))
                    weightsPath = weightFiles[-1]
                    acc1, acc5 = test(model, datasetPath, batchSize, weightsPath, workers, occlusionMethod, coverageTest, colours, invert)
                    accuracyMats[occlusionMethod] = {'acc1': acc1, 'acc5': acc5}

            # save accuracyMats
            pickle.dump(accuracyMats, open(accuracyPath, 'wb'))

        # make plots
        for occlusionMethod in occlusionMethods[:-1]:

            # plots
            for acc in accuracyMats[occlusionMethod]:

                # matrix plot
                fig, ax = plt.subplots(figsize = matFigSizeLevel)
                im = ax.imshow(accuracyMats[occlusionMethod][acc]*100, vmin = 0, vmax = 100, cmap='rainbow')
                ax.set_xticks(np.arange(len(coveragesTest)))
                ax.set_yticks(np.arange(len(coveragesTrain)))
                ax.set_xticklabels(coveragesTest)
                ax.set_yticklabels(coveragesTrain)
                ax.tick_params(direction='in')
                ax.set_xlabel('occlusion level tested on')
                ax.set_ylabel('occlusion level trained on')
                fig.colorbar(im, fraction = .0231)
                for i in range(len(coveragesTrain)):
                    for j in range(len(coveragesTest)):
                        text = ax.text(j, i, f'{int(accuracyMats[occlusionMethod][acc][i, j]*100)}',
                                       ha="center", va="center", color="k")
                ax.set_title(f'{occlusionMethod}, top {acc[3]} accuracy')
                plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
                fig.tight_layout()
                plt.savefig(f'{outDir}/{occlusionMethod}_{acc}_mat.png')
                plt.show()

                # line plot
                plotColours = list(mcolors.TABLEAU_COLORS.keys())
                xpos = np.arange(len(coveragesTest))
                plt.figure(figsize=lineFigSize)
                for ctr, coverageTrain in enumerate(coveragesTrain):
                    yvals = accuracyMats[occlusionMethod][acc][ctr, :]*100
                    plt.plot(xpos, yvals, label=coverageTrain, color=plotColours[ctr], marker = '.')
                plt.xlabel('occlusion level tested on')
                plt.xticks(xpos, coveragesTest)
                plt.tick_params(direction='in')
                plt.ylabel('accuracy (%)')
                plt.ylim((0, 100))
                plt.title(f'{occlusionMethod}, top {acc[3]} accuracy')
                plt.legend(title='occlusion level\ntrained on', bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)
                #plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{outDir}/{occlusionMethod}_{acc}_line.png')
                plt.show()
'''



