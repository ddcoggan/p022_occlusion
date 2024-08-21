# analyses responses from 2_getResponses
import os
import glob
import datetime
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from scipy import stats
from matplotlib.patches import Patch


sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from modelLayerLabels import modelLayerLabels

### CONFIGURATION
models = ['alexnet']
trainsets = ['imagenet16_downsampled'] #, 'places365_standard', 'imagenet16'],
occTypes = ['unoccluded', 'naturalTextured', 'naturalTextured2', 'barHorz08']
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
layers2use = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'] # any layer beginning with any of these strings is analysed
levels = ['category','exemplar']
contrasts = ['within','between']

for model in models:

    # store similarities
    similaritiesStats = {}

    layers = modelLayerLabels[model]
    theseLayers = []
    for l, layer in enumerate(layers):
        for layer2use in layers2use:
            if layer.startswith(layer2use):
                theseLayers.append([l, layer])

    for occTypeTrain in occTypes:

        similarities = {}

        if occTypeTrain not in similaritiesStats.keys():
            similaritiesStats[occTypeTrain] = {}


        for l, layer in theseLayers:

            if layer not in similarities.keys():
                similarities[layer] = {}
            if layer not in similaritiesStats.keys():
                similaritiesStats[occTypeTrain][layer] = {}


            for trainset in trainsets: # WARNING: different trainsets not accounted for in similarities dict


                modelDir = os.path.join('DNN/data', model, trainset, occTypeTrain)
                paramsDir = os.path.join(modelDir, 'params')

                if weights == 'final':
                    paramsFile = sorted(glob.glob(os.path.join(paramsDir, '*.pt')))[-1]

                allResponses = []
                labelDict = {'occlusionType': [], 'category': [], 'exemplar': []}

                for occTypeResponse in occTypes:

                    occTypeDir = os.path.join(modelDir, 'responses', occTypeResponse)

                    imageDir = os.path.join(f'/home/dave/Datasets/{trainset}') # just use original dir for filename
                    imageFiles = []

                    for fileType in ['png', 'tif', 'jpg']:
                        theseImages = sorted(glob.glob(os.path.join(imageDir, f'val/**/*.{fileType}'), recursive=True))
                        imageFiles += theseImages
                    if len(imageFiles) == 0:
                        raise Exception('No png, jpg or tif image files found!')
                    else:
                        print(f'Total of {len(imageFiles)} image files found.')

                    for i, imageFile in enumerate(imageFiles):
                        category = os.path.basename(os.path.dirname(imageFile))
                        imageNameFull = os.path.basename(imageFile)
                        imageFormat = imageNameFull.split('.')[-1]
                        imageName = imageNameFull[:-(len(imageFormat)+1)]
                        responsePath = f'{occTypeDir}/{category}/{imageName}.pkl'
                        print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Collating responses... |'
                              f' Model: {model} | Layer: {layer} | Trained on: {trainset} |'
                              f' OccTypeTrain: {occTypeTrain} | OccTypeResponse: {occTypeResponse} |'
                              f' Image: {imageName} ({i+1}/{len(imageFiles)})')
                        response = pickle.load(open(responsePath, 'rb'))
                        allResponses.append(np.array(torch.Tensor.cpu(response[l].flatten())))

                        labelDict['occlusionType'].append(occTypeResponse)
                        labelDict['category'].append(category)
                        labelDict['exemplar'].append(imageName)

                labelDF = pd.DataFrame(labelDict)
                responseArray = np.array(allResponses)
                RSM = np.corrcoef(responseArray)
                allResponses = None # remove from RAM
                responseArray = None # remove from RAM

                # collate exemplar and categorical similarity across occlusion types
                for row in labelDF.index:
                    occTypeA = labelDF['occlusionType'][row]
                    categoryA = labelDF['category'][row]
                    exemplarA = labelDF['exemplar'][row]

                    if occTypeA not in similarities[layer].keys():
                        similarities[layer][occTypeA] = {}

                    if occTypeA not in similaritiesStats[occTypeTrain][layer].keys():
                        similaritiesStats[occTypeTrain][layer][occTypeA] = {}

                    for col in labelDF.index:
                        occTypeB = labelDF['occlusionType'][col]
                        categoryB = labelDF['category'][col]
                        exemplarB = labelDF['exemplar'][col]

                        if occTypeB not in similarities[layer][occTypeA].keys():
                            similarities[layer][occTypeA][occTypeB] = {'exemplar': {'within': [],
                                                                          'between': []},
                                                                'category': {'within': [],
                                                                             'between': []}}
                        if occTypeB not in similaritiesStats[occTypeTrain][layer][occTypeA].keys():
                            similaritiesStats[occTypeTrain][layer][occTypeA][occTypeB] = {'exemplar': {'within': {},
                                                                                                       'between': {}},
                                                                                          'category': {'within': {},
                                                                                                       'between': {}}}

                        if exemplarA == exemplarB:
                            similarities[layer][occTypeA][occTypeB]['exemplar']['within'].append(RSM[row,col])
                        else:
                            similarities[layer][occTypeA][occTypeB]['exemplar']['between'].append(RSM[row, col])

                        if categoryA == categoryB:
                            similarities[layer][occTypeA][occTypeB]['category']['within'].append(RSM[row,col])
                        else:
                            similarities[layer][occTypeA][occTypeB]['category']['between'].append(RSM[row, col])

                fig, axs = plt.subplots(len(occTypes), len(occTypes))
                fig.set_figwidth(7.5)
                fig.set_figheight(5.5)
                colours = [['#247afd', '#de425b'], ['#002481', '#6d0008']]
                custom_patches = []
                for c, contrast in enumerate(contrasts):
                    for le, level in enumerate(levels):
                        custom_patches.append(Patch(facecolor = colours[le][c],label=contrast))

                for a, occTypeA in enumerate(occTypes):
                    for b, occTypeB in enumerate(occTypes):

                        if b <= a:
                            means = [[], []]
                            sems = [[], []]
                            for level in levels:
                                for c, contrast in enumerate(contrasts):
                                    thisMean = np.mean(similarities[layer][occTypeA][occTypeB][level][contrast])
                                    thisSem = stats.sem(similarities[layer][occTypeA][occTypeB][level][contrast])
                                    means[c].append(thisMean)
                                    sems[c].append(thisSem)
                                    similaritiesStats[occTypeTrain][layer][occTypeA][occTypeB][level][contrast]['mean'] = thisMean
                                    similaritiesStats[occTypeTrain][layer][occTypeA][occTypeB][level][contrast]['sem'] = thisSem
                            x_pos = np.arange(2)
                            width = .33
                            horzOffsets = [-.167, .167]
                            for c, contrast in enumerate(contrasts):
                                axs[a,b].bar(x_pos + horzOffsets[c], means[c], width, yerr=sems[c], align='center',
                                       color=colours[c], ecolor='black', capsize=4, label=contrast)
                        else:
                            axs[a,b].axis('off')

                        if b == 0:
                            axs[a,b].set(ylabel = 'correlation (r)')
                            axs[a, b].text(-1.6,0.5,occTypeA,rotation='vertical',size='large',va='center')
                        if a == 0:
                            axs[a, b].text(.5,1.2,occTypeB,size='large',ha='center')
                        axs[a, b].yaxis.grid(True)
                        axs[a, b].set(xticks=x_pos,
                                      xticklabels = ['category','exemplar'],
                                      ylim=[0,1.05])

                plt.legend(custom_patches, ['within','between','within','between'], bbox_to_anchor = (1.05,1), frameon=False)
                plt.suptitle(f'trained on {occTypeTrain}', y=1)
                plt.tight_layout()
                outDir = f'DNN/analysis/results/{model}/responses/{layer}/{trainset}/{occTypeTrain}'
                os.makedirs(outDir, exist_ok=True)
                plt.savefig(f'{outDir}/responseSimilarity.png')
                plt.show()
                plt.close()

        # plot W v B exemplar across layers
        x_pos = np.arange(len(theseLayers))
        colours = ['darkorange', 'green', 'purple']
        c = 0
        plt.figure(figsize=(8, 6))
        for a, occTypeA in enumerate(occTypes):
            for b, occTypeB in enumerate(occTypes):
                if b < a:
                    withins = np.empty(len(theseLayers))
                    betweens = np.empty(len(theseLayers))
                    layerList = []
                    for l in range(len(theseLayers)):
                        layer = theseLayers[l][1]
                        layerList.append(layer)
                        withins[l] = similaritiesStats[occTypeTrain][layer][occTypeA][occTypeB]['exemplar']['within']['mean']
                        betweens[l] = similaritiesStats[occTypeTrain][layer][occTypeA][occTypeB]['exemplar']['between']['mean']
                    means = (withins - betweens) / (withins + betweens)
                    plt.plot(x_pos, means, color=colours[c], label=f'{occTypeA} v {occTypeB}')
                    c += 1
        #plt.legend(bbox_to_anchor=(1.05, 1), frameon=False)
        plt.xticks(x_pos, layerList)
        plt.ylim([0, .6])
        plt.ylabel('exemplar information (r)')
        plt.title(f'within-between/within+between exemplar (trained on {occTypeTrain})', y=1)
        plt.tight_layout()
        outDir = f'DNN/analysis/results/{model}/acrossLayers/{trainset}/{occTypeTrain}'
        os.makedirs(outDir, exist_ok=True)
        plt.savefig(f'{outDir}/exemplarInformation.png')
        plt.show()
        plt.close()

        # plot W v B category across layers
        x_pos = np.arange(len(theseLayers))
        colours = ['gold', 'darkorange', 'red', 'green', 'purple', 'blue']
        c = 0
        plt.figure(figsize=(8, 6))
        for a, occTypeA in enumerate(occTypes):
            for b, occTypeB in enumerate(occTypes):
                if b <= a:
                    withins = np.empty(len(theseLayers))
                    betweens = np.empty(len(theseLayers))
                    layerList = []
                    for l in range(len(theseLayers)):
                        layer = theseLayers[l][1]
                        layerList.append(layer)
                        withins[l] = similaritiesStats[occTypeTrain][layer][occTypeA][occTypeB]['category']['within']['mean']
                        betweens[l] = similaritiesStats[occTypeTrain][layer][occTypeA][occTypeB]['category']['between']['mean']
                    means = (withins - betweens) / (withins + betweens)
                    plt.plot(x_pos, means, color=colours[c], label=f'{occTypeA} v {occTypeB}')
                    c += 1
        #plt.legend(bbox_to_anchor=(1.05, 1), frameon=False)
        plt.xticks(x_pos, layerList)
        plt.ylim([0, .6])
        plt.ylabel('category information (r)')
        plt.title(f'within-between/within+between category (trained on {occTypeTrain})', y=1)
        plt.tight_layout()
        outDir = f'DNN/analysis/results/{model}/acrossLayers/{trainset}/{occTypeTrain}'
        os.makedirs(outDir, exist_ok=True)
        plt.savefig(f'{outDir}/categoricalInformation.png')
        plt.show()
        plt.close()
