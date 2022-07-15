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
from sklearn.linear_model import LinearRegression

sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from saveOutputs import saveOutputs
from modelLayerLabels import modelLayerLabels
from centreCropResize import centreCropResize

### CONFIGURATION
models = ['alexnet']
datasets = ['imagenet16'] #, 'places365_standard', 'imagenet16'],
occluders = ['unoccluded','naturalTextured2','barHorz08']
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
layers2use = 'relu'
exemplarNames =['bear','bison','elephant','hare','jeep','lamp','sportsCar','teapot']
nExemplars = 8
nOccPos = 28
version = 'slidingBars'
occludersNoLevels = ['unoccluded','naturalTextured','naturalTextured2']


for model in models:

    layers = modelLayerLabels[model]
    theseLayers = []
    for l, layer in enumerate(layers):
        if layer.startswith(layers2use):
            theseLayers.append([l, layer])

    for l, layer in theseLayers:

        for dataset in datasets:

            tableDict = {'occTypeTrain': [], 'contrast': [], 'level': [], 'mean': [], 'sem': []}
            outDir = f'DNN/analysis/results/{model}/{dataset}/RSAfMRIstim/slidingBars/RSMs'
            os.makedirs(outDir, exist_ok=True)

            RSMs = {}
            for occluder in occluders:
                if occluder in occludersNoLevels:
                    coverageString = None
                    modelDir = os.path.join('DNN/data', model, dataset, 'fromPretrained', occluder)
                else:
                    coverage = .5
                    modelDir = os.path.join('DNN/data', model, dataset, 'fromPretrained', occluder, str(int(coverage * 100)))

                responseDir = os.path.join(modelDir, 'responses/fMRIstim', version)
                RSMs[occluder] = np.empty(shape=[nExemplars,nOccPos,nOccPos])

                for e, exemplarName in enumerate(exemplarNames):
                    allResponses = []
                    for o in range(nOccPos):

                        responsePath = f'{responseDir}/{exemplarName}/{o:02}.pkl'
                        print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Collating responses... |'
                              f' Model: {model} | Layer: {layer} | Trained on: {dataset} |'
                              f' OccTypeTrain: {occluder} | cond: {exemplarName} |')
                        response = pickle.load(open(responsePath, 'rb'))
                        allResponses.append(np.array(torch.Tensor.cpu(response[l].flatten())))

                    responseArray = np.array(allResponses)
                    matrix = np.corrcoef(responseArray)
                    RSMs[occluder][e,:,:] = matrix
                RSMs[occluder] = np.mean(RSMs[occluder], axis=0)

                # matrix plot
                fig, ax = plt.subplots()
                im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='rainbow')
                ax.set_xticks(np.arange(nOccPos))
                ax.set_yticks(np.arange(nOccPos))
                ax.tick_params(direction='in')
                ax.set_xlabel('occluder Position')
                ax.set_ylabel('occluder Position')
                fig.colorbar(im, fraction=0.0453)
                ax.set_title(f'occTypeTrain: {occluder}, layer: {layer}')
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                plt.text(28, 15, 'correlation (r)', rotation='vertical', fontsize=12)
                fig.tight_layout()
                plt.savefig(f'{outDir}/{occluder}_{layer}.png')
                plt.show()
                plt.close()


