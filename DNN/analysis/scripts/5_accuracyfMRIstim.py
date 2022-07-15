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
import pandas as pd

sys.path.append('/mnt/HDD12TB/masterScripts')
from DNN.test import test

overwrite = False
models = ['cornet_s']
dataset = 'imagenet16'#, 'places365_standard']
datasetPath = f'/home/dave/Datasets/{dataset}'
occlusionMethods = ['barHorz04', 'barVert12', 'barHorz08'] # relating to different versions
coverage = .5
batchSize = 16
workers = 8
invert = False
occColour = [(0,0,0)]#,(127,127,127),(255,255,255)]
lineFigSize = (7,4)

for model in models:
    v=2
    version = f'v{v+1}'

    occTypesTrain = ['unoccluded', occlusionMethods[v]]
    occTypesTest = ['unoccluded', occlusionMethods[v]]

    outDir = f'DNN/analysis/results/accfMRIstim/{model}/imagenet16/{version}'
    os.makedirs(outDir, exist_ok=True)
    accuracyPath = os.path.join(outDir, 'accuracy.pkl')
    if os.path.isfile(accuracyPath) and not overwrite:
        accuracies = pickle.load(open(accuracyPath, 'rb'))
    else:
        accuracies = {'occTypeTrain':[],'occTypeTest':[],'topk':[],'accuracy':[]}

        for otr, occTypeTrain in enumerate(occTypesTrain):

            modelDir = os.path.join('DNN/data', model, dataset, 'fromPretrained', occTypeTrain)

            if occTypeTrain not in ['unoccluded', 'naturalTextured']:
                modelDir = os.path.join(modelDir, f'mixedLevels')

            weightFiles = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))
            weightsPath = weightFiles[-1]

            for occTypeTest in occTypesTest:

                # call script
                acc1, acc5 = test(model, datasetPath, batchSize, weightsPath, workers, occTypeTest, coverage, 1, occColour, invert)

                accuracies['occTypeTrain'].append(occTypeTrain)
                accuracies['occTypeTest'].append(occTypeTest)
                accuracies['topk'].append('acc1')
                accuracies['accuracy'].append(acc1)

                accuracies['occTypeTrain'].append(occTypeTrain)
                accuracies['occTypeTest'].append(occTypeTest)
                accuracies['topk'].append('acc5')
                accuracies['accuracy'].append(acc5)

                print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | testing on: {occTypeTest} | model: {model} | trained on: {occTypeTrain} | acc1: {acc1} | acc5: {acc5}')


        pickle.dump(accuracies, open(accuracyPath, 'wb'))


    # plots
    accuraciesTable = pd.DataFrame(accuracies)
    accuraciesTable['accuracy'] *= 100
    for acc in ['acc1','acc5']:
        df = accuraciesTable.loc[accuraciesTable['topk'] == acc, :].copy()
        df['occTypeTrain'] = df['occTypeTrain'].astype('category')
        df['occTypeTrain'] = df['occTypeTrain'].cat.reorder_categories(occTypesTrain)
        df['occTypeTest'] = df['occTypeTest'].astype('category')
        df['occTypeTest'] = df['occTypeTest'].cat.reorder_categories(occTypesTest)
        dfPivot = df.pivot(index='occTypeTrain', columns='occTypeTest', values='accuracy')
        dfPivot.plot(kind='bar', ylabel='accuracy (%)', rot=0, figsize=(4, 4))
        plt.tick_params(direction='in')
        plt.legend(title='test occlusion', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
        plt.ylim([0, 100])
        plt.xlabel('occlusion training')
        plt.axhline(y=100 / 16, color='k', linestyle='dotted')
        plt.title(f'classification accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, f'accuracies_{acc}.png'))
        plt.show()
        plt.close()

