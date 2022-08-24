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

nGPUs = 1 # target number of GPUs. Overwritten for known 1 GPU networks, set as -1 for all.
GPUids = 1 # specify GPUs to use if not all are/can be used

overwrite = False
models = ['cornet_s_varRec']
dataset = 'imagenet1000'#, 'places365_standard']
datasetPath = f'/home/dave/Datasets/{dataset}'
occlusionMethods = ['barHorz04', 'barVert12', 'barHorz08'] # relating to different versions
coverage = .5
batchSize = 8
workers = 8
invert = False
occColour = [(0,0,0)]#,(127,127,127),(255,255,255)]
lineFigSize = (7,4)
timeses = ((2,2,4,2),(5,5,10,5))
timesStrings = ['2_2_4_2','5_5_10_5']

for times, timesString in zip(timeses, timesStrings): # for model in models
    model = 'cornet_s_varRec'
    v=2
    version = f'v{v+1}'

    occTypesTrain = ['unoccluded', occlusionMethods[v]]
    occTypesTest = ['unoccluded', occlusionMethods[v]]

    outDir = f'DNN/analysis/results/accfMRIstim/{model}/{timesString}/imagenet1000/{version}'
    os.makedirs(outDir, exist_ok=True)
    accuracyPath = os.path.join(outDir, 'accuracy.pkl')
    if os.path.isfile(accuracyPath) and not overwrite:
        accuracies = pickle.load(open(accuracyPath, 'rb'))
    else:
        accuracies = {'occTypeTrain':[],'occTypeTest':[],'topk':[],'accuracy':[]}

        for otr, occTypeTrain in enumerate(occTypesTrain):

            modelDir = os.path.join('DNN/data', f'{model}_{timesString}', dataset, 'fromPretrained', occTypeTrain)

            if occTypeTrain not in ['unoccluded', 'naturalTextured']:
                modelDir = os.path.join(modelDir, f'50')

            weightFiles = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))
            weightsPath = weightFiles[-1]

            for occTypeTest in occTypesTest:

                # call script
                acc1, acc5 = test(model,
                                  datasetPath,
                                  batchSize,
                                  weightsPath,
                                  workers,
                                  occTypeTest,
                                  coverage,
                                  1,
                                  occColour,
                                  invert,
                                  times=times,
                                  nGPUs=nGPUs,
                                  GPUids=GPUids)

                accuracies['occTypeTrain'].append(occTypeTrain)
                accuracies['occTypeTest'].append(occTypeTest)
                accuracies['topk'].append('acc1')
                accuracies['accuracy'].append(acc1)

                accuracies['occTypeTrain'].append(occTypeTrain)
                accuracies['occTypeTest'].append(occTypeTest)
                accuracies['topk'].append('acc5')
                accuracies['accuracy'].append(acc5)

                print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | testing on: {occTypeTest} | model: {model} | times: {times} | trained on: {occTypeTrain} | acc1: {acc1} | acc5: {acc5}')


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
        plt.axhline(y=100 / 1000, color='k', linestyle='dotted')
        plt.title(f'classification accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, f'accuracies_{acc}.png'))
        plt.show()
        plt.close()

accsStandard = pickle.load(open(f'DNN/analysis/results/accfMRIstim/cornet_s_varRec/2_2_4_2/imagenet1000/{version}/accuracy.pkl', 'rb'))
accsStandard['model'] = ['standard model']*len(accuraciesTable)
accsHighRec = accuracies.copy()
accsHighRec['model'] = ['high-recurrence']*len(accuraciesTable)
allAccuracies = pd.DataFrame(accsStandard)
allAccuracies = allAccuracies.append(pd.DataFrame(accsHighRec))
allAccuracies['accuracy'] *= 100
colours = [['darkgreen','limegreen'],['darkgoldenrod','gold']]
hatches = ['','-']

import matplotlib as mpl
#mpl.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth
mpl.rcParams['hatch.linewidth'] = 6

for acc, ylims in zip(['acc1','acc5'],[(0,70),(0,90)]):
    plt.figure(figsize=(3, 3))
    counter=0
    for teo, (testOcc, testOccLabel) in enumerate(zip(['unoccluded','barHorz08'],['unoccluded','occluded'])):
        for tro, (trainOcc, trainOccLabel) in enumerate(zip(['unoccluded','barHorz08'],['standard training','occlusion training'])):
            for m, model in enumerate(['standard model','high-recurrence']):
                accuracy = allAccuracies['accuracy'][(allAccuracies['topk'] == acc) &
                                                     (allAccuracies['occTypeTest'] == testOcc) &
                                                     (allAccuracies['occTypeTrain'] == trainOcc) &
                                                     (allAccuracies['model'] == model)]
                # plot bar outlines in colour
                plt.bar(counter, accuracy, edgecolor=colours[tro][m], linewidth=2.5)
                # plot fill
                plt.bar(counter, accuracy, hatch=hatches[teo], edgecolor='k', linewidth=0, linestyle=None, color=colours[tro][m])

                counter+=1
        counter += 1
    plt.tick_params(direction='in')
    fig = plt.gcf()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
    plt.ylim(ylims)
    plt.ylabel(f'top {acc[3]} accuracy (%)')
    plt.xlabel('test image condition')
    plt.xticks((1.5,6.5), labels=('unoccluded','occluded'))
    plt.tick_params(axis='x', which='both', bottom=False)
    #plt.axhline(y=100 / 1000, color='k', linestyle='dotted')
    plt.title(f'1000-way Imagenet\nclassification accuracy', size=11)
    plt.tight_layout()
    plt.savefig(f'DNN/analysis/results/accfMRIstim/cornet_s_varRec/accuracies_{acc}.pdf')
    plt.show()
    plt.close()

