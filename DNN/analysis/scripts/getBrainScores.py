import sys
import os
import glob
import functools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import datetime
import numpy as np

sys.path.append(f'{os.path.expanduser("~")}/Dave/masterScripts/DNN')
from getModel import getModel
from loadParams import loadParams

sys.path.append(f'{os.path.expanduser("~")}/Dave/repos/BrainScore/model-tools-master')
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment

sys.path.append(f'{os.path.expanduser("~")}/Dave/repos/BrainScore/brain-score-master')
from brainscore import score_model

overwrite = False
modelDir = 'DNN/data'
modelSearch = 'cornet_s_custom_*'
modelDirs = sorted(glob.glob(f'{modelDir}/{modelSearch}'))
modelArcs = [os.path.basename(i) for i in modelDirs]
occTypeTrain = ['unoccluded','barHorz08']
nGPUs = 1
GPUids = 0

testConfig = {'V1': ['movshon.FreemanZiemba2013public.V1-pls'],
              'V2': ['movshon.FreemanZiemba2013public.V2-pls'],
              'V4': ['dicarlo.MajajHong2015public.V4-pls'],
              'IT': ['dicarlo.MajajHong2015public.IT-pls']}
layers = list(testConfig.keys())

if nGPUs == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPUids}"

modelFiles=glob.glob('/home/dave/.result_caching/**/*custom*', recursive = True)
for file in modelFiles:
    os.remove(file)

outDir = 'DNN/analysis/results/brainScore'
if not os.path.isdir(outDir):
    os.makedirs(outDir)

scoresPath = f'{outDir}/scores.pkl'
scoresPathTxt = os.path.join(outDir, 'scores.txt')

if not os.path.isfile(scoresPath):
    scores = pd.DataFrame({'modelName': [], 'recurrentCycles': [], 'kernelSizes': [], 'training': [], 'layerIdx': [], 'layer': [], 'benchmark': [], 'score': []})
else:
    scores = pickle.load(open(scoresPath, 'rb'))

preprocessing = functools.partial(load_preprocess_images, image_size=224)

for modelDir, modelArc in zip(modelDirs, modelArcs):

    modelName = modelArc[:-21]
    recString, kernelString = modelArc.split('_')[-2:]
    recs = [int(recString[3:].split('-')[i]) for i in range(4)]
    kernels = [int(kernelString[2:].split('-')[i]) for i in range(4)]
    modelLoaded = False

    for occ in occTypeTrain:

        # check if all training is complete by presence of params file
        modelDir = f'DNN/data/{modelArc}/imagenet1000/fromScratch/{occ}'
        if occ != 'unoccluded':
            modelDir = f'{modelDir}/50'
        #paramsPath = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))[-1]
        paramsPath = os.path.join(modelDir, 'params/032.pt')

        if os.path.isfile(paramsPath):

            paramsLoaded = False

            for l, layer in enumerate(layers):

                for benchmark in testConfig[layer]:

                    testPerformed = bool(len(scores.loc[(scores['modelName'] == modelName) &
                                                        (scores['recurrentCycles'] == recString) &
                                                        (scores['kernelSizes'] == kernelString) &
                                                        (scores['training'] == occ) &
                                                        (scores['layer'] == layer) &
                                                        (scores['benchmark'] == benchmark), :]))

                    if not testPerformed:

                        # get model with this architecture
                        if not modelLoaded:
                            config = {'modelName': modelName, 'R': recs, 'K': kernels, 'pretrained': False}
                            model = getModel(**config)
                            modelLoaded = True

                        # load params
                        if not paramsLoaded:
                            model = loadParams(model, paramsPath)
                            paramsLoaded = True

                        # get path through model to layers of interest
                        if modelName.endswith('predify'):
                            layerPrefix = 'backbone.'
                        else:
                            layerPrefix = 'module.'

                        # prep for brainscore
                        identifier = f'{modelArc}_{occ}'
                        activations_model = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
                        modelCommit = ModelCommitment(identifier=identifier, activations_model=activations_model, layers=[f'{layerPrefix}{layer}'])

                        print(f'scoring {modelArc} {occ} on {benchmark} benchmark')
                        result = score_model(model_identifier=identifier, model=modelCommit, benchmark_identifier=benchmark)
                        score = result[0].item()
                        scores = pd.concat([scores, pd.DataFrame({'modelName': [modelName],
                                                                  'recurrentCycles': [recString],
                                                                  'kernelSizes': [kernelString],
                                                                  'training': [occ],
                                                                  'layerIdx': [l],
                                                                  'layer': [layer],
                                                                  'benchmark': [benchmark],
                                                                  'score': [score]})])
                        scores = scores.sort_values(by=list(scores.columns[:5]))
                        pickle.dump(scores, open(scoresPath, 'wb'))

                        # viewable text version of table
                        scoresString = scores.to_string(header=True, index=False)
                        with open(scoresPathTxt, 'w+') as f:
                            f.write(scoresString)
                        f.close()

                        print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | model: {modelArc} | occluder: {occ} | layer: {layer} | benchmark: {benchmark} |\n'
                              f'score: {score} |\n')

# make plots for various comparisons
colours = list(mcolors.TABLEAU_COLORS.keys())
xpos = np.arange(len(testConfig))

# 1: different recurrence levels
scoresRec = scores.loc[(scores['kernelSizes'] == 'RF3-3-3-3') &
                       (scores['modelName'] == 'cornet_s_custom'), :].copy()
plt.figure(figsize=(7, 3))
recurrences = scoresRec['recurrentCycles'].unique()
for r, recurrence in enumerate(recurrences):
    for o, occ in enumerate(occTypeTrain):
        values = scoresRec.loc[(scoresRec['recurrentCycles'] == recurrence) &
                               (scoresRec['training'] == occ), 'score']
        plt.plot(xpos,
                 values,
                 linewidth=1.5,
                 linestyle = ['solid','dashed'][o],
                 color = colours[r],
                 marker='o',
                 markerfacecolor='white',
                 label = f'{recurrence}, trained on {occ}')
fig = plt.gcf()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.fill_between(np.arange(len(layers2use)), 1, 2, color='black', alpha=.2, lw=0)
plt.tick_params(direction='in')
plt.xticks(xpos, labels=layers, size=12)
plt.yticks((0, .25, .5), size=12)
plt.xlabel('model layer', size=12)
plt.ylabel('score', size=12)
plt.ylim((0, .6))
#plt.title('occlusion invariance', size=12)
plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(outDir, f'recurrence_linePlot.pdf'), dpi=300)
plt.show()
plt.close()

# 2: different RF sizes
scoresRF = scores.loc[(scores['recurrentCycles'] == 'Rec1-2-4-2') &
                       (scores['modelName'] == 'cornet_s_custom'), :].copy()
plt.figure(figsize=(7, 3))
kernelSizes = scoresRF['kernelSizes'].unique()
for k, kernelSize in enumerate(kernelSizes):
    for o, occ in enumerate(occTypeTrain):
        values = scoresRF.loc[(scoresRF['kernelSizes'] == kernelSize) &
                               (scoresRF['training'] == occ), 'score']
        if len(values) < 4:
            values = [.1,.1,.1,.1]
        plt.plot(xpos,
                 values,
                 linewidth=1.5,
                 linestyle = ['solid','dashed'][o],
                 color = colours[k],
                 marker='o',
                 markerfacecolor='white',
                 label = f'{kernelSize}, trained on {occ}')
fig = plt.gcf()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.fill_between(np.arange(len(layers2use)), 1, 2, color='black', alpha=.2, lw=0)
plt.tick_params(direction='in')
plt.xticks(xpos, labels=layers, size=12)
plt.yticks((0, .25, .5), size=12)
plt.xlabel('model layer', size=12)
plt.ylabel('score', size=12)
plt.ylim((0, .6))
#plt.title('occlusion invariance', size=12)
plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(outDir, f'kernelSize_linePlot.pdf'), dpi=300)
plt.show()
plt.close()

# 3: standard model v predified model
scoresStanPred = scores.loc[(scores['kernelSizes'] == 'RF3-3-3-3') &
                            (scores['recurrentCycles'] == 'Rec1-2-4-2'), :].copy()
plt.figure(figsize=(8, 3))
modelNames = scoresStanPred['modelName'].unique()
modelLabels = ['standard model', 'predictive coding model']
for m, modelName in enumerate(modelNames):
    for o, occ in enumerate(occTypeTrain):
        values = scoresStanPred.loc[(scoresStanPred['modelName'] == modelName) &
                              (scoresStanPred['training'] == occ), 'score']
        if len(values) < 4:
            values = [.1,.1,.1,.1]
        plt.plot(xpos,
                 values,
                 linewidth=1.5,
                 linestyle = ['solid','dashed'][o],
                 color = colours[m],
                 marker='o',
                 markerfacecolor='white',
                 label = f'{modelLabels[m]}, trained on {occ}')
fig = plt.gcf()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.fill_between(np.arange(len(layers2use)), 1, 2, color='black', alpha=.2, lw=0)
plt.tick_params(direction='in')
plt.xticks(xpos, labels=layers, size=12)
plt.yticks((0, .25, .5), size=12)
plt.xlabel('model layer', size=12)
plt.ylabel('score', size=12)
plt.ylim((0, .6))
#plt.title('occlusion invariance', size=12)
plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(outDir, f'standard_v_predify_linePlot.pdf'), dpi=300)
plt.show()
plt.close()

