import os
import glob
import datetime
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
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
occluders = ['unoccluded', 'naturalTextured2', 'barHorz04']#['unoccluded', 'horzBars', 'polkadot', 'crossBars', 'dropout']
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
layers2use = ['relu1','relu2','relu3','relu4','relu5']
version = 'v1'
if version in ['v1','v3']:
    occPositions = ['none','upper','lower']
else:
    occPositions = ['none','left','right']
exemplarNames =['bear','bison','elephant','hare','jeep','lamp','sportsCar','teapot']
levels = [exemplarNames, occPositions]
condNames = conds = list(itertools.product(*levels))
nConds = len(conds)
condNames = []
for cond in conds:
	condNames.append(f'{cond[0]}_{cond[1]}')
nExemplars = 8
nConds = 24
occludersNoLevels = ['unoccluded','naturalTextured','naturalTextured2']


## models

RSAmodels = {'identity': np.eye(nConds),
             'exemplar': np.ones((nConds, nConds)) * 0,
             'occluderCond': np.ones((nConds, nConds)) * 0,
             'occluderPresence': np.ones((nConds, nConds)) * 0}
RSAmodelNames = list(RSAmodels.keys())

for e in range(nExemplars):
    RSAmodels['exemplar'][(e * 3):(e * 3 + 3), (e * 3):(e * 3 + 3)] = 1
for c in range(nConds):

    RSAmodels['occluderCond'][c % 3::3, c] = 1

    if c % 3 == 0:
        RSAmodels['occluderPresence'][c, np.arange(0, nConds, 3)] = 1
    else:
        RSAmodels['occluderPresence'][c, np.arange(1, nConds, 3)] = 1
        RSAmodels['occluderPresence'][c, np.arange(2, nConds, 3)] = 1

    # remove same image conds
    RSAmodels['exemplar'][c, c] = None
    RSAmodels['occluderCond'][c, c] = None
    RSAmodels['occluderPresence'][c, c] = None



RSMs = {}
for downSampleFactor in [1, 2, 4, 8]:
    RSMs[downSampleFactor] = {}
    for model in models:

        layers = modelLayerLabels[model]
        theseLayers = []
        for l, layer in enumerate(layers):
            for layer2use in layers2use:
                if layer.startswith(layer2use):
                    theseLayers.append([l, layer])

        for l, layer in theseLayers:
            RSMs[downSampleFactor][layer]={}
            for dataset in datasets:

                tableDict = {'occluder': [], 'contrast': [], 'level': [], 'mean': [], 'sem': []}
                outDir = f'DNN/analysis/results/{model}/{dataset}/RSAfMRIstim_downSample/{version}/{downSampleFactor}/RSMs'
                os.makedirs(outDir, exist_ok=True)


                for occluder in occluders:
                    if occluder in occludersNoLevels:
                        coverageString = None
                        modelDir = os.path.join('DNN/data', model, dataset, 'fromPretrained', occluder)
                    else:
                        coverage = .5
                        modelDir = os.path.join('DNN/data', model, dataset, 'fromPretrained', occluder, str(int(coverage * 100)))

                    allResponses = []
                    responseDir = os.path.join(modelDir, f'responses/fMRIstim/{version}')

                    for c, condName in enumerate(condNames):
                        responsePath = f'{responseDir}/{condName}.pkl'
                        print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Collating responses... |'
                              f' Model: {model} | Layer: {layer} | Trained on: {dataset} |'
                              f' OccTypeTrain: {occluder} | cond: {condName} |')
                        response = pickle.load(open(responsePath, 'rb'))[l]
                        if downSampleFactor > 1:
                            response = torch.nn.AvgPool2d(downSampleFactor, downSampleFactor, 0)(response)
                        allResponses.append(np.array(torch.Tensor.cpu(response.flatten())))

                    responses = np.array(allResponses)

                    '''# normalise patterns (separately for each occlusion condition)
                    responsesNormed = np.empty_like(responses)
                    for oc in range(3):
                        meanInd = np.tile(np.mean(responses[oc::3, :], axis=0), (8, 1))
                        responsesNormed[oc::3, :] = responses[oc::3, :] - meanInd
                    '''

                    # normalise patterns (all conds together)
                    responsesNormed = responses - np.tile(np.mean(responses, axis=0), (nConds, 1))

                    matrix = np.corrcoef(responsesNormed)
                    RSMs[downSampleFactor][layer][occluder] = matrix

                    # matrix plot
                    fig, ax = plt.subplots()
                    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='rainbow')
                    ax.set_xticks(np.arange(nConds))
                    ax.set_yticks(np.arange(nConds))
                    ax.set_xticklabels(condNames)
                    ax.set_yticklabels(condNames)
                    ax.tick_params(direction='in')
                    ax.set_xlabel('image')
                    ax.set_ylabel('image')
                    fig.colorbar(im, fraction=0.0453)
                    ax.set_title(f'occluders: {occluder}, layer: {layer}')
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    plt.text(24, 15, 'correlation (r)', rotation='vertical', fontsize=12)
                    fig.tight_layout()
                    plt.savefig(f'{outDir}/{occluder}_{layer}.png')
                    plt.show()
                    plt.close()

                    # 2x2 exemplar by occluder position (occluded images only)
                    sameExemSameOcc = []  # occluded images only, same exemplar, same occluder
                    sameExemDiffOcc = []  # occluded images only, same exemplar, different occluder
                    diffExemSameOcc = []  # occluded images only, different exemplar, same occluder
                    diffExemDiffOcc = []  # occluded images only, different exemplar, different occluder

                    # 2x2, exemplar by one occluded yes/no
                    sameExemBothUnocc = []  # unoccluded images only, same exemplar
                    diffExemBothUnocc = []  # unoccluded images only, different exemplar
                    sameExemOneUnocc = []  # occluded v unoccluded, same exemplar
                    diffExemOneUnocc = []  # occluded v unoccluded, different exemplar

                    for cA, condNameA in enumerate(condNames):

                        exemplarA = condNameA.split('_')[0]
                        occluderA = condNameA.split('_')[1]

                        for cB, condNameB in enumerate(condNames):

                            exemplarB = condNameB.split('_')[0]
                            occluderB = condNameB.split('_')[1]

                            r = matrix[cA, cB]

                            # 2x2 exemplar by occluder position (occluded images only)
                            if occluderA != 'none' and occluderB != 'none':
                                if exemplarA == exemplarB and occluderA == occluderB:
                                    sameExemSameOcc.append(r)
                                elif exemplarA != exemplarB and occluderA == occluderB:
                                    diffExemSameOcc.append(r)
                                elif exemplarA == exemplarB and occluderA != occluderB:
                                    sameExemDiffOcc.append(r)
                                elif exemplarA != exemplarB and occluderA != occluderB:
                                    diffExemDiffOcc.append(r)

                            # 2x2, exemplar by one occluded yes/no
                            elif occluderA == 'none' and occluderB == 'none':  # neither occluded
                                if exemplarA == exemplarB:
                                    sameExemBothUnocc.append(r)
                                else:
                                    diffExemBothUnocc.append(r)
                            elif int(occluderA == 'none') + int(occluderB == 'none') == 1:  # one occluded
                                if exemplarA == exemplarB:
                                    sameExemOneUnocc.append(r)
                                else:
                                    diffExemOneUnocc.append(r)

                    # 2x2 exemplar by occluder position (occluded images only)
                    ## same exemplar, same occluder
                    tableDict['occluder'].append(occluder)
                    tableDict['contrast'].append('occPositionExemplar')
                    tableDict['level'].append('sameExemSameOcc')
                    tableDict['mean'].append(np.mean(sameExemSameOcc))
                    tableDict['sem'].append(stats.sem(sameExemSameOcc))

                    ## same exemplar, different occluder
                    tableDict['occluder'].append(occluder)
                    tableDict['contrast'].append('occPositionExemplar')
                    tableDict['level'].append('sameExemDiffOcc')
                    tableDict['mean'].append(np.mean(sameExemDiffOcc))
                    tableDict['sem'].append(stats.sem(sameExemDiffOcc))

                    ## different exemplar, same occluder
                    tableDict['occluder'].append(occluder)
                    tableDict['contrast'].append('occPositionExemplar')
                    tableDict['level'].append('diffExemSameOcc')
                    tableDict['mean'].append(np.mean(diffExemSameOcc))
                    tableDict['sem'].append(stats.sem(diffExemSameOcc))

                    ## different exemplar, different occluder
                    tableDict['occluder'].append(occluder)
                    tableDict['contrast'].append('occPositionExemplar')
                    tableDict['level'].append('diffExemDiffOcc')
                    tableDict['mean'].append(np.mean(diffExemDiffOcc))
                    tableDict['sem'].append(stats.sem(diffExemDiffOcc))

                    # 2x2, exemplar by one occluded yes/no
                    ## same exemplar, both unoccluded
                    tableDict['occluder'].append(occluder)
                    tableDict['contrast'].append('occVunocc')
                    tableDict['level'].append('sameExemBothUnocc')
                    tableDict['mean'].append(np.mean(sameExemBothUnocc))
                    tableDict['sem'].append(stats.sem(sameExemBothUnocc))

                    ## same exemplar, one occluded
                    tableDict['occluder'].append(occluder)
                    tableDict['contrast'].append('occVunocc')
                    tableDict['level'].append('sameExemOneUnocc')
                    tableDict['mean'].append(np.mean(sameExemOneUnocc))
                    tableDict['sem'].append(stats.sem(sameExemOneUnocc))

                    ## different exemplar, one unoccluded
                    tableDict['occluder'].append(occluder)
                    tableDict['contrast'].append('occVunocc')
                    tableDict['level'].append('diffExemBothUnocc')
                    tableDict['mean'].append(np.mean(diffExemBothUnocc))
                    tableDict['sem'].append(stats.sem(diffExemBothUnocc))

                    ## different exemplar, one unoccluded
                    tableDict['occluder'].append(occluder)
                    tableDict['contrast'].append('occVunocc')
                    tableDict['level'].append('diffExemOneUnocc')
                    tableDict['mean'].append(np.mean(diffExemOneUnocc))
                    tableDict['sem'].append(stats.sem(diffExemOneUnocc))

                table = pd.DataFrame(tableDict)

                # make plots

                # 2x2 exemplar by occluder position (occluded images only)
                df = table.loc[table['contrast'] == 'occPositionExemplar', :].copy()
                df['level'] = df['level'].astype('category')
                df['level'] = df['level'].cat.reorder_categories(['sameExemSameOcc', 'sameExemDiffOcc', 'diffExemSameOcc', 'diffExemDiffOcc'])
                df['occluder'] = df['occluder'].astype('category').cat.reorder_categories(occluders)
                dfMeans = df.pivot(index='occluder', columns='level', values='mean')
                dfSEMs = df.pivot(index='occluder', columns='level', values='sem').values
                dfPlot = dfMeans.plot(kind='bar', yerr=dfSEMs.transpose(), ylabel='correlation (r)', rot=0, figsize=(6, 4))
                fig = dfPlot.get_figure()
                plt.tick_params(direction='in')
                plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
                plt.ylim((-.6, 1.1))
                plt.title(f'layer: {layer}')
                plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                plt.tight_layout()
                outDir = f'DNN/analysis/results/{model}/{dataset}/RSAfMRIstim_downSample/{version}/{downSampleFactor}/occluded'
                os.makedirs(outDir, exist_ok=True)
                fig.savefig(os.path.join(outDir, f'{layer}.png'))
                plt.show()
                plt.close()

                # 2x2, exemplar by one occluded yes/no
                df = table.loc[table['contrast'] == 'occVunocc', :].copy()
                df['level'] = df['level'].astype('category')
                df['level'] = df['level'].cat.reorder_categories(['sameExemBothUnocc', 'sameExemOneUnocc', 'diffExemBothUnocc', 'diffExemOneUnocc'])
                df['occluder'] = df['occluder'].astype('category').cat.reorder_categories(occluders)
                dfMeans = df.pivot(index='occluder', columns='level', values='mean')
                dfSEMs = df.pivot(index='occluder', columns='level', values='sem').values
                dfPlot = dfMeans.plot(kind='bar', yerr=dfSEMs.transpose(), ylabel='correlation (r)', rot=0, figsize=(6, 4))
                fig = dfPlot.get_figure()
                plt.tick_params(direction='in')
                plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
                plt.ylim((-.5, 1.1))
                plt.title(f'layer: {layer}')
                plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                plt.tight_layout()
                outDir = f'DNN/analysis/results/{model}/{dataset}/RSAfMRIstim_downSample/{version}/{downSampleFactor}/occVunocc'
                os.makedirs(outDir, exist_ok=True)
                fig.savefig(os.path.join(outDir, f'{layer}.png'))
                plt.show()
                plt.close()

                # regression
                regrTableDict = {'occluder': [], 'model': [], 'fit': []}
                for occluder in occluders:
                    for RSAmodel in RSAmodels.keys():
                        modelFlat = RSAmodels[RSAmodel].flatten()
                        modelFlatNoDiag = modelFlat[np.isfinite(modelFlat)]  # remove where model flat == 0

                        matrixFlat = RSMs[downSampleFactor][layer][occluder].flatten()
                        matrixFlatNoDiag = matrixFlat[np.isfinite(modelFlat)]  # remove where model flat == 0

                        regr = LinearRegression()
                        fit = regr.fit(modelFlatNoDiag.reshape(-1, 1), matrixFlatNoDiag.reshape(-1, 1)).coef_[0][0]
                        regrTableDict['occluder'].append(occluder)
                        regrTableDict['model'].append(RSAmodel)
                        regrTableDict['fit'].append(fit)

                dfr = pd.DataFrame(regrTableDict)
                dfr.loc[:, dfr.columns == 'fit'] = dfr.loc[:, dfr.columns == 'fit'].astype('float')
                dfr.loc[:, dfr.columns == 'model'] = dfr.loc[:, dfr.columns == 'model'].astype('category')
                dfr['model'] = dfr['model'].cat.reorder_categories(RSAmodelNames)
                dfr['occluder'] = dfr['occluder'].astype('category').cat.reorder_categories(occluders)
                dfrPivot = dfr.pivot(index='occluder', columns='model', values='fit')
                dfrPivot.plot(kind='bar', ylabel='coefficient (beta)', rot=0, figsize=(6, 4))
                plt.tick_params(direction='in')
                plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
                plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                plt.ylim([-0.1, 1.1])
                plt.title(f'layer: {layer}')
                plt.tight_layout()
                outDir = f'DNN/analysis/results/{model}/{dataset}/RSAfMRIstim_downSample/{version}/{downSampleFactor}/regression'
                os.makedirs(outDir, exist_ok=True)
                plt.savefig(os.path.join(outDir, f'{layer}.png'))
                plt.show()
                plt.close()

RSMpath = f'DNN/analysis/results/{model}/{dataset}/RSAfMRIstim_downSample/{version}/RSMs.pkl'
pickle.dump(RSMs, open(RSMpath, 'wb'))