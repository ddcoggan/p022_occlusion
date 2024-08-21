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
import time
#time.sleep(3600)

### CONFIGURATION
models = ['cornet_s_custom']#['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'alexnet', 'cornet_s', 'inception_v3', 'vgg19', 'PredNetImageNet']
datasets = ['imagenet1000'] #, 'places365_standard', 'imagenet16'],
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
versions = ['v3']
coverage = '50' # 'mixedLevels'
layers2use = ['V1','V2','V4','IT']
exemplarNames = ['bear','bison','elephant','hare','jeep','lamp','sportsCar','teapot']

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




for version in versions:
    if version in ['v1','v3']:
        occPositions = ['none','upper','lower']
    else:
        occPositions = ['none','left','right']

    occluders = ['unoccluded']#, 'naturalTextured']  # ['unoccluded', 'horzBars', 'polkadot', 'crossBars', 'dropout']
    if version == 'v1':
        occluders.append('barHorz04')
    elif version == 'v2':
        occluders.append('barVert12')
    elif version == 'v3':
        occluders.append('barHorz08')
    levels = [exemplarNames, occPositions]
    condNames = conds = list(itertools.product(*levels))
    nConds = len(conds)
    condNames = []
    for cond in conds:
        condNames.append(f'{cond[0]}_{cond[1]}')

    for norm in ['allConds']:#, 'occluder']:
        RSMs = {}

        allIdxPlot = np.empty(shape=(2,2,2,4))
        for mod, model in enumerate(models):

            templateResponsePath = os.path.join('DNN/data', model, f'{datasets[0]}/fromPretrained/unoccluded/responses/fMRIstim/{version}/bear_none.pkl')
            layers = list(pickle.load(open(templateResponsePath, 'rb')).keys())

            tableAll = pd.DataFrame({'layer': [], 'dataset': [], 'occluder': [], 'contrast': [], 'level': [], 'mean': [], 'sem': []})
            for l, layer in enumerate(layers):
                RSMs[layer]={}
                for dataset in datasets:

                    tableDict = {'occluder': [], 'contrast': [], 'level': [], 'mean': [], 'sem': []}
                    analysisDir = f'DNN/analysis/results/RSAfMRIstim/{version}/{model}/{dataset}/{norm}'
                    os.makedirs(analysisDir, exist_ok=True)

                    for occluder in occluders:

                        modelDir = os.path.join('DNN/data', model, dataset, 'fromPretrained', occluder)
                        if occluder not in occludersNoLevels:
                            modelDir = os.path.join(modelDir, coverage)

                        allResponses = []
                        responseDir = os.path.join(modelDir, f'responses/fMRIstim/{version}')

                        for c, condName in enumerate(condNames):
                            responsePath = f'{responseDir}/{condName}.pkl'
                            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Collating responses... |'
                                  f' Model: {model} | Layer: {layer} | Trained on: {dataset} |'
                                  f' OccTypeTrain: {occluder} | cond: {condName} |')
                            response = pickle.load(open(responsePath, 'rb'))
                            allResponses.append(np.array(torch.Tensor.cpu(response[layer].flatten())))

                        responses = np.array(allResponses)
                        responsesNormed = np.empty_like(responses)

                        if norm == 'allConds':
                            responsesNormed = responses - np.tile(np.mean(responses, axis=0), (nConds, 1))
                        elif norm == 'occluder':
                            for oc in range(3):
                                meanResp = np.tile(np.mean(responses[oc::3, :], axis=0), (8, 1))
                                responsesNormed[oc::3, :] = responses[oc::3, :] - meanResp

                        matrix = np.corrcoef(responsesNormed)
                        RSMs[layer][occluder] = matrix

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
                        outDir = f'{analysisDir}/RSMs'
                        os.makedirs(outDir, exist_ok=True)
                        plt.savefig(f'{outDir}/{occluder}_{l}_{layer}.pdf')
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

                    # append to larger table of all results
                    start = len(tableAll)
                    tableAll = pd.concat([tableAll,table])
                    stop = len(tableAll)
                    tableAll['layer'][start:stop] = layer
                    tableAll['dataset'][start:stop] = dataset

                    # make plots

                    # 2x2 exemplar by occluder position (occluded images only)
                    df = table.loc[table['contrast'] == 'occPositionExemplar', :].copy()
                    df['level'] = df['level'].astype('category')
                    df['level'] = df['level'].cat.reorder_categories(['sameExemSameOcc', 'sameExemDiffOcc', 'diffExemSameOcc', 'diffExemDiffOcc'])
                    df['occluder'] = df['occluder'].astype('category').cat.reorder_categories(occluders)
                    dfMeans = df.pivot(index='occluder', columns='level', values='mean')
                    dfSEMs = df.pivot(index='occluder', columns='level', values='sem').values
                    dfPlot = dfMeans.plot(kind='bar', yerr=dfSEMs.transpose(), ylabel='correlation (r)', rot=0, figsize=(5, 4))
                    fig = dfPlot.get_figure()
                    plt.tick_params(direction='in')
                    #plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
                    plt.xlabel('occlusion training')
                    plt.ylim((-.6, 1.1))
                    plt.title(f'layer: {layer}')
                    plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                    plt.tight_layout()
                    outDir = f'{analysisDir}/occluded'
                    os.makedirs(outDir, exist_ok=True)
                    fig.savefig(os.path.join(outDir, f'{l}_{layer}.pdf'))
                    plt.show()
                    plt.close()

                    # 2x2, exemplar by one occluded yes/no
                    df = table.loc[table['contrast'] == 'occVunocc', :].copy()
                    df['level'] = df['level'].astype('category')
                    df['level'] = df['level'].cat.reorder_categories(['sameExemBothUnocc', 'sameExemOneUnocc', 'diffExemBothUnocc', 'diffExemOneUnocc'])
                    df['occluder'] = df['occluder'].astype('category').cat.reorder_categories(occluders)
                    dfMeans = df.pivot(index='occluder', columns='level', values='mean')
                    dfSEMs = df.pivot(index='occluder', columns='level', values='sem').values
                    dfPlot = dfMeans.plot(kind='bar', yerr=dfSEMs.transpose(), ylabel='correlation (r)', rot=0, figsize=(6, 4), color=['tab:purple','tab:pink','tab:olive','tab:cyan'])
                    fig = dfPlot.get_figure()
                    plt.tick_params(direction='in')
                    plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
                    plt.ylim((-.5, 1.1))
                    plt.title(f'layer: {layer}')
                    plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                    plt.tight_layout()
                    outDir = f'{analysisDir}/occVunocc'
                    os.makedirs(outDir, exist_ok=True)
                    fig.savefig(os.path.join(outDir, f'{l}_{layer}.pdf'))
                    plt.show()
                    plt.close()

                    # regression
                    regrTableDict = {'occluder': [], 'model': [], 'fit': []}
                    for occluder in occluders:
                        for RSAmodel in RSAmodels.keys():
                            modelFlat = RSAmodels[RSAmodel].flatten()
                            modelFlatNoDiag = modelFlat[np.isfinite(modelFlat)]  # remove where model flat == 0

                            matrixFlat = RSMs[layer][occluder].flatten()
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
                    outDir = f'{analysisDir}/regression'
                    os.makedirs(outDir, exist_ok=True)
                    plt.savefig(os.path.join(outDir, f'{l}_{layer}.pdf'))
                    plt.show()
                    plt.close()

                    RSMpath = f'{analysisDir}/RSMs.pkl'
                    pickle.dump(RSMs, open(RSMpath, 'wb'))

            # plot results across layers:
            for dataset in datasets:
                analysisDir = f'DNN/analysis/results/RSAfMRIstim/{version}/{model}/{dataset}/{norm}'
                for occ, occluder in enumerate(occluders):

                    # 2x2 exemplar by occluder position (occluded images only)
                    df = tableAll.loc[(tableAll['dataset'] == dataset) &
                                      (tableAll['occluder'] == occluder) &
                                      (tableAll['contrast'] == 'occPositionExemplar') &
                                      (tableAll['layer'].isin(layers2use)), :].copy()
                    df['level'] = df['level'].astype('category').cat.reorder_categories(['sameExemSameOcc', 'sameExemDiffOcc', 'diffExemSameOcc', 'diffExemDiffOcc'])
                    df['layer'] = df['layer'].astype('category').cat.reorder_categories(layers2use)
                    dfMeans = df.pivot(index='layer', columns='level', values='mean')
                    dfSEMs = df.pivot(index='layer', columns='level', values='sem').values
                    dfPlot = dfMeans.plot(kind='bar', yerr=dfSEMs.transpose(), ylabel='correlation (r)', rot=0, figsize=(3.5, 2.25), width=.8, legend=False)
                    fig = plt.gcf()
                    ax = plt.gca()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tick_params(direction='in')
                    plt.xticks(size= 12)
                    plt.yticks(size=12)
                    plt.xlabel('model layer', size = 12)
                    plt.ylabel('correlation (r)', size=12)
                    plt.ylim((-.6, 1.1))
                    '''
                    if occluder == 'unoccluded':
                        plt.title(f'standard training', size=16)
                    else:
                        plt.title(f'occlusion trained', size=16)
                    '''
                    #plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=10)
                    plt.tight_layout()
                    outDir = f'{analysisDir}/occluded'
                    os.makedirs(outDir, exist_ok=True)
                    fig.savefig(os.path.join(outDir, f'allLayers_{occluder}.pdf'))
                    plt.show()
                    plt.close()

                    # object completion index
                    means, sems = [],[]
                    for layer in layers2use:

                        sameExemSameOcc = df['mean'][(df['layer'] == layer) &
                                                     (df['level'] == 'sameExemSameOcc')].item()
                        sameExemDiffOcc = df['mean'][(df['layer'] == layer) &
                                                     (df['level'] == 'sameExemDiffOcc')].item()
                        diffExemSameOcc = df['mean'][(df['layer'] == layer) &
                                                     (df['level'] == 'diffExemSameOcc')].item()
                        diffExemDiffOcc = df['mean'][(df['layer'] == layer) &
                                                     (df['level'] == 'diffExemDiffOcc')].item()

                        occEffect = sameExemSameOcc - sameExemDiffOcc
                        exemEffect = (sameExemSameOcc + sameExemDiffOcc) - (diffExemSameOcc + diffExemDiffOcc)
                        #objCompIdx = (exemEffect - occEffect) / (exemEffect + occEffect) # full completion = 1, no completion = 0
                        #objCompIdx = (sameExemDiffOcc - diffExemSameOcc) / (np.abs(sameExemDiffOcc) + np.abs(diffExemSameOcc))
                        #objCompIdx = (sameExemDiffOcc - diffExemDiffOcc) / (sameExemSameOcc - diffExemDiffOcc)
                        objCompIdx = sameExemDiffOcc / sameExemSameOcc
                        means.append(objCompIdx)

                    objCompDF = pd.DataFrame({'layer': layers2use, 'mean': means})
                    objCompPlot = objCompDF.plot(kind='line', linewidth=1.5, rot=0, legend=False, figsize=(2.5, 2.25), marker='o', markerfacecolor='white')
                    fig = plt.gcf()
                    ax = plt.gca()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.fill_between(np.arange(len(layers2use)), 1, 2, color='black', alpha=.2, lw=0)
                    plt.tick_params(direction='in')
                    plt.xticks(np.arange(len(layers2use)), labels=layers2use, size=12)
                    plt.yticks((-.5,0,.5,1), size=12)
                    plt.xlabel('model layer', size = 12)
                    plt.ylabel('OCI',size=12)
                    plt.ylim((-.5, 1.))
                    #plt.title('object completion', size=12)
                    #plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=10)
                    plt.tight_layout()
                    outDir = f'{analysisDir}/occluded'
                    os.makedirs(outDir, exist_ok=True)
                    fig.savefig(os.path.join(outDir, f'allLayers_{occluder}_objCompIdx.pdf'), dpi = 300)
                    plt.show()
                    plt.close()

                    allIdxPlot[0, mod, occ, :] = means

                    # 2x2, exemplar by one occluded yes/no
                    df = tableAll.loc[(tableAll['dataset'] == dataset) &
                                      (tableAll['occluder'] == occluder) &
                                      (tableAll['contrast'] == 'occVunocc') &
                                      (tableAll['layer'].isin(layers2use)), :].copy()
                    df['level'] = df['level'].astype('category').cat.reorder_categories(['sameExemBothUnocc', 'sameExemOneUnocc', 'diffExemBothUnocc', 'diffExemOneUnocc'])
                    df['layer'] = df['layer'].astype('category').cat.reorder_categories(layers2use)
                    dfMeans = df.pivot(index='layer', columns='level', values='mean')
                    dfSEMs = df.pivot(index='layer', columns='level', values='sem').values
                    dfPlot = dfMeans.plot(kind='bar', yerr=dfSEMs.transpose(), ylabel='correlation (r)', rot=0, figsize=(3.5, 2.25), width=.8, legend=False, color=['tab:purple','tab:pink','tab:olive','tab:cyan'])
                    fig = plt.gcf()
                    ax = plt.gca()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tick_params(direction='in')
                    plt.xticks(size=12)
                    plt.yticks(size=12)
                    plt.xlabel('model layer', size=12)
                    plt.ylabel('correlation (r)', size=12)
                    plt.ylim((-.6, 1.1))
                    '''
                    if occluder == 'unoccluded':
                        plt.title(f'standard training', size=16)
                    else:
                        plt.title(f'occlusion trained', size=16)
                    '''
                    #plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=10)
                    plt.tight_layout()
                    outDir = f'{analysisDir}/occVunocc'
                    os.makedirs(outDir, exist_ok=True)
                    fig.savefig(os.path.join(outDir, f'allLayers_{occluder}.pdf'))
                    plt.show()
                    plt.close()

                    # occlusion invariance index
                    means, sems = [], []
                    for layer in layers2use:
                        sameExemOneUnocc = df['mean'][(df['layer'] == layer) &
                                                     (df['level'] == 'sameExemOneUnocc')].item()
                        sameExemBothUnocc = df['mean'][(df['layer'] == layer) &
                                                     (df['level'] == 'sameExemBothUnocc')].item()
                        diffExemOneUnocc = df['mean'][(df['layer'] == layer) &
                                                     (df['level'] == 'diffExemOneUnocc')].item()
                        diffExemBothUnocc = df['mean'][(df['layer'] == layer) &
                                                     (df['level'] == 'diffExemBothUnocc')].item()

                        occEffect = sameExemSameOcc - sameExemDiffOcc
                        exemEffect = (sameExemSameOcc + sameExemDiffOcc) - (diffExemSameOcc + diffExemDiffOcc)
                        # objCompIdx = (exemEffect - occEffect) / (exemEffect + occEffect) # full completion = 1, no completion = 0
                        # objCompIdx = (sameExemDiffOcc - diffExemSameOcc) / (np.abs(sameExemDiffOcc) + np.abs(diffExemSameOcc))
                        #occInvIdx = (sameExemOneUnocc - diffExemOneUnocc) / (sameExemBothUnocc + diffExemBothUnocc)
                        occInvIdx = sameExemOneUnocc / sameExemBothUnocc
                        means.append(occInvIdx)

                    objCompDF = pd.DataFrame({'layer': layers2use, 'mean': means})
                    objCompPlot = objCompDF.plot(kind='line', linewidth=1.5, rot=0, legend=False, figsize=(2.5, 2.25), color = 'tab:purple', marker='o', markerfacecolor='white')
                    fig = plt.gcf()
                    ax = plt.gca()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.fill_between(np.arange(len(layers2use)), 1, 2, color='black', alpha=.2, lw=0)
                    plt.tick_params(direction='in')
                    plt.xticks(np.arange(len(layers2use)), labels=layers2use, size=12)
                    plt.yticks((-.5,0, .5, 1), size=12)
                    plt.xlabel('model layer', size=12)
                    plt.ylabel('OII', size=12)
                    plt.ylim((-.5, 1.))
                    #plt.title('occlusion invariance', size=12)
                    # plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=10)
                    plt.tight_layout()
                    outDir = f'{analysisDir}/occVunocc'
                    os.makedirs(outDir, exist_ok=True)
                    fig.savefig(os.path.join(outDir, f'allLayers_{occluder}_occInvIdx.pdf'), dpi=300)
                    plt.show()
                    plt.close()

                    allIdxPlot[1, mod, occ, :] = means

        colours = [['darkgreen', 'limegreen'], ['darkgoldenrod', 'gold']]
        for i, (idx, id) in enumerate(zip(['object completion', 'occlusion invariance'], ['OCI','OII'])):
            ylims = [(-.5, .7),(-.25,.8)][i]
            yticks = [(-.4, -.2, 0, .2, .4, .6),(-.2,0, .2,.4,.6)][i]
            plt.figure(figsize=(5, 3))
            for t, trainingName in enumerate(['standard training', 'occlusion training']):
                for mod, modelName in enumerate(['standard model', 'high-recurrence']):
                    plt.plot(np.arange(len(layers2use)), allIdxPlot[i,mod,t,:], marker='o', markerfacecolor='white', color=colours[t][mod], label=f'{trainingName}\n{modelName}')
            fig = plt.gcf()
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tick_params(direction='in')
            plt.xticks(np.arange(len(layers2use)), labels=layers2use, size=12)
            plt.yticks(yticks, size=12)
            plt.xlabel('model layer', size=12)
            plt.ylabel(id, size=12)
            plt.ylim(ylims)
            plt.title(idx, size=16)
            plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=12)
            plt.tight_layout()
            outDir = f'DNN/analysis/results/RSAfMRIstim/{version}/'
            os.makedirs(outDir, exist_ok=True)
            fig.savefig(os.path.join(outDir, f'cornet_s_varRec_{id}.pdf'), dpi=300)
            plt.show()
            plt.close()


# make plot of effective RF size

imSize = 224
cycles = [[2,2,4,2],[5,5,10,5]]
RFsizes = np.empty(shape=(2,4))
for recLevel in range(2):
    RFsize = 1
    for l, layer in enumerate(cycles[recLevel]):
        for cycle in range(layer):
            print()
            RFsize += 4
        RFsizes[recLevel, l] = min(1, RFsize/imSize)

plt.plot(np.arange(len(layers2use)), RFsizes[0], label = 'standard CORnet-S')
plt.plot(np.arange(len(layers2use)), RFsizes[1], label = 'high-recurrence CORnet-S')
fig = plt.gcf()
fig.set_size_inches(6, 2.25)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(direction='in')
plt.xticks(np.arange(len(layers2use)), labels=layers2use, size=12)
plt.yticks(np.arange(0,.6,.1), size=12)
plt.xlabel('model layer', size=12)
plt.ylabel('RF size / image size', size=12)
plt.ylim((0, .5))
plt.title('effective RF size', size=12)
plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=10)
plt.tight_layout()
outDir = f'/home/dave/Desktop'
fig.savefig(os.path.join(outDir, f'RFsize_cornet_s_varRec.pdf'), dpi=300)
plt.show()
plt.close()


cycles = [[2,2,4,2],[5,5,10,5]]
totalConvs = np.empty(shape=(2,4))
for recLevel in range(2):
    convs = 0
    for l, layer in enumerate(cycles[recLevel]):
        convs += layer
        totalConvs[recLevel, l] = convs


plt.plot(np.arange(len(layers2use)), totalConvs[0], label = 'standard CORnet-S')
plt.plot(np.arange(len(layers2use)), totalConvs[1], label = 'high-recurrence CORnet-S')
fig = plt.gcf()
fig.set_size_inches(6, 2.25)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(direction='in')
plt.xticks(np.arange(len(layers2use)), labels=layers2use, size=12)
plt.yticks(np.arange(0,26,5), size=12)
plt.xlabel('model layer', size=12)
plt.ylabel('cycles', size=12)
plt.ylim((0, 25))
plt.title('cumulative recurrent cycles', size=12)
plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=10)
plt.tight_layout()
outDir = f'/home/dave/Desktop'
fig.savefig(os.path.join(outDir, f'Cycles_cornet_s_varRec.pdf'), dpi=300)
plt.show()
plt.close()


