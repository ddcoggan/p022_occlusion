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
import time
#time.sleep(18000)
sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from saveOutputs import saveOutputs
from modelLayerLabels import modelLayerLabels
from centreCropResize import centreCropResize

### CONFIGURATION
models = ['PredNetImageNet']
trainsets = ['imagenet16'] #, 'places365_standard', 'imagenet16'],
occTypesTrain = ['unoccluded','naturalTextured2','barHorz08']
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
layers2use = ''
exemplarNames =['bear','bison','elephant','hare','jeep','lamp','sportsCar','teapot']
condNames = ['bear_none',
 'bear_upper',
 'bear_lower',
 'bison_none',
 'bison_upper',
 'bison_lower',
 'elephant_none',
 'elephant_upper',
 'elephant_lower',
 'hare_none',
 'hare_upper',
 'hare_lower',
 'jeep_none',
 'jeep_upper',
 'jeep_lower',
 'lamp_none',
 'lamp_upper',
 'lamp_lower',
 'sportsCar_none',
 'sportsCar_upper',
 'sportsCar_lower',
 'teapot_none',
 'teapot_upper',
 'teapot_lower']
nExemplars = 8
nConds = 24
version = 'v3'

RSAmodels = {'image': np.eye(nConds),
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

model = 'PredNetImageNet'
trainset = 'imagenet16'
templateResponsePath = f'DNN/data/{model}/{trainset}/fromPretrained/unoccluded/responses/fMRIstim/{version}/bear_none.pkl'
templateResponse = pickle.load(open(templateResponsePath, 'rb'))

RSMs = {}

for passType in templateResponse:
    if passType != 'input':

        passTypeName = passType[2:-4]
        
        RSMs[passTypeName] = {}
        
        for cycle in range(len(templateResponse[passType])):
            
            RSMs[passTypeName][cycle] = {}
            
            for layer in range(len(templateResponse[passType][cycle])):
                
                RSMs[passTypeName][cycle][layer] = {}

                tableDict = {'occTypeTrain': [], 'contrast': [], 'level': [], 'mean': [], 'sem': []}
                outDir = f'DNN/analysis/results/{model}/{trainset}/RSAfMRIstim/{version}/RSMs'
                os.makedirs(outDir, exist_ok=True)

                for occTypeTrain in occTypesTrain:
                    if occTypeTrain == 'unoccluded':
                        modelDir = os.path.join(os.getcwd(), f'DNN/data/{model}/{trainset}/fromPretrained', occTypeTrain)
                    elif occTypeTrain in ['horzBar8','naturalTypes']:
                        modelDir = os.path.join(os.getcwd(), f'DNN/data/{model}/{trainset}/fromPretrained', occTypeTrain, 'mixedCoverages')
                
                    allResponses = []
                    responseDir = os.path.join(modelDir, 'responses/fMRIstim', version)
                
                
                    for c, condName in enumerate(condNames):
                
                        responsePath = f'{responseDir}/{condName}.pkl'
                        print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Collating responses... |'
                              f' Model: {model} | Cycle: {cycle}: Layer: {layer} | Trained on: {trainset} |'
                              f' OccTypeTrain: {occTypeTrain} | cond: {condName} |')
                        response = pickle.load(open(responsePath, 'rb'))
                        allResponses.append(torch.Tensor.cpu(response[passType][cycle][layer].flatten()).detach().numpy())
                
                    responses = np.array(allResponses)
                    # normalise patterns (all conds together)
                    responsesNormed = responses - np.tile(np.mean(responses, axis=0), (nConds, 1))
                    matrix = np.corrcoef(responsesNormed)
                    RSMs[passTypeName][cycle][layer][occTypeTrain] = matrix
                
                    # matrix plot
                    fig, ax = plt.subplots(figsize=(8,6))
                    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='rainbow')
                    ax.set_xticks(np.arange(nConds))
                    ax.set_yticks(np.arange(nConds))
                    ax.set_xticklabels(condNames)
                    ax.set_yticklabels(condNames)
                    ax.tick_params(direction='in')
                    ax.set_xlabel('image')
                    ax.set_ylabel('image')
                    fig.colorbar(im, fraction=0.0453)
                    ax.set_title(f'occTypeTrain: {occTypeTrain}\npassType: {passTypeName}\ncycle: {cycle}, layer: {layer}')
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    plt.text(24, 15, 'correlation (r)', rotation='vertical', fontsize=12)
                    fig.tight_layout()
                    plt.savefig(f'{outDir}/{occTypeTrain}_{passTypeName}_{cycle}_{layer}.png')
                    plt.show()
                    plt.close()

                    # 2x2 exemplar by occluder position (occluded images only)
                    sameExemSameOcc = []  # occluded images only, same exemplar, same occluder
                    sameExemDiffOcc = []  # occluded images only, same exemplar, different occluder
                    diffExemSameOcc = []  # occluded images only, different exemplar, same occluder
                    diffExemDiffOcc = []  # occluded images only, different exemplar, different occluder

                    # 2x2, exemplar by one occluded yes/no
                    sameExemBothUnocc = [] # unoccluded images only, same exemplar
                    diffExemBothUnocc = [] # unoccluded images only, different exemplar
                    sameExemOneUnocc = []  # occluded v unoccluded, same exemplar
                    diffExemOneUnocc = []  # occluded v unoccluded, different exemplar
                
                    for cA, condNameA in enumerate(condNames):
                
                        exemplarA = condNameA.split('_')[0]
                        occluderA = condNameA.split('_')[1]
                
                        for cB, condNameB in enumerate(condNames):
                
                            exemplarB = condNameB.split('_')[0]
                            occluderB = condNameB.split('_')[1]
                
                            r = matrix[cA,cB]


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
                    tableDict['occTypeTrain'].append(occTypeTrain)
                    tableDict['contrast'].append('occPositionExemplar')
                    tableDict['level'].append('sameExemSameOcc')
                    tableDict['mean'].append(np.mean(sameExemSameOcc))
                    tableDict['sem'].append(stats.sem(sameExemSameOcc))

                    ## same exemplar, different occluder
                    tableDict['occTypeTrain'].append(occTypeTrain)
                    tableDict['contrast'].append('occPositionExemplar')
                    tableDict['level'].append('sameExemDiffOcc')
                    tableDict['mean'].append(np.mean(sameExemDiffOcc))
                    tableDict['sem'].append(stats.sem(sameExemDiffOcc))

                    ## different exemplar, same occluder
                    tableDict['occTypeTrain'].append(occTypeTrain)
                    tableDict['contrast'].append('occPositionExemplar')
                    tableDict['level'].append('diffExemSameOcc')
                    tableDict['mean'].append(np.mean(diffExemSameOcc))
                    tableDict['sem'].append(stats.sem(diffExemSameOcc))

                    ## different exemplar, different occluder
                    tableDict['occTypeTrain'].append(occTypeTrain)
                    tableDict['contrast'].append('occPositionExemplar')
                    tableDict['level'].append('diffExemDiffOcc')
                    tableDict['mean'].append(np.mean(diffExemDiffOcc))
                    tableDict['sem'].append(stats.sem(diffExemDiffOcc))

                    # 2x2, exemplar by one occluded yes/no
                    ## same exemplar, both unoccluded
                    tableDict['occTypeTrain'].append(occTypeTrain)
                    tableDict['contrast'].append('occVunocc')
                    tableDict['level'].append('sameExemBothUnocc')
                    tableDict['mean'].append(np.mean(sameExemBothUnocc))
                    tableDict['sem'].append(stats.sem(sameExemBothUnocc))
                
                    ## same exemplar, one occluded
                    tableDict['occTypeTrain'].append(occTypeTrain)
                    tableDict['contrast'].append('occVunocc')
                    tableDict['level'].append('sameExemOneUnocc')
                    tableDict['mean'].append(np.mean(sameExemOneUnocc))
                    tableDict['sem'].append(stats.sem(sameExemOneUnocc))

                    ## different exemplar, one unoccluded
                    tableDict['occTypeTrain'].append(occTypeTrain)
                    tableDict['contrast'].append('occVunocc')
                    tableDict['level'].append('diffExemBothUnocc')
                    tableDict['mean'].append(np.mean(diffExemBothUnocc))
                    tableDict['sem'].append(stats.sem(diffExemBothUnocc))

                    ## different exemplar, one unoccluded
                    tableDict['occTypeTrain'].append(occTypeTrain)
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
                df['occTypeTrain'] = df['occTypeTrain'].astype('category').cat.reorder_categories(occTypesTrain)
                dfMeans = df.pivot(index='occTypeTrain', columns='level', values='mean')
                dfSEMs = df.pivot(index='occTypeTrain', columns='level', values='sem').values
                dfPlot = dfMeans.plot(kind='bar', yerr=dfSEMs.transpose(), ylabel='correlation (r)', rot=0, figsize=(6, 4))
                fig = dfPlot.get_figure()
                plt.tick_params(direction='in')
                plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
                plt.ylim((-.6, 1.1))
                plt.title(f'passType: {passTypeName}\ncycle: {cycle}, layer: {layer}')
                plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                plt.tight_layout()
                outDir = f'DNN/analysis/results/{model}/{trainset}/RSAfMRIstim/{version}/occluded'
                os.makedirs(outDir, exist_ok=True)
                fig.savefig(os.path.join(outDir, f'{passTypeName}_{cycle}_{layer}.png'))
                plt.show()
                plt.close()

                # 2x2, exemplar by one occluded yes/no
                df = table.loc[table['contrast'] == 'occVunocc', :].copy()
                df['level'] = df['level'].astype('category')
                df['level'] = df['level'].cat.reorder_categories(['sameExemBothUnocc', 'sameExemOneUnocc', 'diffExemBothUnocc', 'diffExemOneUnocc'])
                df['occTypeTrain'] = df['occTypeTrain'].astype('category').cat.reorder_categories(occTypesTrain)
                dfMeans = df.pivot(index='occTypeTrain', columns='level', values='mean')
                dfSEMs = df.pivot(index='occTypeTrain', columns='level', values='sem').values
                dfPlot = dfMeans.plot(kind='bar', yerr=dfSEMs.transpose(), ylabel='correlation (r)', rot=0, figsize=(6, 4))
                fig = dfPlot.get_figure()
                plt.tick_params(direction='in')
                plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
                plt.ylim((-.6, 1.1))
                plt.title(f'passType: {passTypeName}\ncycle: {cycle}, layer: {layer}')
                plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                plt.tight_layout()
                outDir = f'DNN/analysis/results/{model}/{trainset}/RSAfMRIstim/{version}/occVunocc'
                os.makedirs(outDir, exist_ok=True)
                fig.savefig(os.path.join(outDir, f'{passTypeName}_{cycle}_{layer}.png'))
                plt.show()
                plt.close()
                
                # regression
                regrTableDict = {'occTypeTrain': [], 'model': [], 'fit': []}
                for occTypeTrain in occTypesTrain:
                    for RSAmodel in RSAmodels.keys():
                
                        modelFlat = RSAmodels[RSAmodel].flatten()
                        modelFlatNoDiag = modelFlat[np.isfinite(modelFlat)]  # remove where model flat == 0
                
                        matrixFlat = RSMs[passTypeName][cycle][layer][occTypeTrain].flatten()
                        matrixFlatNoDiag = matrixFlat[np.isfinite(modelFlat)]  # remove where model flat == 0
                
                        regr = LinearRegression()
                        fit = regr.fit(modelFlatNoDiag.reshape(-1, 1), matrixFlatNoDiag.reshape(-1, 1)).coef_[0][0]
                        regrTableDict['occTypeTrain'].append(occTypeTrain)
                        regrTableDict['model'].append(RSAmodel)
                        regrTableDict['fit'].append(fit)
                
                dfr = pd.DataFrame(regrTableDict)
                dfr.loc[:, dfr.columns == 'fit'] = dfr.loc[:, dfr.columns == 'fit'].astype('float')
                dfr.loc[:, dfr.columns == 'model'] = dfr.loc[:, dfr.columns == 'model'].astype('category')
                dfr['model'] = dfr['model'].cat.reorder_categories(RSAmodelNames)
                dfr['occTypeTrain'] = dfr['occTypeTrain'].astype('category').cat.reorder_categories(occTypesTrain)
                dfrPivot = dfr.pivot(index='occTypeTrain', columns='model', values='fit')
                dfrPivot.plot(kind='bar', ylabel='coefficient (beta)', rot=0, figsize=(6, 4))
                plt.tick_params(direction='in')
                plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
                plt.legend(title='', bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                plt.ylim([-0.6, 1.1])
                plt.title(f'passType: {passTypeName}\ncycle: {cycle}, layer: {layer}')
                plt.tight_layout()
                outDir = f'DNN/analysis/results/{model}/{trainset}/RSAfMRIstim/{version}/regression'
                os.makedirs(outDir, exist_ok=True)
                plt.savefig(os.path.join(outDir, f'{passTypeName}_{cycle}_{layer}.png'))
                plt.show()
                plt.close()

RSMpath = f'DNN/analysis/results/{model}/{trainset}/RSAfMRIstim/{version}/RSMs.pkl'
pickle.dump(RSMs, open(RSMpath, 'wb'))