import os
import sys
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
#time.sleep(3600)

sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from getLRPmaps import getLRPmaps
transform = None
version = 'v3'
models = ['alexnet']
datasets = ['imagenet16'] #, 'places365_standard', 'imagenet16'],
occludersBehavioural = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 'polkadot','polkasquare','crossBarOblique','crossBarCardinal', 'naturalUntexturedCropped2']
occludersNoLevels = ['unoccluded','naturalTextured','naturalTextured2']
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
for modelName in models:

    for dataset in datasets:
        datasetDir = f'/home/dave/Datasets/{dataset}'

        occluders = occludersBehavioural + ['unoccluded']
        maps = np.empty((len(occluders),24,224**2))

        outDir = f'DNN/analysis/results/{modelName}/{dataset}/LRPcorrelation'
        os.makedirs(outDir, exist_ok=True)

        for o, occluder in enumerate(occluders):

            modelDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occluder)
            if occluder not in occludersNoLevels:
                modelDir = f'{modelDir}/mixedLevels'

            paramsDir = os.path.join(modelDir, 'params')

            if weights == 'final':
                paramsFile = sorted(glob.glob(os.path.join(paramsDir, '*.pt')))[-1]

            imagesUnocc = sorted(glob.glob(f'fMRI/{version}/images/unoccluded/**/*.JPEG', recursive=True))
            if version in ['v1','v3']:
                imagesOccA = sorted(glob.glob(f'fMRI/{version}/images/occluded/*/top/*.JPEG'))
                imagesOccB = sorted(glob.glob(f'fMRI/{version}/images/occluded/*/bot/*.JPEG'))
            elif version in ['v2']:
                imagesOccA = sorted(glob.glob(f'fMRI/{version}/images/occluded/*/left/*.JPEG'))
                imagesOccB = sorted(glob.glob(f'fMRI/{version}/images/occluded/*/right/*.JPEG'))
            imagePaths = []
            condNames = []
            classIDs16 = [3,6,7,4,9,14,12,15]
            classIDs = []
            for e in range(8):
                imagePaths.append(imagesUnocc[e])
                imagePaths.append(imagesOccA[e])
                imagePaths.append(imagesOccB[e])
                condNames.append(f'{imagesUnocc[e].split("/")[-2]}_none')
                if version in ['v1', 'v3']:
                    condNames.append(f'{imagesUnocc[e].split("/")[-2]}_upper')
                    condNames.append(f'{imagesUnocc[e].split("/")[-2]}_lower')
                elif version in ['v2']:
                    condNames.append(f'{imagesUnocc[e].split("/")[-2]}_left')
                    condNames.append(f'{imagesUnocc[e].split("/")[-2]}_right')
                classIDs += [classIDs16[e],classIDs16[e],classIDs16[e]]


            # loop over images and measure responses
            for i, imageFile in enumerate(imagePaths):
                outPath = None
                if i in [0,1,2]:
                    exampleDir = f'{outDir}/LRPmaps'
                    os.makedirs(exampleDir, exist_ok=True)
                    outPath = f'{exampleDir}/{occluder}_{os.path.basename(imageFile)}'
                classID = classIDs[i]
                print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Performing LRP |'
                      f' Model: {modelName} | Trained on: {dataset} | OccTypeTrain: {occluder} |'
                      f' image: {condNames[i]}')
                LRPmap = getLRPmaps(modelName, dataset, paramsFile, imageFile, outPath=outPath, classID=classID, returnMap=True)
                maps[o,i,:] = LRPmap.flatten()


        for occUnocc, idx in zip(['unoccluded','occluded'],[np.arange(0,24,3),[1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23]]):
            cormat = np.empty((len(occluders), len(occluders)))
            for occA in range(len(occluders)):
                occAresp = maps[occA,:,:]
                for occB in range(len(occluders)):
                    occBresp = maps[occB, :, :]
                    corrs = np.empty(len(idx))
                    for i, id in enumerate(idx):
                        corrs[i] = np.corrcoef(occAresp[id,:], occBresp[id,:])[0,1]
                    cormat[occA,occB] = np.mean(corrs)
            fig, ax = plt.subplots(figsize=(8,8))
            im = ax.imshow(cormat, vmin=-1, vmax=1, cmap='rainbow')
            ax.set_xticks(np.arange(len(occluders)))
            ax.set_yticks(np.arange(len(occluders)))
            ax.set_xticklabels(occluders)
            ax.set_yticklabels(occluders)
            ax.tick_params(direction='in')
            ax.set_xlabel('occlusion type trained on')
            ax.set_ylabel('occlusion type trained on')
            fig.colorbar(im, fraction=0.0453)
            for i in range(len(occluders)):
                for j in range(len(occluders)):
                    text = ax.text(j, i, f'{cormat[i, j]:.2f}',
                                   ha="center", va="center", color="k")
            ax.set_title(f'correlation of LRP maps of {occUnocc} images')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
            plt.tight_layout()
            outDir = f'DNN/analysis/results/LRPcorrelation/{modelName}/{dataset}'
            os.makedirs(outDir, exist_ok=True)
            plt.savefig(f'{outDir}/{occUnocc}.png')
            plt.show()


