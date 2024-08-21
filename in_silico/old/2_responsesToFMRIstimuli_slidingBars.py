import os
import sys
import glob
import datetime
import torchvision.transforms as transforms
import time
#time.sleep(3600)

sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from saveOutputs import saveOutputs
transform = None
version = 'v3'
models = ['alexnet']
datasets = ['imagenet16'] #, 'places365_standard', 'imagenet16'],
occluders = ['unoccluded', 'naturalTextured2', 'barHorz08']#['unoccluded', 'horzBars', 'polkadot', 'crossBars', 'dropout']
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
occludersNoLevels = ['unoccluded','naturalTextured','naturalTextured2']
for modelName in models:

    for dataset in datasets:
        datasetDir = f'/home/dave/Datasets/{dataset}'

        for occluder in occluders:

            modelDir = None
            if occluder in occludersNoLevels:
                coverageString = None
                modelDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occluder)
            else:
                coverage = .5
                modelDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occluder, str(int(coverage * 100)))

            paramsDir = os.path.join(modelDir, 'params')

            if weights == 'final':
                paramsFile = sorted(glob.glob(os.path.join(paramsDir, '*.pt')))[-1]

            imageDir = f'DNN/images/slidingHorzBar8/occluded'
            responseDir = os.path.join(modelDir, 'responses', 'fMRIstim', 'slidingBars')
            os.makedirs(responseDir, exist_ok=True)

            # loop over images and measure responses
            exemplarNames = ['bear', 'bison', 'elephant', 'hare', 'jeep', 'lamp', 'sportsCar', 'teapot']
            for exemplarName in exemplarNames:
                for o in range(28):
                    imagePath = f'{imageDir}/{exemplarName}/{o:02}.png'
                    responsePath = f'{responseDir}/{exemplarName}/{o:02}.pkl'
                    os.makedirs(os.path.dirname(responsePath), exist_ok=True)
                    print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Measuring responses |'
                          f' Model: {modelName} | Trained on: {dataset} | OccTypeTrain: {occluder} |'
                          f' image: {exemplarName} {o}')
                    saveOutputs(modelName, dataset, paramsFile, imagePath, responsePath, transform)