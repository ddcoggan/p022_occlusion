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
overwrite = True
versions = ['v3']
models = ['cornet_s']#['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] #'alexnet', 'inception_v3', 'cornet_s', 'PredNetImageNet',
trainsets = ['imagenet16'] #, 'places365_standard', 'imagenet16'],
occludersNoLevels = ['unoccluded','naturalTextured','naturalTextured2']
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
for version in versions:

    occlusionTypes = ['unoccluded', 'naturalTextured']  # ['unoccluded', 'horzBars', 'polkadot', 'crossBars', 'dropout']
    if version == 'v1':
        occlusionTypes.append('barHorz04')
    elif version == 'v2':
        occlusionTypes.append('barVert12')
    elif version == 'v3':
        occlusionTypes.append('barHorz08')
    for model in models:

        for trainset in trainsets:
            datasetDir = f'/home/dave/Datasets/{trainset}'

            for occTypeTrain in occlusionTypes:

                if occTypeTrain in occludersNoLevels:
                    coverageString = None
                    modelDir = os.path.join('DNN/data', model, trainset, 'fromPretrained', occTypeTrain)
                else:
                    modelDir = os.path.join('DNN/data', model, trainset, 'fromPretrained', occTypeTrain, 'mixedLevels')

                paramsDir = os.path.join(modelDir, 'params')

                if weights == 'final':
                    paramsFile = sorted(glob.glob(os.path.join(paramsDir, '*.pt')))[-1]

                imagesUnocc = sorted(glob.glob(f'fMRI/{version}/images/unoccluded/**/*.JPEG', recursive=True))

                if version in ['v1','v3']:
                    imagesOccUpper = sorted(glob.glob(f'fMRI/{version}/images/occluded/*/top/*.JPEG'))
                    imagesOccLower = sorted(glob.glob(f'fMRI/{version}/images/occluded/*/bot/*.JPEG'))
                    imagePaths = []
                    condNames = []
                    for e in range(8):
                        imagePaths.append(imagesUnocc[e])
                        imagePaths.append(imagesOccUpper[e])
                        imagePaths.append(imagesOccLower[e])
                        condNames.append(f'{imagesUnocc[e].split("/")[-2]}_none')
                        condNames.append(f'{imagesUnocc[e].split("/")[-2]}_upper')
                        condNames.append(f'{imagesUnocc[e].split("/")[-2]}_lower')

                elif version == 'v2':
                    imagesOccLeft = sorted(glob.glob(f'fMRI/{version}/images/occluded/*/left/*.JPEG'))
                    imagesOccRight = sorted(glob.glob(f'fMRI/{version}/images/occluded/*/right/*.JPEG'))
                    imagePaths = []
                    condNames = []
                    for e in range(8):
                        imagePaths.append(imagesUnocc[e])
                        imagePaths.append(imagesOccUpper[e])
                        imagePaths.append(imagesOccLower[e])
                        condNames.append(f'{imagesUnocc[e].split("/")[-2]}_none')
                        condNames.append(f'{imagesUnocc[e].split("/")[-2]}_left')
                        condNames.append(f'{imagesUnocc[e].split("/")[-2]}_right')

                responseDir = os.path.join(modelDir, 'responses', 'fMRIstim', version)
                os.makedirs(responseDir, exist_ok=True)

                # loop over images and measure responses
                for i, imageFile in enumerate(imagePaths):
                    imageNameFull = os.path.basename(imageFile)
                    imageFormat = imageNameFull.split('.')[-1]
                    imageName = imageNameFull[:-(len(imageFormat) + 1)]
                    responsePath = f'{responseDir}/{condNames[i]}.pkl'
                    if not os.path.exists(responsePath) or overwrite:
                        print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Measuring responses |'
                              f' Model: {model} | Trained on: {trainset} | OccTypeTrain: {occTypeTrain} |'
                              f' version: {version} | image: {condNames[i]}')
                        saveOutputs(model, trainset, paramsFile, imageFile, responsePath, transform)
