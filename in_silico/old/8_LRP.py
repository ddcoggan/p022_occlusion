import os
import sys
import glob
import datetime
import torchvision.transforms as transforms
import time
#time.sleep(3600)

sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from getLRPmaps import getLRPmaps
transform = None
version = 'v3'
models = ['alexnet']
trainsets = ['imagenet16'] #, 'places365_standard', 'imagenet16'],
occlusionTypes = ['unoccluded', 'barHorz08', 'naturalTextured2']#['unoccluded', 'horzBars', 'polkadot', 'crossBars', 'dropout']
occludersBehavioural = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 'polkadot','polkasquare','crossBarOblique','crossBarCardinal', 'naturalUntexturedCropped2']
occlusionMethodsNoLevels = ['unoccluded','naturalTextured','naturalTextured2']
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
for model in models:

    for trainset in trainsets:
        datasetDir = f'/home/dave/Datasets/{trainset}'

        for occTypeTrain in occlusionTypes:

            if occTypeTrain in occlusionMethodsNoLevels:
                modelDir = os.path.join(os.getcwd(), 'DNN/data', model, trainset, 'fromPretrained', occTypeTrain)
            else:
                modelDir = os.path.join(os.getcwd(), 'DNN/data', model, trainset, 'fromPretrained', occTypeTrain, '50')

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


            responseDir = os.path.join(modelDir, 'LRP', 'fMRIstim', version)
            os.makedirs(responseDir, exist_ok=True)

            # loop over images and measure responses
            for i, imageFile in enumerate(imagePaths):
                imageNameFull = os.path.basename(imageFile)
                imageFormat = imageNameFull.split('.')[-1]
                imageName = imageNameFull[:-(len(imageFormat) + 1)]
                responsePath = f'{responseDir}/{condNames[i]}.png'
                classID = classIDs[i]
                print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Performing LRP |'
                      f' Model: {model} | Trained on: {trainset} | OccTypeTrain: {occTypeTrain} |'
                      f' image: {condNames[i]}')
                getLRPmaps(model, trainset, paramsFile, imageFile, responsePath, classID, returnFigure=False)