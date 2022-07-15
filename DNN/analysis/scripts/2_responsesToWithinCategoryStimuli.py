import os
import sys
import glob
import datetime
import torchvision.transforms as transforms
import time
#time.sleep(3600)

sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from saveOutputs import saveOutputs
overwrite = True
transform = None
version = 'v3'
models = ['alexnet']#['PredNetImageNet', 'alexnet', 'resnet152', 'cornet_s', 'vgg19', 'inception_v3']
datasets = ['imagenet16'] #, 'places365_standard', 'imagenet16'],
occluders = ['unoccluded', 'naturalTextured2', 'barHorz04']#['unoccluded', 'horzBars', 'polkadot', 'crossBars', 'dropout']
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
occludersNoLevels = ['unoccluded','naturalTextured','naturalTextured2']

for modelName in models:

    for dataset in datasets:
        datasetDir = f'/home/dave/Datasets/{dataset}'

        for occluder in occluders:

            if occluder in occludersNoLevels:
                coverageString = None
                modelDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occluder)
            else:
                coverage = .5
                modelDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occluder, str(int(coverage * 100)))

            paramsDir = os.path.join(modelDir, 'params')

            if weights == 'final':
                paramsFile = sorted(glob.glob(os.path.join(paramsDir, '*.pt')))[-1]

            imagesUnocc = sorted(glob.glob(f'DNN/images/withinCategoryBarHorz04/*unoccluded*'))
            imagesOccUpper = sorted(glob.glob(f'DNN/images/withinCategoryBarHorz04/*upper*'))
            imagesOccLower = sorted(glob.glob(f'DNN/images/withinCategoryBarHorz04/*lower*'))
            imagePaths = []
            condNames = []
            for e in range(8):
                imagePaths.append(imagesUnocc[e])
                imagePaths.append(imagesOccUpper[e])
                imagePaths.append(imagesOccLower[e])
                condNames.append(f'{os.path.basename(imagesUnocc[e]).split(".")[0]}')
                condNames.append(f'{os.path.basename(imagesOccUpper[e]).split(".")[0]}')
                condNames.append(f'{os.path.basename(imagesOccLower[e]).split(".")[0]}')


            responseDir = os.path.join(modelDir, 'responses', 'withinCategory')
            os.makedirs(responseDir, exist_ok=True)

            # loop over images and measure responses
            for i, imageFile in enumerate(imagePaths):
                imageNameFull = os.path.basename(imageFile)
                imageFormat = imageNameFull.split('.')[-1]
                imageName = imageNameFull[:-(len(imageFormat) + 1)]
                responsePath = f'{responseDir}/{condNames[i]}.pkl'
                if not os.path.isfile(responsePath) or overwrite:
                    print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Measuring responses |'
                          f' Model: {modelName} | Trained on: {dataset} | OccTypeTrain: {occluder} |'
                          f' image: {condNames[i]}')
                    saveOutputs(modelName, dataset, paramsFile, imageFile, responsePath, transform)