import os
import sys
import glob
import datetime
import torchvision.transforms as transforms
import time
#time.sleep(3600)

overwrite = 0
sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from saveOutputs import saveOutputs
from DNN.analysis.scripts import alterImages

### CONFIGURATION
models = ['cornet_s_custom*']
#trainsets = ['ILSVRC2012'] #, 'places365_standard', 'imagenet16'],
#occlusionTypes = ['unaltered','barHorz08','behaviouralOccs_mixedVis']#['unoccluded', 'horzBars', 'polkadot', 'crossBars', 'dropout']
weights = 'final' # select which weights to use. 'final' is last training epoch. TODO 'maxEval' is epoch with highest evaluation performance.
coverages = [.4]#[.1, .2, .4, .8]
colours = [(0,0,0),(127,127,127),(255,255,255)]

with open(outPath, 'wb') as f:
    pickle.dump(activation, f, pickle.HIGHEST_PROTOCOL)
f.close()

for model in models:

    for trainset in trainsets:
        datasetDir = f'/home/dave/Datasets/{trainset}'

        for occTypeTrain in occlusionTypes:

            for coverageTrain in coverages:

                if occTypeTrain != 'unoccluded' or coverageTrain == .1: # don't run across multiple occlusion levels for unoccluded

                    modelDir = os.path.join(os.getcwd(), 'DNN/data', model, trainset, occTypeTrain, f'{int(coverageTrain*100)}')
                    if occTypeTrain == 'unoccluded': # no occlusion level subdirectory for unoccluded
                        modelDir = os.path.dirname(modelDir)

                    paramsDir = os.path.join(modelDir, 'params')

                    if weights == 'final':
                        paramsFile = sorted(glob.glob(os.path.join(paramsDir, '*.pt')))[-1]

                    for occTypeResponse in occlusionTypes:

                        for coverageResponse in coverages:

                            # to save disk space, planned analyses currently only look across coverage levels OR across occlusion types
                            if occTypeTrain == occTypeResponse or coverageTrain == coverageResponse:

                                transformSequence = [transforms.Resize(224),
                                                     transforms.CenterCrop(size=224),
                                                     transforms.ToTensor()]  # transform PIL image to Tensor.
                                if occTypeResponse != 'unoccluded':
                                    transformSequence.append(alterImages.occludeImage(method=occTypeResponse, coverage=coverageResponse, colour=colours, invert=False))
                                transformSequence.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))  # Only normalize on Tensor datatype.
                                transform = transforms.Compose(transformSequence)

                                categoryPaths = sorted(glob.glob(os.path.join(datasetDir, 'val/*')))

                                for c, categoryPath in enumerate(categoryPaths):
                                    category = os.path.basename(categoryPath)
                                    imageFiles = []
                                    for fileType in ['png', 'tif', 'jpg']:
                                        theseImages = sorted(glob.glob(os.path.join(categoryPath, f'*.{fileType}'), recursive=True))
                                        imageFiles += theseImages
                                    if len(imageFiles) == 0:
                                        raise Exception('No png, jpg or tif image files found!')
                                    else:
                                        print(f'Total of {len(imageFiles)} image files found.')

                                    responseDir = os.path.join(modelDir, 'responses', occTypeResponse,  f'{int(coverageResponse*100)}', category)
                                    os.makedirs(responseDir, exist_ok=True)

                                    # loop over images and measure responses
                                    for i, imageFile in enumerate(imageFiles):
                                        imageNameFull = os.path.basename(imageFile)
                                        imageFormat = imageNameFull.split('.')[-1]
                                        imageName = imageNameFull[:-(len(imageFormat) + 1)]
                                        responsePath = f'{responseDir}/{imageName}.pkl'
                                        if not os.path.isfile(responsePath) and imageFile:
                                            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Measuring responses |'
                                                  f' Model: {model} | Trained on: {trainset} | OccTypeTrain: {occTypeTrain} |'
                                                  f' OccTypeResponse: {occTypeResponse} | Category: {category} ({c + 1}/{len(categoryPaths)}) |'
                                                  f' Image: {imageName} ({i + 1}/{len(imageFiles)}) |')
                                            saveOutputs(model, trainset, paramsFile, imageFile, responsePath, transform)