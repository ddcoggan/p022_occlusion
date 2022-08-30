import os
import glob
import sys

if os.uname().nodename == 'finn':
    masterScriptsDir = '/mnt/HDD12TB/masterScripts'
    datasetsDir = '/home/dave/Datasets'
elif os.uname().nodename == 'u110380':
    masterScriptsDir = '/home/exx/Dave/masterScripts'
    datasetsDir = '/home/exx/Datasets'
sys.path.append(f'{masterScriptsDir}/DNN')
from train import train
import time
#time.sleep(18000)

nGPUs = 1 # target number of GPUs. Overwritten for known 1 GPU networks, set as -1 for all.
GPUids = 1 # specify GPUs to use if not all are/can be used

overwrite = False

noise=False
indCoverages = [.1,.2,.4,.8]
occluders = []
for x in sorted(glob.glob('DNN/images/occluders/*')):
        occluders.append(os.path.basename(x))
occluders.append('unoccluded')
occludersFMRI = ['barHorz04','barVert12','barHorz08']
occludersNoLevels = ['unoccluded','naturalTextured','naturalTextured2']
occludersBehavioural = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 'polkadot','polkasquare','crossBarOblique','crossBarCardinal', 'naturalUntexturedCropped2']
occludersWithLevels = occluders
for o in occludersNoLevels:
    occludersWithLevels.remove(o)

skipZeroth = False
nEpochs = 25
workers = 8
pretrained = True
colours = [(0,0,0),(127,127,127),(255,255,255)]
invert=False
propOccluded = 0.8
cycles=3 # currently just for prednet
momentum=.9 # currently just for prednet
weight_decay=1e-4 # currently just for prednet

'''
config = {'allAlexnet': {'alexnet': {'imagenet16': {'occluders': occluders, 'coverages': indCoverages}}},
          'mixedTypesMixedLevels': {'vgg19': {'imagenet16': {'occluders': [occludersBehavioural], 'coverages': [indCoverages]}}},
          'mixedLevelsMixedBlur': {'alexnet': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}}},
          'mixedLevels': {#'resnet18': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          #'resnet34': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          #'resnet50': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          #'resnet101': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'resnet152': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'vgg19': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'cornet_s': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'PredNetImageNet': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'inception_v3': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'alexnet': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}}},
                                      #'imagenet1000': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}}},
          'fMRIandNatural': {'alexnet': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'cornet_s': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             #'resnet18': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             #'resnet34': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             #'resnet50': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             #'resnet101': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'resnet152': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'vgg19': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'inception_v3': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'PredNetImageNet': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}}},
                             '''
config = {#'cornet_s_varRec_1_1_1_1': {'cornet_s_varRec': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': [1,1,1,1]}}}}
          #'cornet_s_varRec_2_2_4_2': {'cornet_s_varRec': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': [2,2,4,2]}}},
          #'cornet_s_varRec_5_5_10_5': {'cornet_s_varRec': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': [5,5,10,5]}}},
          #'cornet_s_varRec_1_1_1_1': {'cornet_s_varRec': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': [1,1,1,1]}}}}
          'cornet_s_varRec_varRF_2_2_4_2-9': {'cornet_s_varRec_varRF': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': [2,2,4,2], 'RF': 9}}},
          'cornet_s_varRec_varRF_2_2_4_2-7': {'cornet_s_varRec_varRF': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': [2,2,4,2], 'RF': 7}}},
          'cornet_s_varRec_varRF_2_2_4_2-5': {'cornet_s_varRec_varRF': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': [2,2,4,2], 'RF': 5}}}}

learningRate = .001
optimizerName = 'SGD'
batchSizes = {'alexnet': 1024,
              'vgg19': 128,
              'cornet_s': 256,
              'cornet_s_varRec': 8,
              'cornet_s_varRec_varRF': 8,
              'resnet18': 512,
              'resnet34': 256,
              'resnet50': 64,
              'resnet101': 64,
              'resnet152': 64,
              'inception_v3': 32,
              'PredNetImageNet': 8}


for analysis in config:

    for modelName in config[analysis]:

        # force 1 gpu for certain networks
        if modelName.startswith('cornet') or modelName in ['PredNetImageNet']:
            nGPUs=1


        batchSize = batchSizes[modelName]

        for dataset in config[analysis][modelName]:

            datasetPath = f'{datasetsDir}/{dataset}'

            times = None
            RF=None
            modelNameFull = modelName
            if modelName.startswith('cornet_s_varRec'):
                times = config[analysis][modelName][dataset]['times']
                modelNameFull = analysis
                if modelName.endswith('varRF'):
                    RF = config[analysis][modelName][dataset]['RF']

            for occluder in config[analysis][modelName][dataset]['occluders']:

                if type(occluder) is list:
                    occluderString = 'mixedOccluders'
                else:
                    occluderString = occluder

                for coverage in config[analysis][modelName][dataset]['coverages']:

                    if type(coverage) is list:
                        coverageString = 'mixedLevels'
                    else:
                       coverageString = str(int(coverage*100))

                    if occluder in occludersNoLevels:
                        outDir = os.path.join('DNN/data', modelNameFull, dataset, 'fromPretrained', occluder)
                    else:
                        outDir = os.path.join('DNN/data', modelNameFull, dataset, 'fromPretrained', occluderString, coverageString)

                    if analysis.endswith('MixedBlur'):
                        outDir += 'MixedBlur'
                        blur = True
                    else:
                        blur = False


                    # get restart from file if necessary
                    weightFiles = sorted(glob.glob(os.path.join(outDir, 'params/*.pt')))
                    if 0 < len(weightFiles) and overwrite == False:
                        restartFrom = weightFiles[-1]
                    else:
                        restartFrom = None

                    # print out these values during training
                    printOut = {'model': modelNameFull,
                                'dataset': dataset,
                                'occluder': occluder,
                                'coverage': str(coverage)}

                    # call script
                    if len(weightFiles) < nEpochs+1 or overwrite:

                        # set GPUs first
                        if nGPUs == 1:
                            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                            os.environ["CUDA_VISIBLE_DEVICES"] = f'{GPUids}'

                        train(modelName=modelName,
                              model=None,
                              datasetPath=datasetPath,
                              pretrained=pretrained,
                              learningRate=learningRate,
                              optimizerName=optimizerName,
                              batchSize=batchSize,
                              nEpochs=nEpochs,
                              restartFrom = restartFrom,
                              workers=workers,
                              outDir=outDir,
                              occlude=True,
                              occlusionMethod=occluder,
                              coverage=coverage,
                              propOccluded=propOccluded,
                              colour=colours,
                              invert=invert,
                              cycles=cycles,
                              momentum=momentum,
                              weight_decay=weight_decay,
                              printOut=printOut,
                              blur=blur,
                              blurSigmas = [0,1,2,4,8],
                              blurWeights=[.2,.2,.2,.2,.2],
                              noise=noise,
                              noiseLevels=[1,.8,.4,.2,.1],
                              times=times,
                              RF=RF,
                              nGPUs=nGPUs,
                              GPUids=GPUids,
                              skipZeroth=skipZeroth)

