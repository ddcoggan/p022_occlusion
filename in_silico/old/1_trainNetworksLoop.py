import os
import glob
import sys
import datetime

sys.path.append(f'{os.path.expanduser("~")}/Dave/masterScripts/DNN')
from train import train
import time
#time.sleep(18000)

overwrite = False

# set default / general parameters
printTheseParams = ['modelName','times','RF','occluder']
config = {'modelParams': {'pretrained': False},
          'datasetParams': {'propOccluded': .8,
                            'colours': [(0,0,0),(127,127,127),(255,255,255)],
                            'invert': False},
          'trainingParams': {'learningRate': 2**-7,
                             'optimizerName': 'SGD',
                             'nEpochs': 32,
                             'workers': 8,
                             'batchSize': 32,
                             'nGPUs': 1,
                             'GPUids': 1}}

#analyses = {'test': {'alexnet': {'imagenet1000': {'occluders': ['unoccluded']}}}}
analyses = {'cornet_s_custom_Rec1-1-1-1_RF3-3-3-3': {'cornet_s_custom': {'imagenet1000': {'occluders': ['unoccluded'], 'coverages': [.5], 'times': (1,1,1,1), 'RF': (3,3,3,3)}}}}
            #'cornet_s_custom_Rec1-2-4-2_RF3-3-3-3': {'cornet_s_custom': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': (1,2,4,2), 'RF': (3,3,3,3)}}}}
            #'cornet_s_custom_Rec2-4-8-4_RF3-3-3-3': {'cornet_s_custom': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': (2,4,8,4), 'RF': (3,3,3,3)}}}}
            #'cornet_s_custom_Rec1-2-4-2_RF5-5-5-5': {'cornet_s_custom': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': (1,2,4,2), 'RF': (5,5,5,5)}}}}
            #'cornet_s_custom_Rec1-2-4-2_RF7-7-7-7': {'cornet_s_custom': {'imagenet1000': {'occluders': ['barHorz08', 'unoccluded'], 'coverages': [.5], 'times': (1,2,4,2), 'RF': (7,7,7,7)}}}}
            #'cornet_s_custom_predify_Rec1-2-4-2_RF3-3-3-3': {'cornet_s_custom_predify': {'imagenet1000': {'occluders': ['barHorz08'], 'coverages': [.5], 'times': (1,2,4,2), 'RF': (3,3,3,3)}}}}

# list various occluder types and levels
occluders = []
for x in sorted(glob.glob('DNN/images/occluders/*')):
    occluders.append(os.path.basename(x))
occluders.append('unoccluded')

occludersFMRI = ['barHorz04','barVert12','barHorz08']
occludersBehavioural = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 'polkadot','polkasquare','crossBarOblique','crossBarCardinal', 'naturalUntexturedCropped2']
occludersNoLevels = ['naturalTextured','naturalTextured2']
occludersWithLevels = occluders
for o in occludersNoLevels:
    occludersWithLevels.remove(o)
occludersWithLevels.remove('unoccluded')

# standard coverage levels
indCoverages = [.1,.2,.4,.8]

for analysis in analyses:

    for modelName in analyses[analysis]:

        config['modelParams']['modelName'] = modelName

        # force 1 gpu for certain networks
        if modelName.startswith('cornet') or modelName in ['PredNetImageNet']:
            config['trainingParams']['nGPUs']=1

        # force pretrained for certain networks
        if modelName.startswith('cornet_s_custom'):
            config['modelParams']['pretrained'] = False
        elif modelName in 'PredNetImageNet':
            config['modelParams']['pretrained'] = False

        # set pretrained string
        if config['modelParams']['pretrained']:
            pretrainedString = 'fromPretrained'
        else:
            pretrainedString = 'fromScratch'

        # get batch size
        config['trainingParams']['batchSize'] = batchSizes[modelName]

        # remove warnings for predified networks
        if modelName.endswith('predify'):
            import warnings
            warnings.simplefilter("ignore")

        for dataset in analyses[analysis][modelName]:

            config['datasetParams']['datasetPath'] = f'{os.path.expanduser("~")}/Datasets/{dataset}'
            config['datasetParams']['datasetName'] = dataset

            modelNameFull = modelName
            if modelName.startswith('cornet_s_'):
                config['modelParams']['times'] = analyses[analysis][modelName][dataset]['times']
                modelNameFull = analysis
                config['modelParams']['RF'] = analyses[analysis][modelName][dataset]['RF']

            for occluder in analyses[analysis][modelName][dataset]['occluders']:

                config['datasetParams']['occluder'] = occluder
                if type(occluder) is list:
                    occluderString = 'mixedOccluders' # if different occluder types are used simultaneously
                else:
                    occluderString = occluder

                if occluder == 'unoccluded':
                    coverages = [None]
                    coverageString = ['no coverage']

                elif occluder in occludersNoLevels:
                    coverages = [None]
                    coverageString = ['all occluders']
                else:
                    coverages = analyses[analysis][modelName][dataset]['coverages']

                outDir = os.path.join('DNN/data', modelNameFull, dataset, pretrainedString, occluder)

                for coverage in coverages:

                    # set up out dir based on occlusion, blur, noise

                    # occlusion
                    if type(coverage) is list:
                        coverageString = 'mixedLevels' # if different occluder levels are used simultaneously
                    elif coverage:
                        coverageString = str(int(coverage*100))

                    if coverage:
                        outDir = os.path.join(outDir, coverageString)
                        config['datasetParams']['coverage'] = coverage

                    # blur
                    if 'mixedBlur' in analysis:
                        outDir += '_mixedBlur'
                        config['datasetParams']['blur'] = True
                        config['datasetParams']['blurSigmas'] = [0, 1, 2, 4, 8],
                        config['datasetParams']['blurWeights'] = [.2, .2, .2, .2, .2]
                    else:
                        config['datasetParams']['blur'] = False

                    # noise
                    if 'mixedNoise' in analysis:
                        outDir += '_mixedNoise'
                        config['datasetParams']['noise'] = True
                        config['datasetParams']['noiseLevels'] = [1,.8,.4,.2,.1]
                    else:
                        config['datasetParams']['noise'] = False

                    config['modelParams']['outDir'] = outDir

                    if os.path.exists(f'{outDir}/plotStats.pkl'):
                        import shutil
                        shutil.copy(f'{outDir}/plotStats.pkl', f'{outDir}/plots/plotStats.pkl')

                    # get most recent params file if possible
                    paramsPaths = sorted(glob.glob(os.path.join(outDir, 'params/*.pt')))
                    if paramsPaths and not overwrite:
                        config['modelParams']['lastEpoch'] = int(os.path.basename(paramsPaths[-1])[:-3])
                    else:
                        config['modelParams']['lastEpoch'] = None

                    # call script
                    if len(paramsPaths) < config['trainingParams']['nEpochs']+1 or overwrite:

                        # set GPUs
                        if config['trainingParams']['nGPUs'] == 1:
                            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                            os.environ["CUDA_VISIBLE_DEVICES"] = f"{config['trainingParams']['GPUids']}"

                        # create config dictionary
                        printOut = {}
                        for param in printTheseParams:
                            for set in config:
                                if param in config[set]:
                                    printOut[param] = config[set][param]

                        # print out config information
                        printString = f'Started at {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
                        for p in printOut:
                            printString += f' | {p}: {printOut[p]}'
                        print(printString)

                        # train
                        train(**config)

'''
analyses = {'allAlexnet': {'alexnet': {'imagenet16': {'occluders': occluders, 'coverages': indCoverages}}},
            'mixedTypes_mixedLevels': {'vgg19': {'imagenet16': {'occluders': [occludersBehavioural], 'coverages': [indCoverages]}}},
            'mixedLevels_mixedBlur': {'alexnet': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}}},
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
