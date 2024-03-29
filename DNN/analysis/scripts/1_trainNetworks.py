import os
import glob
import sys
import datetime
import time
#time.sleep(18000)

sys.path.append(f'{os.path.expanduser("~")}/Dave/masterScripts/DNN')


### CONFIGURATION ###

# often used occluders and visibility levels
occludersFMRI = ['barHorz04','barVert12','barHorz08']
occludersBehavioural = ['barHorz04','barVert04','barObl04','mudSplash','polkadot','polkasquare','crossBarOblique','crossBarCardinal', 'naturalUntexturedCropped2']
occludersNoLevels = ['naturalTextured','naturalTextured1','naturalTextured2']
visibilities = [.1,.2,.4,.8]

# model (modelName*, model, R, K, pretrained, outDir, lastEpoch)
# dataset (path*,occlusion{type,label,propOccluded,visibility,colour,label},
#          blur{sigmas,weights,label}, sigmas: [0, 1, 2, 4, 8]
#          noise{type,ssnr,label}) ssnr: [.1,.2,.4,.8,1]
# training (optimizerName*, nEpochs*, nGPUs, GPUids, learningRate*, batchSize*, workers)
m = { # model parameters
    'modelName': 'cornet_s_custom',
    'R': (1,2,4,2), # recurrence (cornet_s_custom)
    'K': (5,5,5,5), # kernel size (cornet_s_custom)
    }
d = { # dataset parameters
    'dataset': 'ILSVRC2012',
    'occlusion': {'type': occludersBehavioural,'label': 'behaviouralOccs',
                  'propOccluded': .8,'visibility': visibilities, 'visLabel': 'mixedVis',
                  'colour': [(0,0,0),(127,127,127),(255,255,255)]}
    }
t = { # training parameters
    'optimizerName': 'SGD',
    'nEpochs': 32,
    'nGPUs': 1,  # set to -1 to use all available GPUs, 0 to use CPU
    'GPUids': 1,  # ignored if nGPUs in [-1,0]
    'learningRate': 2**-7,
    'batchSize': 2**5
    }
printTheseParams = ['modelName','R','K']

### END OF CONFIGURATION ###


# set pretrained to False by default
if not 'pretrained' in m:
    m['pretrained'] = False

# enforce no pretraining for some models
elif m['pretrained'] and m['modelName'] in ['cornet_s_custom', 'cornet_s_custom_predify']:
    UserWarning(f"Pretrained weights for {m['modelName']} are not available, training from random initialized weights instead.")
    m['pretrained'] = False

# set model training directory
if 'outDir' not in m:
    pretrainedString = ['fromScratch', 'fromPretrained'][m['pretrained']]
    pathItems = [m['modelName']]
    if 'custom' in m['modelName']:
        pathItems += [f"recCycles_{m['R'][0]}-{m['R'][1]}-{m['R'][2]}-{m['R'][3]}"]
        pathItems += [f"kernelSizes_{m['K'][0]}-{m['K'][1]}-{m['K'][2]}-{m['K'][3]}"]
    pathItems += [d['dataset'], pretrainedString]

    # image distortions
    if 'occlusion' not in d and 'blur' not in d and 'noise' not in d:
        imageDist = 'unaltered'
    else:
        if 'occlusion' in d:
            try: imageDist = d['occlusion']['label']
            except: imageDist = d['occlusion']['type']
            try: imageDist += f"_vis-{d['occlusion']['visLabel']}"
            except: imageDist += f"_vis-{int(d['occlusion']['visibility']*100):02}"
        else:
            imageDist = 'unoccluded'
        if 'blur' in d:
            imageDist = f"__blur-{d['blur']['label']}"
        if 'noise' in d:
            imageDist += f"__noise-{d['noise']['label']}"

    pathItems += [imageDist]

    m['outDir'] = 'DNN/data'
    for item in pathItems:
        m['outDir'] = os.path.join(m['outDir'], item)

# get most recent params file if possible
if 'lastEpoch' not in m:
    paramsPaths = sorted(glob.glob(os.path.join(m['outDir'], 'params/*.pt')))
    if paramsPaths:
        m['lastEpoch'] = int(os.path.basename(paramsPaths[-1])[:-3])
    else:
        m['lastEpoch'] = None

# call script
if m['lastEpoch'] is None or m['lastEpoch'] < t['nEpochs']:

    # enforce single GPUs for some models
    if t['nGPUs'] not in [0, 1] and m['modelName'] in ['cornet_s_custom', 'cornet_s_custom_predify', 'PredNetImageNet']:
        print(f"{m['modelName']} cannot be trained on multiple GPUs, training on 1 GPU instead.")
        t['nGPUs'] = 1

    # set GPU visibility (do BEFORE anything using cuda (e.g. torch) is imported)
    if t['nGPUs'] == 1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{t['GPUids']}"

    # remove warnings for predified networks
    if m['modelName'].endswith('predify'):
        import warnings
        warnings.simplefilter("ignore")

    # make training configuration dictionary
    config = {'modelParams': m, 'datasetParams': d, 'trainingParams': t}

    # print out training configuration information
    printString = f'Started at {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
    for param in printTheseParams:
        if '.' not in param:
            for thisDict in config:
                if param in config[thisDict]:
                    printString += f' | {param}: {config[thisDict][param]}'
        else:
            thisDict, thisSubdict, thisParam = param.split('.')
            if thisDict in config and thisParam in config[thisDict][thisSubdict]:
                printString += f' | {thisParam}: {config[thisDict][thisSubdict][thisParam]}'
    print(printString)

    # train
    from train import train
    train(**config)
