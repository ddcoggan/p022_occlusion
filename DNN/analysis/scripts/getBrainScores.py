import sys
import os
import glob
import functools
import pandas as pd
import pickle

if os.uname().nodename == 'finn':
    sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
elif os.uname().nodename == 'u110380':
    sys.path.append('/home/exx/Dave/masterScripts/DNN')

from getModel import getModel
from loadParams import loadParams

sys.path.append('/mnt/HDD12TB/masterScripts/DNN/BrainScore/model-tools-master')
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment

sys.path.append('/mnt/HDD12TB/masterScripts/DNN/BrainScore/brain-score-master')
from brainscore import score_model

overwrite = False
modelNames = ['cornet_s_varRec']
times = [[2,2,4,2], [5,5,10,5]]
occTypeTrain = ['unoccluded','barHorz08']

testConfig = {'V1': ['movshon.FreemanZiemba2013public.V1-pls'],
              'V2': ['movshon.FreemanZiemba2013public.V2-pls'],
              'V4': ['dicarlo.MajajHong2015public.V4-pls'],
              'IT': ['dicarlo.MajajHong2015public.IT-pls']}

outDir = 'analysis/results/brainScore'
if not os.path.isdir(outDir):
    os.makedirs(outDir)

scoresPath = 'analysis/results/brainScore/scores.pkl'
if not os.path.isdir(os.path.dirname(scoresPath)):
    scores = pd.DataFrame({'modelName': [], 'times': [], 'occTypeTrain': [], 'layer': [], 'benchmark': [], 'score': []})
else:
    scores = pickle.load(open(scoresPath, 'rb'))

for modelName in modelNames:
    for time in times:

        timeString = f'_{time[0]}_{time[1]}_{time[2]}_{time[3]}'
        modelNameFull = f'{modelName}{timeString}'

        # load model
        config = {'pretrained': False, 'times': time}
        model = getModel(modelName, **config)
        model = model.module # remove module wrapping to not confuse brainscore

        for occ in occTypeTrain:

            # get weights
            modelDir = f'DNN/data/{modelNameFull}/imagenet1000/fromPretrained/{occ}'
            if occ != 'unoccluded':
                modelDir = f'{modelDir}/50'
            paramsPath = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))[-1]
            model = loadParams(model, paramsPath)

            # set up model for brain score
            preprocessing = functools.partial(load_preprocess_images, image_size=224)
            identifier = modelNameFull
            activations_model = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)

            # test each layer on each specified benchmark
            for layer in testConfig:

                modelCommit = ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layer)

                for benchmark in testConfig[layer]:

                    alreadyDone = len(scores[(scores['modelName'] == modelName) &
                                             (scores['times'] == timeString[1:]) &
                                             (scores['occTypeTrain'] == occ) &
                                             (scores['layer'] == layer) &
                                             (scores['benchmark'] == benchmark)]) > 0

                    if not alreadyDone or overwrite:

                        #result = score_model(model_identifier=identifier, model=modelCommit, benchmark_identifier=benchmark)
                        #score = result[0].item()
                        score = 1
                        scores = pd.concat([scores, pd.DataFrame({'modelName': [modelName],
                                                                  'times': [timeString[1:]],
                                                                  'occTypeTrain': [occ],
                                                                  'layer': [layer],
                                                                  'benchmark': [benchmark],
                                                                  'score': [score]})])
                        pickle.dump(scores, open(scoresPath, 'wb'))
