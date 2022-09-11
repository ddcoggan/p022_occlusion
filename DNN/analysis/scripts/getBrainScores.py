import sys
import os
import glob
import functools
import pandas as pd
import pickle
import datetime

sys.path.append(f'{os.path.expanduser("~")}/Dave/masterScripts/DNN')
from getModel import getModel
from loadParams import loadParams

sys.path.append(f'{os.path.expanduser("~")}/Dave/repos/BrainScore/model-tools-master')
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment

sys.path.append(f'{os.path.expanduser("~")}/Dave/repos/BrainScore/brain-score-master')
from brainscore import score_model

overwrite = False
modelNames = ['cornet_s_custom']
times = [[1,1,1,1]]#[2,2,4,2], [5,5,10,5]]
RFs = [[3,3,3,3]]#[2,2,4,2], [5,5,10,5]]
occTypeTrain = ['unoccluded','barHorz08']

testConfig = {'V1': ['movshon.FreemanZiemba2013public.V1-pls'],
              'V2': ['movshon.FreemanZiemba2013public.V2-pls'],
              'V4': ['dicarlo.MajajHong2015public.V4-pls'],
              'IT': ['dicarlo.MajajHong2015public.IT-pls']}


modelFiles=glob.glob('/home/dave/.result_caching/**/*custom*', recursive = True)
for file in modelFiles:
    os.remove(file)

outDir = 'analysis/results/brainScore'
if not os.path.isdir(outDir):
    os.makedirs(outDir)

scoresPath = f'{outDir}/scores.pkl'
if not os.path.isfile(scoresPath):
    scores = pd.DataFrame({'modelName': [], 'times': [], 'RFs': [], 'occTypeTrain': [], 'layer': [], 'benchmark': [], 'score': []})
else:
    scores = pickle.load(open(scoresPath, 'rb'))

for modelName in modelNames:
    for time in times:
        for RF in RFs:

            timeString = f'_Rec{time[0]}-{time[1]}-{time[2]}-{time[3]}'
            RFstring = f'_RF{RF[0]}-{RF[1]}-{RF[2]}-{RF[3]}'
            modelNameFull = f'{modelName}{timeString}{RFstring}'

            # load model
            config = {'modelName': 'cornet_s_custom', 'times': time, 'RF': RF, 'pretrained': False}
            model = getModel(**config)
            model = model.module # remove module wrapping to not confuse brainscore

            for occ in occTypeTrain:

                # get weights
                modelDir = f'DNN/data/{modelNameFull}/imagenet1000/fromScratch/{occ}'
                if occ != 'unoccluded':
                    modelDir = f'{modelDir}/50'
                paramsPath = sorted(glob.glob(os.path.join(modelDir, 'params/*.pt')))[-1]
                model = loadParams(model, paramsPath)

                # set up model for brain score
                preprocessing = functools.partial(load_preprocess_images, image_size=224)
                identifier = f'{modelNameFull}_{occ}'
                activations_model = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)

                # test each layer on each specified benchmark
                for layer in testConfig:

                    modelCommit = ModelCommitment(identifier=identifier, activations_model=activations_model, layers=[layer])

                    for benchmark in testConfig[layer]:

                        alreadyDone = len(scores[(scores['modelName'] == modelName) &
                                                 (scores['times'] == timeString[1:]) &
                                                 (scores['RFs'] == RFstring[1:]) &
                                                 (scores['occTypeTrain'] == occ) &
                                                 (scores['layer'] == layer) &
                                                 (scores['benchmark'] == benchmark)]) > 0

                        if not alreadyDone or overwrite:

                            result = score_model(model_identifier=identifier, model=modelCommit, benchmark_identifier=benchmark)
                            score = result[0].item()
                            scores = pd.concat([scores, pd.DataFrame({'modelName': [modelName],
                                                                      'times': [timeString[1:]],
                                                                      'RFs': [RFstring[1:]],
                                                                      'occTypeTrain': [occ],
                                                                      'layer': [layer],
                                                                      'benchmark': [benchmark],
                                                                      'score': [score]})])
                            pickle.dump(scores, open(scoresPath, 'wb'))
                            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | model: {modelNameFull} | occluder: {occ} | layer: {layer} | score: {score} |')

