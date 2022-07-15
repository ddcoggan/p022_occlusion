import os
import glob
import sys
import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from DNN.analysis.scripts.alterImages import occludeImages, addNoise, blurImages

sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from accuracy import accuracy
import zoo
from zoo.prednet import *

sys.path.append('/mnt/HDD12TB/masterScripts/DNN/zoo/CORnet_master')
import cornet

def plotFilters(modelName=None,modelDir=None,epoch=32,nClasses=16):

    weightsPath = f'{modelDir}/params/{epoch:03}.pt'
    weights = torch.load(weightsPath)['model']
    outDir = f'{modelDir}/filters_epoch{epoch}'
    os.makedirs(outDir, exist_ok=True)
    if modelName == 'alexnet':
        layer = 'module.features.0.weight'
    filters = weights[layer]
    nFilters,nChannels,x,y = filters.shape
    gridSize = math.ceil(np.sqrt(nFilters))
    montageSize = (x*gridSize, y*gridSize)
    montage = Image.new(size=montageSize, mode='RGB')
    for i in range(nFilters):
        imageArray = np.array(torch.Tensor.cpu(filters[i,:,:,:].permute(1, 2, 0)))
        imagePos = imageArray - imageArray.min() # rescale to between 0,255 for PIL
        imageScaled = imagePos * (255.0 / imagePos.max())
        image = Image.fromarray(imageScaled.astype(np.uint8))
        offset_x = (i % gridSize) * x
        offset_y = int(i / gridSize) * y
        montage.paste(image, (offset_x, offset_y))
    montage.save(f'{outDir}/{layer}.png')
    print(imageArray[0:3,0,0])

for occluder in ['barHorz02','barHorz04','barHorz08','barHorz16','barVert02','barVert04','barVert08','barVert16']:
    modelDir = f'DNN/data/alexnet/imagenet16/fromPretrained/{occluder}/mixedLevels'
    plotFilters('alexnet', modelDir, epoch=32, nClasses=16)
