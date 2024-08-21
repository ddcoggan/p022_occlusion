# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for publically available pretrained models
"""

import sys
import os.path as op
sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from plot_utils import distinct_colors
COLS = list(distinct_colors.values())

models = {
    'AlexNet': {
        'path': 'alexnet/pretrained',
        'color': COLS[0],
        'weights': 'IMAGENET1K_V1',
    },
    'CORnet-Z': {
        'path': 'cornet_z/pretrained',
        'color': COLS[1],
    },
    #'CORnet-RT': {
    #    'path': 'cornet_rt/pretrained',
    #    'color': COLS[2],
    #},
    'CORnet-S': {
        'path': 'cornet_s/pretrained',
        'color': COLS[3],
    },
    'VGG-19': {
        'path': 'vgg19/pretrained',
        'color': COLS[4],
        'weights': 'IMAGENET1K_V1',
    },
    'ResNet-101': {
        'path': 'resnet101/pretrained',
        'color': COLS[5],
        'weights': 'IMAGENET1K_V2',
    },
    'ResNext-101': {
        'path': 'resnext101_32x8d/pretrained',
        'color': COLS[6],
        'weights': 'IMAGENET1K_V1',
    },
    'ResNet-152': {
        'path': 'resnet152/pretrained',
        'color': COLS[7],
        'weights': 'IMAGENET1K_V2',
    },
    'ViT_B_16': {
        'path': 'vit_b_16/pretrained',
        'color': COLS[8],
        'weights': 'IMAGENET1K_V1',
    },
    #'PredNet': {
    #    'path': 'prednet/pretrained',
    #    'color': COLS[9],
    #},
}
