# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for publically available pretrained models
"""

import matplotlib
TABCOLS = matplotlib.cm.tab10.colors

models = {
    'pretrained': {
        'path': 'vit_b_16/pretrained',
        'color': TABCOLS[0],
        'kwargs': {'weights': 'IMAGENET1K_V1'},
    },
    'blur-trained': {
        'path': 'vit_b_16/blur-trained',
        'color': TABCOLS[1],
        'kwargs': {'image_size': 384},
    },
}
