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
    'PredNet': {
        'path': 'prednet/pretrained',
        'color': COLS[9],
    },
}
