# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for base model plus minor architectural changes
"""

import matplotlib
TAB20 = matplotlib.cm.tab20.colors

models = {
    'base model': {
        'path': 'cornet_s/pretrained',
        'color': TAB20[0],
    },
    'V1 model': {
        'path': 'cornet_s_V1/v2',
        'color': TAB20[1],
    },
}
