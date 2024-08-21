# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for base model plus minor architectural changes
"""

import matplotlib
TAB20 = matplotlib.cm.tab20.colors

models = {
    'CORnet_S_V1': {
        'path': 'cornet_s_V1/v2',
        'color': TAB20[0],
    },
    'CORnet_S_V1_v6': {
        'path': 'cornet_s_V1_v6/xform-cont',
        'color': TAB20[0],
    },
    'CogNet_batchnorm': {
        'path': 'cognet_v9/v9',
        'color': TAB20[1],
    },
    'CogNet_groupnorm': {
        'path': 'cognet_v10/v10',
        'color': TAB20[2],
    },
}
