# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for larger model developed for better contrastive learning
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'supervised, unoccluded (standard xform)': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128',
        'color': TAB20C[0],
    },
    'supervised, unoccluded (contrastive xform)': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_xform-cont',
        'color': TAB20C[1],
    },
    'self-sup., unoccluded (contrastive xform)': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_task-cont/transfer_unocc',
        'color': TAB20C[2],
    },
    'supervised, natural occluders (standard xform)': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat',
        'color': TAB20C[8],
    },
    'supervised, natural occluders (contrastive xform)': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat_xform-cont',
        'color': TAB20C[9],
    },
    'self-sup., natural occluders (contrastive xform)': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ'
             '-nat_task-cont/transfer_unocc',
        'color': TAB20C[10],
    },
    'supervised, natural untext. occluders (contrastive xform)': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat-untex_xform-cont',
        'color': TAB20B[5],
    },
    'supervised, behavioral occluders (standard xform)': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-beh',
        'color': TAB20C[4],
    },
    'supervised, behavioral occluders (contrastive xform)': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-beh_xform-cont',
        'color': TAB20C[5],
    },
    'self-sup., behavioral occluders (contrastive xform)': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-beh_task-cont/'
                'transfer_unocc',
        'color': TAB20C[6],
    },
}
