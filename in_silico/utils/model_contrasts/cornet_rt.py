# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'supervised, unoccluded, standard xform': {
        'path': 'cornet_rt_hw3/standard-training',
        'color': TAB20C[0],
    },
    'supervised, unoccluded, contrastive xform': {
        'path': 'cornet_rt_hw3/xform-cont',
        'color': TAB20C[1],
    },
    'self-supervised, unoccluded, contrastive xform': {
        'path': 'cornet_rt_hw3/task-cont',
        'color': TAB20C[2],
    },
    'supervised, natural occluders (high-vis), contrastive xform': {
        'path': 'cornet_rt_hw3/occ-nat_xform-cont',
        'color': TAB20C[10],
    },
    'self-sup, natural occluders (high-vis), contrastive xform': {
        'path': 'cornet_rt_hw3/occ-nat_task-cont/transfer_unocc',
        'color': TAB20C[11],
    },
    'supervised, natural occluders (all-vis), contrastive xform': {
        'path': 'cornet_rt_hw3/occ-nat_vis-beh_xform-cont',
        'color': TAB20C[8],
    },
    'self-sup, natural occluders (all-vis), contrastive xform': {
        'path': 'cornet_rt_hw3/occ-nat_vis-beh_task-cont/transfer_unocc',
        'color': TAB20C[9],
    },
    'supervised, behavioral occluders (high-vis), contrastive xform': {
        'path': 'cornet_rt_hw3/occ-beh_xform-cont',
        'color': TAB20C[6],
    },
    'self-sup, behavioral occluders (high-vis), contrastive xform': {
        'path': 'cornet_rt_hw3/occ-beh_task-cont/transfer_unocc',
        'color': TAB20C[7],
    },
    'supervised, behavioral occluders (all-vis), contrastive xform': {
        'path': 'cornet_rt_hw3/occ-beh_vis-beh_xform-cont',
        'color': TAB20C[4],
    },
    'self-sup, behavioral occluders (all-vis), contrastive xform': {
        'path': 'cornet_rt_hw3/occ-beh_vis-beh_task-cont/transfer_unocc',
        'color': TAB20C[5],
    },
}
   