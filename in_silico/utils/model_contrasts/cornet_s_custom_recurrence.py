# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for base model with and without recurrence
"""

import matplotlib
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'supervised, unnoccluded, no recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_xform-cont',
        'color': TAB20C[0],
    },
    'self-sup., unnoccluded, no recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_task-cont/'
                'transfer_unocc',
        'color': TAB20C[1],
    },
    'supervised, unnoccluded, standard recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_xform-cont',
        'color': TAB20C[2],
    },
    'self-sup., unnoccluded, standard recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_task-cont/transfer_unocc',
        'color': TAB20C[3],
    },
    'supervised, natural occluders, no recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_occ-nat_xform-cont',
        'color': TAB20C[8],
    },
    'self sup., natural occluders, no recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_occ-nat_task-cont/'
                'transfer_unocc',
        'color': TAB20C[9],
    },
    'supervised, natural occluders, standard recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat_xform-cont',
        'color': TAB20C[10],
    },
    'self sup., natural occluders, standard recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat_task-cont/'
                'transfer_unocc',
        'color': TAB20C[11],
    },
    'supervised, behavioral occluders, no recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_occ-beh_xform-cont',
        'color': TAB20C[4],
    },
    'self sup., behavioral occluders, no recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_occ-beh_task-cont/transfer_unocc',
        'color': TAB20C[5],
    },
    'supervised, behavioral occluders, standard recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-beh_xform-cont',
        'color': TAB20C[6],
    },
    'self sup., behavioral occluders, standard recurrence': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-beh_task-cont'
                '/transfer_unocc',
        'color': TAB20C[7],
    },
}
