# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for base model plus minor architectural changes
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'pretrained': {
        'path': 'cornet_s/pretrained',
        'color': TAB20B[11],
    },
    'supervised, unoccluded': {
        'path': 'cornet_s_custom/base-model',
        'color': TAB20C[0],
    },
    'supervised, unoccluded (more features': {
        'path': 'cornet_s_custom/feat-512',
        'color': TAB20B[16],
    },
    'supervised, unoccluded (larger kernel': {
        'path': 'cornet_s_custom/kern-5',
        'color': TAB20B[17],
    },
    'supervised, unoccluded (no recurrence': {
        'path': 'cornet_s_custom/rec-0',
        'color': TAB20B[18],
    },
    'supervised, unoccluded (double recurrence': {
        'path': 'cornet_s_custom/rec-2x',
        'color': TAB20B[19],
    },
    'supervised, horz bar occluders': {
        'path': 'cornet_s_custom/occ-fmri',
        'color': TAB20B[10],
    },
    'supervised, behav. occluders': {
        'path': 'cornet_s_custom/occ-beh',
        'color': TAB20C[4],
    },
    'supervised, natural occluders': {
        'path': 'cornet_s_custom/occ-nat',
        'color': TAB20C[8],
    },
    'finetune_mixed, behav. occluders': {
        'path': 'cornet_s_custom/base-model/'
                'finetune_task-class-cont_occ-beh_high-vis',
        'color': TAB20B[14],
    },
    'finetune_mixed, natural occluders': {
        'path': 'cornet_s_custom/base-model/'
                'finetune_task-class-cont_occ-nat_high-vis',
        'color': TAB20B[15],
    },
}
