# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for base model plus minor architectural changes
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'supervised, unoccluded (shared weights)': {
        'path': 'cornet_s/xform-cont',
        'color': TAB20C[0],
    },
    'supervised, unoccluded (unshared weights)': {
        'path': 'cornet_s_unshared/xform-cont',
        'color': TAB20C[1],
    },
    'self-supervised, unoccluded (shared weights)': {
        'path': 'cornet_s/task-cont',
        'color': TAB20C[2],
    },
    'self-supervised, unoccluded (unshared weights)': {
        'path': 'cornet_s_unshared/task-cont',
        'color': TAB20C[3],
    },
    'supervised, natural occluders (shared weights)': {
        'path': 'cornet_s/xform-cont_occ-nat',
        'color': TAB20C[8],
    },
    'supervised, natural occluders (unshared weights)': {
        'path': 'cornet_s_unshared/xform-cont_occ-nat',
        'color': TAB20C[9],
    },
    'self-supervised, natural occluders (shared weights)': {
        'path': 'cornet_s/task-cont_occ-nat',
        'color': TAB20C[10],
    },
    'self-supervised, natural occluders (unshared weights)': {
        'path': 'cornet_s_unshared/task-cont_occ-nat',
        'color': TAB20C[11],
    },
    'supervised, behavioral occluders (shared weights)': {
        'path': 'cornet_s/xform-cont_occ-beh',
        'color': TAB20C[4],
    },
    'supervised, behavioral occluders (unshared weights)': {
        'path': 'cornet_s_unshared/xform-cont_occ-beh',
        'color': TAB20C[5],
    },
    'self-supervised, behavioral occluders (shared weights)': {
        'path': 'cornet_s/task-cont_occ-beh',
        'color': TAB20C[6],
    },
    'self-supervised, behavioral occluders (unshared weights)': {
        'path': 'cornet_s_unshared/task-cont_occ-beh',
        'color': TAB20C[7],
    },
}
