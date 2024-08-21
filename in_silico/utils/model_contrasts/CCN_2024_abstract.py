# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for larger model developed for better contrastive learning
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

models = {
    #'supervised, unoccluded, standard transform (pretrained)': {
    #    'path': 'cornet_s/pretrained',
    #    'color': TAB20C[0],
    #},
    'supervised, unoccluded': {
        'path': 'cornet_s/xform-cont',
        'color': TAB20C[0],
        'xpos': 0
    },
    #'self-supervised, unoccluded': {
    #    'path': 'cornet_s/task-cont',
    #    'color': TAB20C[2],
    #},
    'supervised, natural occluders with textures': {
        'path': 'cornet_s/occ-nat_tex-nat_vis-beh_xform-cont',
        'color': TAB20B[4],
        'xpos': 3
    },
    #'self-supervised, natural occluders with textures': {
    #    'path': 'cornet_s/occ-nat_tex-nat_vis-beh_task-cont',
    #    'color': TAB20B[6],
    #},
    'supervised, natural occluders without textures': {
        'path': 'cornet_s/occ-nat_tex-uni_vis-beh_xform-cont',
        'color': TAB20C[8],
        'xpos': 2
    },
    #'self-supervised, natural occluders without textures': {
    #    'path': 'cornet_s/occ-nat_tex-uni_vis-beh_task-cont',
    #    'color': TAB20C[10],
    #},
    'supervised, artificial occluders without textures': {
        'path': 'cornet_s/occ-art_vis-beh_xform-cont',
        'color': TAB20C[4],
        'xpos': 1
    },
    #'self-supervised, artificial occluders without textures': {
    #    'path': 'cornet_s/occ-art_vis-beh_task-cont',
    #    'color': TAB20C[6],
    #},
}
