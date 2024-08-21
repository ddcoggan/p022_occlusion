# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for larger model developed for better contrastive learning
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
cols = matplotlib.cm.viridis.colors[::(256//3)]

models = {
    #'supervised, unoccluded, standard transform (pretrained)': {
    #    'path': 'cornet_s/pretrained',
    #    'color': TAB20C[0],
    #},
    'unoccluded, supervised': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_xform-cont',
        'color': cols[0],#TAB20C[0],
        'xpos': 0,
        'group': 'supervised\nCNNs',
    },
    'unoccluded, self-supervised': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_task-cont/transfer_unocc',
        'color': cols[0],#TAB20C[0],
        'xpos': 0,
        'group': 'self-supervised\nCNNs',
    },
    'artificial shapes, supervised': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-art-weak_xform-cont',
        'color': cols[1],#TAB20C[4],
        'xpos': 1,
        'group': 'supervised\nCNNs',
    },
    'artificial shapes, self-supervised': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-art-weak_task-cont'
                '/transfer_unocc',
        'color': cols[1],#TAB20C[4],
        'xpos': 1,
        'group': 'self-supervised\nCNNs',
    },
    'natural shapes, supervised': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat-untex-weak_xform'
                '-cont',
        'color': cols[2],#TAB20C[8],
        'xpos': 2,
        'group': 'supervised\nCNNs',
    },
    'natural shapes, self-supervised': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat-untex-weak_task'
                '-cont/transfer_unocc',
        'color': cols[2],#TAB20C[8],
        'xpos': 2,
        'group': 'self-supervised\nCNNs',
    },
    'natural shapes and textures, supervised': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat-weak_xform-cont',
        'color': cols[3],#TAB20B[4],
        'xpos': 3,
        'group': 'supervised\nCNNs',
    },
    'natural shapes and textures, self-supervised': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat-weak_task-cont/'
                'transfer_unocc',
        'color': cols[3],#TAB20B[4],
        'xpos': 3,
        'group': 'self-supervised\nCNNs',
    },
}
