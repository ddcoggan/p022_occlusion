# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
cols = [matplotlib.cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 3)]

models = {

    # supervised ResNet101
    'ResNet101, unoccluded, classification': {
        'path': 'resnet101/xform-cont',
        'color': 'k',#cols[0],#TAB20C[0],
        'xpos': 0,
        'group': 'ResNet101\nclassification',
    },
    #'ResNet101, artificial shapes, classification': {
    #    'path': 'resnet101/occ-art-strong_xform-cont',
    #    'color': cols[0],#TAB20C[4],
    #    'xpos': 1,
    #    'group': 'ResNet101\nclassification',
    #},
    'ResNet101, artificial shapes 2, classification': {
        'path': 'resnet101/occ-art2-strong_xform-cont',
        'color': cols[0],#TAB20C[4],
        'xpos': 1,
        'group': 'ResNet101\nclassification',
    },
    'ResNet101, natural shapes, classification': {
        'path': 'resnet101/occ-nat-untex-strong_xform-cont',
        'color': cols[1],#TAB20C[8],
        'xpos': 2,
        'group': 'ResNet101\nclassification',
    },
    'ResNet101, natural shapes and textures, classification': {
        'path': 'resnet101/occ-nat-strong_xform-cont',
        'color': cols[2],#TAB20B[4],
        'xpos': 3,
        'group': 'ResNet101\nclassification',
    },

    # supervised CORnet-S
    'CORnet-S, unoccluded, classification': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_xform-cont',
        'color': 'k',#cols[0],#TAB20C[0],
        'xpos': 0,
        'group': 'CORnet-S+\nclassification',
    },
    #'CORnet-S, artificial shapes, classification': {
    #    'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-art-weak_xform-cont',
    #    'color': cols[0],#TAB20C[4],
    #    'xpos': 1,
    #    'group': 'CORnet-S+\nclassification',
    #},
    'CORnet-S, artificial shapes 2, classification': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-art2-weak_xform-cont',
        'color': cols[0],#TAB20C[4],
        'xpos': 1,
        'group': 'CORnet-S+\nclassification',
    },
    'CORnet-S, natural shapes, classification': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat-untex-weak_xform'
                '-cont',
        'color': cols[1],#TAB20C[8],
        'xpos': 2,
        'group': 'CORnet-S+\nclassification',
    },
    'CORnet-S, natural shapes and textures, classification': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat-weak_xform-cont',
        'color': cols[2],#TAB20B[4],
        'xpos': 3,
        'group': 'CORnet-S+\nclassification',
    },

    # self-supervised CORnet-S
    'CORnet-S, unoccluded, SimCLR': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_task-cont/transfer_unocc',
        'color': 'k',#cols[0],#TAB20C[0],
        'xpos': 0,
        'group': 'CORnet-S+\nSimCLR',
    },
    #'CORnet-S, artificial shapes, SimCLR': {
    #    'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-art-weak_task-cont'
    #            '/transfer_unocc',
    #    'color': cols[0],#TAB20C[4],
    #    'xpos': 1,
    #    'group': 'CORnet-S+\nSimCLR',
    #},
    'CORnet-S, artificial shapes 2, SimCLR': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-art2-weak_task-cont'
                '/transfer_unocc',
        'color': cols[0],#TAB20C[4],
        'xpos': 1,
        'group': 'CORnet-S+\nSimCLR',
    },
    'CORnet-S, natural shapes, SimCLR': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat-untex-weak_task'
                '-cont/transfer_unocc',
        'color': cols[1],#TAB20C[8],
        'xpos': 2,
        'group': 'CORnet-S+\nSimCLR',
    },
    'CORnet-S, natural shapes and textures, SimCLR': {
        'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_occ-nat-weak_task-cont/'
                'transfer_unocc',
        'color': cols[2],#TAB20B[4],
        'xpos': 3,
        'group': 'CORnet-S+\nSimCLR',
    },
}
"""
    # Pix2Pix
    'Pix2Pix, artificial shapes, reconstruction': {
        'path': 'pix2pix/occ-art',
        'color': cols[0],#TAB20C[4],
        'xpos': 0,
        'group': 'Pix2Pix\nreconstruction',
    },
    'Pix2Pix, artificial shapes 2, reconstruction': {
        'path': 'pix2pix/occ-art2',
        'color': cols[0],#TAB20C[4],
        'xpos': 1,
        'group': 'Pix2Pix\nreconstruction',
    },
    'Pix2Pix, natural shapes, reconstruction': {
        'path': 'pix2pix/occ-nat-untex',
        'color': cols[1],#TAB20C[8],
        'xpos': 2,
        'group': 'Pix2Pix\nreconstruction',
    },
    'Pix2Pix, natural shapes and textures, reconstruction': {
        'path': 'pix2pix/occ-nat',
        'color': cols[2],#TAB20C[8],
        'xpos': 3,
        'group': 'Pix2Pix\nreconstruction',
    },
"""
