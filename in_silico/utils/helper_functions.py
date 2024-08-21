# /usr/bin/python
# Created by David Coggan on 2024 01 04
import pickle as pkl
from types import SimpleNamespace
import os.path as op
import sys
import glob
from datetime import datetime

sys.path.append(op.expanduser('~/david/master_scripts'))
from DNN.utils import get_model, load_params


def now():
    return datetime.now().strftime("%y/%m/%d %H:%M:%S")


def get_trained_model(model_dir, return_states=False, return_blocks=None):
    model_name = model_dir.split('models/')[-1].split('/')[0]

    # get kwargs for configurable models
    if model_name == 'cornet_s_custom':
        with open(f'{model_dir}/config.pkl', 'rb') as f:
            M = pkl.load(f).M
        M.model_name = model_name
        M.return_states = return_states
        kwargs = {'M': M}
    elif model_name in ['cornet_rt_hw3', 'cognet_v9', 'cognet_v10']:
        kwargs = {'return_states': return_states,
                  'return_blocks': return_blocks}
    elif model_name == 'vit_b_16':
        kwargs = {'image_size': 224}
    else:
        kwargs = {}

    model = get_model(model_name, kwargs)

    # load parameters
    params_path = sorted(glob.glob(f"{model_dir}/params/*.pt*"))[-1]
    model = load_params(params_path, model, 'model')

    return model


def reorg_dict(activations):
    new_dict = dict()
    for layer, activ in activations.items():
        if type(activ) is dict:
            for cycle, tensor in activ.items():
                new_dict[f'{layer}_{cycle}'] = tensor
        else:
            new_dict[layer] = activ

    return new_dict


distinct_colors_255 = {
    'red': (230, 25, 75),
    'green': (60, 180, 75),
    'yellow': (255, 225, 25),
    'blue': (0, 130, 200),
    'orange': (245, 130, 48),
    'purple': (145, 30, 180),
    'cyan': (70, 240, 240),
    'magenta': (240, 50, 230),
    'lime': (210, 245, 60),
    'pink': (250, 190, 212),
    'teal': (0, 128, 128),
    'lavender': (220, 190, 255),
    'brown': (170, 110, 40),
    'beige': (255, 250, 200),
    'maroon': (128, 0, 0),
    'mint': (170, 255, 195),
    'olive': (128, 128, 0),
    'apricot': (255, 215, 180),
    'navy': (0, 0, 128),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (127, 127, 127)}

distinct_colors = {k: tuple([x / 255. for x in v])
                   for k, v in distinct_colors_255.items()}
