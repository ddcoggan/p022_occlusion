import sys
import os
import os.path as op
import glob
import functools
import numpy as np
import argparse
import pickle as pkl
from types import SimpleNamespace
import os.path as op
import sys
import glob

sys.path.append(op.expanduser('~/david/master_scripts'))
from DNN.utils import get_model, load_params


for package in glob.glob(op.expanduser('~/david/repos/BrainScore_allrepos/*')):
    sys.path.append(package)

from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment
from brainscore import score_model

parser = argparse.ArgumentParser()
parser.add_argument('-model_dir', type=str, help='directory for trained model')
parser.add_argument('-layer', type=str, help='model layer to test')
parser.add_argument('-benchmark', type=str, help='brainscore benchmark to use')

"""
def get_score(model, identifier, layer, benchmark):


    preproc = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(
        identifier=identifier,
        model=model,
        preprocessing=preproc)
    model_commit = ModelCommitment(
        identifier=identifier,
        activations_model=activations_model,
        layers=[layer])
    result = score_model(
        model_identifier=identifier,
        model=model_commit,
        benchmark_identifier=benchmark)
    score_ceiled, sem = result.values
    score_unceiled = result.raw.values[0]

    return score_ceiled, score_unceiled, sem

"""

from brainscore_vision.benchmarks import public_benchmark_pool

def get_score(model, identifier, layer, benchmark):
    benchmark = public_benchmark_pool['dicarlo.MajajHong2015public.IT-pls']
    preproc = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(
        identifier=identifier,
        model=model,
        preprocessing=preproc)
    model_commit = ModelCommitment(
        identifier=identifier,
        activations_model=activations_model,
        layers=[layer])
    result = benchmark(model_commit)
    score_ceiled, sem = result.values
    score_unceiled = result.raw.values[0]

    return score_ceiled, score_unceiled, sem

def get_trained_model(model_dir, return_states=False, return_blocks=None,
                      model_name=None):

    model_name = model_dir.split('/')[2] if model_name is None else model_name

    # get kwargs for configurable models
    if model_name == 'cornet_s_custom':
        M = pkl.load(open(f'{model_dir}/config.pkl', 'rb')).M
        M.model_name = model_name
        M.return_states = return_states
        kwargs = {'M': M}
    elif model_name == 'cornet_rt_hw3':
        kwargs = {'return_states': return_states,
                  'return_blocks': return_blocks}
    elif model_name == 'vit_b_16':
        kwargs = {'image_size': 224}
    elif model_name == 'prednet':
        kwargs = {
            'stack_sizes': (3, 48, 96, 192),
            'R_stack_sizes': (3, 48, 96, 192),
            'A_filter_sizes': (3, 3, 3),
            'Ahat_filter_sizes': (3, 3, 3, 3),
            'R_filter_sizes': (3, 3, 3, 3),
            'output_mode': 'prediction',
            'data_format': 'channels_first',
            'return_sequences': True}
    else:
        kwargs = {}

    model = get_model(model_name, **kwargs)

    # load parameters
    params_path = sorted(glob.glob(f"{model_dir}/params/*.pt*"))[-1]
    model = load_params(params_path, model, 'model')

    return model


if __name__ == "__main__":


    args = parser.parse_args()
    if args.model_dir is None:
        args.model_dir = op.expanduser(
            '~/david/projects/p022_occlusion/in_silico/models/resnext101_32x8d'
            '/pretrained')
    if args.layer is None:
        args.layer = 'layer3.6'
    if args.benchmark is None:
        args.benchmark = 'movshon.FreemanZiemba2013public.V1-pls'
    model = get_trained_model(args.model_dir, return_states=False,
                              model_name='resnext101_32x8d')
    get_score(model, args.model_dir, args.layer, args.benchmark)
