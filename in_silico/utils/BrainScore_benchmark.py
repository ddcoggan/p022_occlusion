import sys
import os
import os.path as op
import glob
import functools
import matplotlib.pyplot as plt
import time
import pandas as pd
import pickle as pkl
import numpy as np
import datetime
from types import SimpleNamespace

dtnow = datetime.datetime.now
nowstr = "%d/%m/%y %H:%M:%S"

sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from seconds_to_text import seconds_to_text
from plot_utils import make_legend, custom_defaults
plt.rcParams.update(custom_defaults)
from .model_contrasts import model_contrasts, region_to_layer
from in_vivo.fMRI.scripts.utils import line_plot

REGIONS = ['V1', 'V2', 'V4', 'IT']

def measure_scores(model_dir, m=0, total_models=0, overwrite=False):

    python_brainscore = op.expanduser(
        '~/david/repos/brain-score/.venv/bin/python')
    get_brainscore = op.abspath(
        'in_silico/analysis/scripts/utils/get_BrainScore.py')

    test_config = {'V1': ['movshon.FreemanZiemba2013public.V1-pls'],
                   'V2': ['movshon.FreemanZiemba2013public.V2-pls'],
                   'V4': ['dicarlo.MajajHong2015public.V4-pls'],
                   'IT': ['dicarlo.MajajHong2015public.IT-pls']}

    model_info = model_dir.split('models/')[-1].split('/')
    model_name, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'

    outdir = f'{model_dir}/BrainScore'
    os.makedirs(outdir, exist_ok=True)

    scores_path = f'{outdir}/scores.csv'
    if not op.isfile(scores_path) or overwrite:

        print(f'{dtnow().strftime(nowstr)} Measuring BrainScore for '
              f'model {m + 1}/{total_models} at '
              f'{model_name}/{identifier}/{transfer_dir})')

        overwrite = True
        scores = pd.DataFrame()

        # delete any cached results
        model_files = glob.glob(op.expanduser(
            f'~/.result_caching/**/*{model_dir}*'), recursive=True)
        for file in model_files:
            if op.isfile(file):
                os.remove(file)

        # set up preprocessing
        for region, benchmarks in test_config.items():

            layer = region_to_layer[model_name][region]

            for benchmark in benchmarks:

                score_ceiled, score_unceiled, sem = os.popen(
                    f'{python_brainscore} {get_brainscore} '
                    f'-model_dir {model_dir} '
                    f'-layer {layer} '
                    f'-benchmark {benchmark}').read().split()

                scores = pd.concat([scores, pd.DataFrame({
                    'region': [region],
                    'layer': [layer],
                    'benchmark': [benchmark],
                    'score_unceiled': [score_unceiled],
                    'score_ceiled': [score_ceiled],
                    'sem': [sem]})]).reset_index(drop=True)
                scores.to_csv(scores_path)

                print(f'{dtnow().strftime(nowstr)} | '
                      f'model: {model_dir} | '
                      f'layer: {layer} | benchmark: {benchmark} | '
                      f'score: {score_ceiled:.3f}({score_unceiled:.3f}) | '
                      f'sem: {sem:.3f}) |\n')

    return overwrite


def compare_models(overwrite=False):
    print('Comparing models (BrainScore)...')

    for model_contrast, config in model_contrasts.items():

        outdir = f'in_silico/analysis/results/BrainScore/{model_contrast}'
        os.makedirs(outdir, exist_ok=True)

        for score_type in ['ceiled', 'unceiled']:

            outpath = f'{outdir}/scores_{score_type}.png'
            if not op.isfile(outpath) or overwrite:

                plot_df = pd.DataFrame()

                for label, info in config.items():
                    path, color = info['path'], info['color']
                    scores = pd.read_csv(
                        f'in_silico/models/{path}/BrainScore/'
                        f'scores.csv', index_col=0)
                    scores['model'] = [label] * len(scores)
                    scores['color'] = [color] * len(scores)
                    plot_df = pd.concat([plot_df, scores])

                line_plot(
                    plot_df, outpath, ylabel=f'score ({score_type})',
                    x_var='region', title='BrainScore',
                    cond_var='model', col_var='color',
                    y_var=f'score_{score_type}', error_var='sem',
                    ylims=(0, .8), yticks=np.arange(0, 1, .2),
                    figsize=(3.5, 4))

        # save legends separately
        make_legend(
            outpath=outpath.replace('.png', '_legend.png'),
            markeredgecolors=[m['color'] for m in config.values()],
            labels=list(config.keys()),
            markers='o',
            colors='white',
            linestyles='solid')


if __name__ == "__main__":
    start = time.time()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPUids}"
    model_search = 'cornet_s?*'  # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] #'alexnet', 'inception_v3', 'cornet_s', 'PredNetImageNet',
    model_dirs = sorted(glob.glob(f'in_silico/models/{model_search}/*'))
    measure_scores(model_dirs, overwrite=True)
    compare_models()
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')
