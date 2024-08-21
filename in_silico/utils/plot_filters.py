'''
This scripts tests the accuracy of CNNs on classifying the exact images presented in the human behavioral experiment.
'''

import os
import os.path as op
import glob
import sys

from .helper_functions import get_trained_model

sys.path.append(op.expanduser('~/david/master_scripts/DNN/utils'))
from plot_conv_filters import plot_conv_filters


def plot_filters(model_dirs, overwrite=False):

    for model_dir in model_dirs:

        # model parameters
        # model = get_trained_model(model_dir)
        # torch.nn.Sequential(*list(model.children())[:-1])
        # params = model.children()[0].state_dict()
        params_path = sorted(glob.glob(f"{model_dir}/params/*.pt*"))[-1]
        params_name = op.basename(params_path).split('.pt')[0]

        # output directory
        outdir = f'{model_dir}/kernel_plots'
        os.makedirs(outdir, exist_ok=True)

        if 'cognet' in model_dir:
            layer = 'V1.conv.weight'
        else:
            layer = 0
        #skip = max([x in model_dir for x in ['vit']])
        #if not skip:
        outpath = f'{outdir}/{params_name}_{layer}.png'
        if 'vit' not in model_dir:
             if not op.isfile(outpath) or overwrite:
                plot_conv_filters(layer, params_path, outpath)


if __name__ == "__main__":

    import time
    from .model_contrasts import model_dirs
    sys.path.append(op.expanduser('~/david/master_scripts/misc'))
    from seconds_to_text import seconds_to_text

    start = time.time()
    plot_filters(model_dirs, overwrite=False)
    finish = time.time()

    print(f'analysis took {seconds_to_text(finish - start)} to complete')
