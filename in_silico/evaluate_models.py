#!/usr/bin/python
"""
controller script for running the analysis pipeline
"""

import os
import os.path as op
import time
import sys
import warnings
from joblib import Parallel, delayed
import itertools
import numpy as np

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_procs = 8

# non-specific custom imports
sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from seconds_to_text import seconds_to_text

if __name__ == "__main__":

    start = time.time()
    #warnings.filterwarnings('error')

    from utils import model_dirs, model_contrasts

    print('Plot convolutional filters...')
    from utils import plot_filters
    plot_filters(model_dirs, overwrite=False)

    print('Tile sample inputs...')
    from utils import tile_sample_inputs
    tile_sample_inputs(model_dirs, overwrite=False)
    
    print('Behavioral benchmark...')
    from utils import make_pca_dataset, make_svc_dataset
    exps = ['exp1', 'exp2']
    make_svc_dataset(overwrite=False)
    make_pca_dataset(overwrite=False)

    # get model responses in serial to avoid memory issues
    overwrite_analyses = np.empty((len(model_dirs), 2), dtype=bool)
    from utils import (train_svc, get_responses, analyse_performance,
                       evaluate_reconstructions)
    for m, model_dir in enumerate(model_dirs):
        overwrite_responses = train_svc(
            model_dir, m, len(model_dirs), num_procs=num_procs,
            overwrite=False)
        for e, exp in enumerate(exps):
            overwrite_analyses[m,e] = get_responses(
                model_dir, m, len(model_dirs), exp,
                overwrite=overwrite_responses,
                num_procs=num_procs)
            if 'pix2pix' in model_dir:
                overwrite_analyses[m, e] = evaluate_reconstructions(
                    model_dir, m, len(model_dirs), exp,
                    overwrite=overwrite_responses,
                    num_procs=num_procs)
            ''' analyze performance in serial '''
            overwrite = analyse_performance(
                model_dir, m, len(model_dirs), exp, remake_plots=True,
                overwrite=overwrite_analyses[m,e])  #'pix2pix' in model_dir)#


    ''' analyze performance in parallel 
    exps_models = itertools.product(enumerate(exps), enumerate(model_dirs))
    overwrite_analyses = Parallel(n_jobs=num_procs)(delayed(
        analyse_performance)(model_dir, m, len(model_dirs), exp, 
        overwrite=overwrite_analyses[m,e], remake_plots=False) for (e, exp), 
        (m, model_dir) in exps_models)
    '''
    from utils import behavioral_plots, compare_models_behav
    behavioral_plots(overwrite=False)
    #recompare_models = overwrite_analyses.max()
    compare_models_behav(overwrite=True)#recompare_models)

    print('pixel attribution...')
    from utils import (pixel_attribution, evaluate_salience,
                       plot_pixel_attribution, compare_models_pixel)
    for m, model_dir in enumerate(model_dirs):
        pixel_attribution(model_dir, m, len(model_dirs), num_procs=num_procs,
                          overwrite=False)
        evaluate_salience(model_dir, m, len(model_dirs), overwrite=False)
        plot_pixel_attribution(model_dir, m, len(model_dirs), overwrite=False)
    compare_models_pixel(overwrite=False)

    print('fMRI benchmark...')
    from utils import (get_model_responses, RSA_fMRI,
         generate_reconstructions, model_dirs_gen, compare_models_fMRI)
    recompare_models = False
    for m, model_dir in enumerate(model_dirs):
        overwrite = False
        overwrite = get_model_responses(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        overwrite = RSA_fMRI(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        if overwrite:
            recompare_models = True
    compare_models_fMRI(overwrite=recompare_models)
    from utils.fMRI_benchmark import get_prednet_responses
    for model_dir in model_dirs_gen:
        get_prednet_responses(model_dir, overwrite=False)
        RSA_fMRI(model_dir, 0, 1, overwrite=False)
        generate_reconstructions(model_dir, overwrite=False)

    """
    Brainscore is in the middle of an upgrade, nothing works right now
    print('BrainScore benchmark...')
    recompare_models = True
    for m, model_dir in enumerate(model_dirs):
        overwrite = False
        overwrite = measure_scores(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        if overwrite:
            recompare_models = True
    compare_models_brainscore(overwrite=recompare_models)
    """

    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')

