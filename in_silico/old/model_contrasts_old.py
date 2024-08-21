"""
Stores all model comparison sets in one place
These are then used across various activation and accuracy analysis scripts
"""

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

model_contrasts = {}
tabcols = list(mcolors.TABLEAU_COLORS.keys())



# analysis of variable independently


# use fMRI activations to get list of all models
data = pd.read_csv(f"in_silico/analysis/results/unit_activations/fMRI_stim/RSA.csv", index_col=0)

defaults = {'model_name': ['cornet_s_custom'],
            'recurrent_cycles': ['R-1-2-4-2'],
            'kernel_size': ['K-3-3-3-3'],
            'feature_maps': ['F-0064-0128-0256-0512'],
            'occluder_train': ['unaltered'],
            'learning': ['supervised_classification'],
            'identifier': ['instance1'],
            'norm': ['none']}

for v, variable in enumerate(defaults):
    
    model_contrasts[variable] = {'model_configs': [],
                          'model_labels': []}

    # config to make subset of models with default settings except for this variable
    config = defaults.copy()
    config[variable] = data[variable].unique() # do not filter out any of this variable
    
    # all models in 'learning' comparison trained with behavioral occluders
    if variable == 'learning':
        config['occluder_train'] = ['behaviouralOccs_vis-mixedVis']

    # make subset
    df = data[
        (data['model_name'].isin(config['model_name'])) &
        (data['recurrent_cycles'].isin(config['recurrent_cycles'])) &
        (data['kernel_size'].isin(config['kernel_size'])) &
        (data['feature_maps'].isin(config['feature_maps'])) &
        (data['occluder_train'].isin(config['occluder_train'])) &
        (data['learning'].isin(config['learning'])) &
        (data['identifier'].isin(config['identifier'])) &
        (data['norm'].isin(config['norm']))
        ].copy()
    
    # get final set of levels for the variable of interest
    levels = df[variable].unique()

    # set plot colours
    base_colour = tabcols[v]
    colours, alphas = [], []
    alpha_range = np.linspace(0, 1, len(levels))
    alpha_counter = 1
    
    # loop through models, store parameters
    for l, level in enumerate(levels):

        # store model config, removing list wrapper
        this_config = config.copy()
        for key, item in this_config.items():
            this_config[key] = item[0]

        # change the current level
        this_config[variable] = level

        # save model config and label
        model_contrasts[variable]['model_configs'].append(this_config)
        model_contrasts[variable]['model_labels'].append(level)
        
        # get plot colours and alphas 
        # base model is coloured black, except for 'learning' variable, 
        # which is red, this makes it consistent with the 'occluder_train' plots
        if level == defaults[variable][0]:
            if variable == 'learning':
                colours.append(list(mcolors.TABLEAU_COLORS.keys())[3])
            else:
                colours.append('black')
            alphas.append(1)
        else:
            colours.append(base_colour)
            alphas.append(alpha_range[alpha_counter])
            alpha_counter += 1

    model_contrasts[variable]['colours'] = colours
    model_contrasts[variable]['alphas'] = alphas


# custom model_contrasts 

# custom analysis 1: various classificiaton and contrastive models
model_contrasts['classification_v_contrastive'] = {'model_configs': [],
                                                   'model_labels': [],
                                                   'colours': tabcols,
                                                   'alphas': [1] * len(tabcols)}
model_contrasts['classification_v_contrastive']['model_labels'].append('classification unaltered')
model_contrasts['classification_v_contrastive']['model_configs'].append({'model_name': 'cornet_s_custom',
        'recurrent_cycles': 'R-1-2-4-2',
        'kernel_size': 'K-3-3-3-3',
        'feature_maps': 'F-0064-0128-0256-0512',
        'scaling_factor': 4,
        'dataset': 'ILSVRC2012',
        'occluder_train': 'unaltered',
        'learning': 'supervised_classification',
        'identifier': 'instance1',
        'norm': 'none'})
model_contrasts['classification_v_contrastive']['model_labels'].append('classification occluded')
model_contrasts['classification_v_contrastive']['model_configs'].append({
        'model_name': 'cornet_s_custom',
        'recurrent_cycles': 'R-1-2-4-2',
        'kernel_size': 'K-3-3-3-3',
        'feature_maps': 'F-0064-0128-0256-0512',
        'scaling_factor': 4,
        'dataset': 'ILSVRC2012',
        'occluder_train': 'behaviouralOccs_vis-mixedVis',
        'learning': 'supervised_classification',
        'identifier': 'instance1',
        'norm': 'none'})
model_contrasts['classification_v_contrastive']['model_labels'].append('classification_and_contrastive_shifted_occluder')
model_contrasts['classification_v_contrastive']['model_configs'].append({
    'model_name': 'cornet_s_custom',
    'recurrent_cycles': 'R-1-2-4-2',
    'kernel_size': 'K-3-3-3-3',
    'feature_maps': 'F-0064-0128-0256-0512',
    'scaling_factor': 4,
    'dataset': 'ILSVRC2012',
    'occluder_train': 'behaviouralOccs_vis-mixedVis',
    'learning': 'supervised_classification_and_unsupervised_contrastive',
    'identifier': 'occluder_translate',
    'norm': 'none'})
model_contrasts['classification_v_contrastive']['model_labels'].append('contrastive_shifted_occluder_retrained_on_classification')
model_contrasts['classification_v_contrastive']['model_configs'].append({
    'model_name': 'cornet_s_custom',
    'recurrent_cycles': 'R-1-2-4-2',
    'kernel_size': 'K-3-3-3-3',
    'feature_maps': 'F-0064-0128-0256-0512',
    'scaling_factor': 4,
    'dataset': 'ILSVRC2012',
    'occluder_train': 'behaviouralOccs_vis-mixedVis',
    'learning': 'supervised_classification',
    'identifier': 'retrained_from_contrastive_occluder_translate',
    'norm': 'none'
})
model_contrasts['classification_v_contrastive']['model_labels'].append('contrastive_random_occluder')
model_contrasts['classification_v_contrastive']['model_configs'].append({
    'model_name': 'cornet_s_custom',
    'recurrent_cycles': 'R-1-2-4-2',
    'kernel_size': 'K-3-3-3-3',
    'feature_maps': 'F-0064-0128-0256-0512',
    'scaling_factor': 4,
    'dataset': 'ILSVRC2012',
    'occluder_train': 'behaviouralOccs_vis-mixedVis',
    'learning': 'unsupervised_contrastive',
    'identifier': 'instance1',
    'norm': 'none'
})
model_contrasts['classification_v_contrastive']['model_labels'].append('contrastive_shifted_occluder')
model_contrasts['classification_v_contrastive']['model_configs'].append({
    'model_name': 'cornet_s_custom',
    'recurrent_cycles': 'R-1-2-4-2',
    'kernel_size': 'K-3-3-3-3',
    'feature_maps': 'F-0064-0128-0256-0512',
    'scaling_factor': 4,
    'dataset': 'ILSVRC2012',
    'occluder_train': 'behaviouralOccs_vis-mixedVis',
    'learning': 'unsupervised_contrastive',
    'identifier': 'occluder_translate',
    'norm': 'none'
})
model_contrasts['classification_v_contrastive']['model_labels'].append('contrastive_shifted_occluder_no_recurrence')
model_contrasts['classification_v_contrastive']['model_configs'].append({
    'model_name': 'cornet_s_custom',
    'recurrent_cycles': 'R-1-1-1-1',
    'kernel_size': 'K-3-3-3-3',
    'feature_maps': 'F-0064-0128-0256-0512',
    'scaling_factor': 4,
    'dataset': 'ILSVRC2012',
    'occluder_train': 'behaviouralOccs_vis-mixedVis',
    'learning': 'unsupervised_contrastive',
    'identifier': 'occluder_translate',
    'norm': 'none'
})


# custom analysis 2: sequentially build up plots for CCN presentation
model_contrasts['CCN'] = {'model_configs': [],
                          'model_labels': [],
                          'colours': tabcols,
                          'alphas': [1] * len(tabcols)}
model_contrasts['CCN']['model_labels'].append('classification, unoccluded (base model)')
model_contrasts['CCN']['model_configs'].append({'model_name': 'cornet_s_custom',
        'recurrent_cycles': 'R-1-2-4-2',
        'kernel_size': 'K-3-3-3-3',
        'feature_maps': 'F-0064-0128-0256-0512',
        'scaling_factor': 4,
        'dataset': 'ILSVRC2012',
        'occluder_train': 'unaltered',
        'learning': 'supervised_classification',
        'identifier': 'instance1',
        'norm': 'none'})
model_contrasts['CCN']['model_labels'].append('classification (fMRI occluders)')
model_contrasts['CCN']['model_configs'].append({
        'model_name': 'cornet_s_custom',
        'recurrent_cycles': 'R-1-2-4-2',
        'kernel_size': 'K-3-3-3-3',
        'feature_maps': 'F-0064-0128-0256-0512',
        'scaling_factor': 4,
        'dataset': 'ILSVRC2012',
        'occluder_train': 'barHorz08_vis-50',
        'learning': 'supervised_classification',
        'identifier': 'instance1',
        'norm': 'none'})
model_contrasts['CCN']['model_labels'].append('classification (behav. occluders)')
model_contrasts['CCN']['model_configs'].append({
        'model_name': 'cornet_s_custom',
        'recurrent_cycles': 'R-1-2-4-2',
        'kernel_size': 'K-3-3-3-3',
        'feature_maps': 'F-0064-0128-0256-0512',
        'scaling_factor': 4,
        'dataset': 'ILSVRC2012',
        'occluder_train': 'behaviouralOccs_vis-mixedVis',
        'learning': 'supervised_classification',
        'identifier': 'instance1',
        'norm': 'none'})
model_contrasts['CCN']['model_labels'].append('self-sup. contrastive (random occluder)')
model_contrasts['CCN']['model_configs'].append({
    'model_name': 'cornet_s_custom',
    'recurrent_cycles': 'R-1-2-4-2',
    'kernel_size': 'K-3-3-3-3',
    'feature_maps': 'F-0064-0128-0256-0512',
    'scaling_factor': 4,
    'dataset': 'ILSVRC2012',
    'occluder_train': 'behaviouralOccs_vis-mixedVis',
    'learning': 'unsupervised_contrastive',
    'identifier': 'instance1_transfer',
    'norm': 'none'
})
model_contrasts['CCN']['model_labels'].append('self-sup. contrastive (shifted occluder)')
model_contrasts['CCN']['model_configs'].append({
    'model_name': 'cornet_s_custom',
    'recurrent_cycles': 'R-1-2-4-2',
    'kernel_size': 'K-3-3-3-3',
    'feature_maps': 'F-0064-0128-0256-0512',
    'scaling_factor': 4,
    'dataset': 'ILSVRC2012',
    'occluder_train': 'behaviouralOccs_vis-mixedVis',
    'learning': 'unsupervised_contrastive',
    'identifier': 'occluder_translate_transfer',
    'norm': 'none'
})
'''
model_contrasts['CCN']['model_labels'].append('sup. contrastive (shifted occluder)')
model_contrasts['CCN']['model_configs'].append({
    'model_name': 'cornet_s_custom',
    'recurrent_cycles': 'R-1-2-4-2',
    'kernel_size': 'K-3-3-3-3',
    'feature_maps': 'F-0064-0128-0256-0512',
    'scaling_factor': 4,
    'dataset': 'ILSVRC2012',
    'occluder_train': 'behaviouralOccs_vis-mixedVis',
    'learning': 'supervised_contrastive',
    'identifier': 'occluder_translate_transfer',
    'norm': 'none'})
'''
model_contrasts['CCN']['model_labels'].append('class. and contr. (shifted occluder)')
model_contrasts['CCN']['model_configs'].append({
    'model_name': 'cornet_s_custom',
    'recurrent_cycles': 'R-1-2-4-2',
    'kernel_size': 'K-3-3-3-3',
    'feature_maps': 'F-0064-0128-0256-0512',
    'scaling_factor': 4,
    'dataset': 'ILSVRC2012',
    'occluder_train': 'behaviouralOccs_vis-mixedVis',
    'learning': 'supervised_classification_and_unsupervised_contrastive',
    'identifier': 'occluder_translate',
    'norm': 'none'})