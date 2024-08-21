# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for base model plus minor architectural changes
"""

import os.path as op
import glob

# get dirs for contrasting an effect across many models
PROJ_DIR = op.expanduser('~/david/projects/p022_occlusion')
model_contrasts = {}
effect_colors = {}
all_model_dirs = sorted(glob.glob(
    f'{PROJ_DIR}/in_silico/models/*/*/params'))
all_model_dirs = [d.split('models/')[1].split('/params')[0] for d in
                  all_model_dirs]

# effect of occluder training
occluders = ['beh', 'fmri', 'nat']
beh_dirs = [d for d in all_model_dirs if 'occ-beh' in d]
fmri_dirs = [d for d in all_model_dirs if 'occ-fmri' in d]
nat_dirs = [d for d in all_model_dirs if 'occ-nat' in d]
clusters = {}
for occ_dir in beh_dirs:
    model_name, occ_dir_bn, = occ_dir.split('/')[:2]
    occ_dir_info = occ_dir_bn.split('_')
    unocc_dir = (
        f'{model_name}/'
        f'{"_".join([i for i in occ_dir_info if not i.startswith("occ-beh")])}')
    if unocc_dir in all_model_dirs:
        clusters[unocc_dir] = [occ_dir]
        fmri_dir = occ_dir.replace('occ-beh', 'occ-fmri')
        if fmri_dir in fmri_dirs:
            clusters[unocc_dir].append(fmri_dir)
            fmri_dirs.remove(fmri_dir)
        nat_dir = occ_dir.replace('occ-beh', 'occ-nat')
        if nat_dir in nat_dirs:
            clusters[unocc_dir].append(nat_dir)
            nat_dirs.remove(nat_dir)
for occ_dir in fmri_dirs:
    model_name, occ_dir_bn, = occ_dir.split('/')[:2]
    occ_dir_info = occ_dir_bn.split('_')
    unocc_dir = (
        f'{model_name}/'
        f'{"_".join([i for i in occ_dir_info if i != "occ-fmri"])}')
    if unocc_dir in all_model_dirs:
        clusters[unocc_dir] = [occ_dir]
        nat_dir = occ_dir.replace('occ-fmri', 'occ-nat')
        if nat_dir in nat_dirs:
            clusters[unocc_dir].append(nat_dir)
            nat_dirs.remove(nat_dir)
for occ_dir in nat_dirs:
    model_name, occ_dir_bn, = occ_dir.split('/')[:2]
    occ_dir_info = occ_dir_bn.split('_')
    unocc_dir = (
        f'{model_name}/{"_".join([i for i in occ_dir_info if not i.startswith("occ-nat")])}')
    if unocc_dir in all_model_dirs:
        clusters[unocc_dir] = [occ_dir]
clusters['cornet_s_custom/base-model'] = [
    f'cornet_s_custom/occ-beh',
    f'cornet_s_custom/occ-fmri',
    f'cornet_s_custom/occ-nat']

model_contrasts['occlusion_training'] = {}
xpos = 0
for unocc, occs in clusters.items():
    model_contrasts['occlusion_training'][unocc] = {
        'path': unocc,
        'color': 'tab:blue',
        'xpos': xpos}
    xpos += 1
    for occ in occs:
        model_contrasts['occlusion_training'][occ] = {
            'path': occ,
            'color': 'tab:red',
            'xpos': xpos}
        xpos += 1
    xpos += 1

# effect of natural / artificial occluders
model_contrasts['occluder_shape'] = {}
xpos = 0
for unocc, occs in clusters.items():
    if any(['occ-nat' in o for o in occs]) and len(occs) > 1:
        model_contrasts['occluder_shape'][unocc] = {
            'path': unocc, 'color': 'tab:blue', 'xpos': xpos}
        xpos += 1
        for nat in [o for o in occs if 'occ-nat' in o]:
            model_contrasts['occluder_shape'][nat] = {
                'path': nat, 'color': 'tab:green', 'xpos': xpos}
            xpos += 1
        for art in [o for o in occs if 'occ-nat' not in o]:
            model_contrasts['occluder_shape'][art] = {
                'path': art, 'color': 'tab:orange', 'xpos': xpos}
            xpos += 1
        xpos += 1
effect_colors['occluder_shape'] = {
    'without_occlusion': 'tab:blue',
    'natural_occluder_shapes': 'tab:green',
    'artificial_occluder_shapes': 'tab:orange'}

# effect of supervised versus self-supervised training
cont_dirs = [d for d in all_model_dirs if 'task-cont' in d]
model_contrasts['learning_objective'] = {}
xpos = 0
for cont_dir in cont_dirs:
    model_name, occ_dir_bn, = cont_dir.split('/')[:2]
    cont_dir_info = occ_dir_bn.split('_')
    sup_dir = (
        f'{model_name}/'
        f'{"_".join([i for i in cont_dir_info if i != "task-cont"])}')
    if sup_dir in all_model_dirs:
        transfer_cont_dir = [d.split(f'models/')[-1] for d in glob.glob(
            f'{PROJ_DIR}/in_silico/models/{cont_dir}/*transfer*')]
        if len(transfer_cont_dir):
            selected_cont_dir = transfer_cont_dir[0]
            # TODO: make flexible to multiple transfer dirs
        else:
            selected_cont_dir = cont_dir
        model_contrasts['learning_objective'][sup_dir] = {
            'path': sup_dir, 'color': 'tab:blue', 'xpos': xpos}
        xpos += 1
        model_contrasts['learning_objective'][selected_cont_dir] = {
            'path': selected_cont_dir, 'color': 'tab:purple', 'xpos': xpos}
        xpos += 2

effect_colors['learning_objective'] = {
    'supervised': 'tab:blue',
    'self-supervised': 'tab:purple'}
