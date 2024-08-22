# Created by David Coggan on 2024 08 22

import os
import os.path as op
import numpy as np
import pandas as pd
from PIL import Image
import shutil

"""
Creates occluded and unoccluded versions of the entire stimulus set from 
behavioral exp 1 and places images into a single directory for ease of use.
"""

exp_dir  = op.expanduser('~/david/projects/p022_occlusion/data/in_vivo'
                         '/behavioral/exp1')
trials = pd.read_parquet(op.join(exp_dir, 'analysis/trials.parquet'))
subjects = sorted(trials.subject.unique())
out_dir_orig = op.join(exp_dir, 'images/final')
os.makedirs(out_dir_orig, exist_ok=True)
out_dir_unocc = op.join(exp_dir, 'images/final_unoccluded')
os.makedirs(out_dir_unocc, exist_ok=True)
for t, trial in trials.iterrows():

    # get original stimulus
    old_stim_path = exp_dir + trial.occluded_object_path.split('exp1')[1]
    assert op.isfile(old_stim_path)
    subj = subjects.index(trial.subject)  # 0-indexed
    stimulus_id = f'{t:05}_sub-{subj:02}_trial-{trial.trial - 1:03}'

    # save copy in new directory
    new_stim_path = op.join(out_dir_orig, stimulus_id + '.png')
    if not op.isfile(new_stim_path):
        shutil.copy(old_stim_path, new_stim_path)

    # save unoccluded version in new directory
    new_stim_path = op.join(out_dir_unocc, stimulus_id + '.png')
    if not op.isfile(new_stim_path):
        object_pil = Image.open(trial.object_path).convert('L')
        image_size = 256  # desired size
        old_im_size = object_pil.size
        min_length = min(old_im_size)
        smallest_dim = old_im_size.index(min_length)
        biggest_dim = np.setdiff1d([0, 1], smallest_dim)[0]
        new_max_length = int(
            (image_size / old_im_size[smallest_dim]) * old_im_size[biggest_dim])
        new_shape = [0, 0]
        new_shape[smallest_dim] = image_size
        new_shape[biggest_dim] = new_max_length
        resized_image = object_pil.resize(new_shape)
        left = int((new_shape[0] - image_size) / 2)
        right = new_shape[0] - left
        top = int((new_shape[1] - image_size) / 2)
        bottom = new_shape[1] - top
        processed_object = resized_image.crop((left, top, right, bottom))
        processed_object.save(new_stim_path)

    if t % 100 == 0:
        print(f'processed {t} stimuli')
