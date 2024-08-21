'''
This scripts tests the accuracy of CNNs on classifying the exact images presented in the human behavioral experiment.
'''

import os
import os.path as op
import glob
import sys
import time
import itertools

sys.path.append(op.expanduser('~/david/master_scripts/image'))
from image_processing import tile
from config import PROJ_DIR

def collate_RSA_plots(overwrite=False):

    os.chdir(f'{PROJ_DIR}/in_vivo/fMRI')  # tells python which experient
    outdir = 'figures/RSA'
    os.makedirs(outdir, exist_ok=True)

    for derdir, space, norm, norm_method, similarity in itertools.product(
        ['derivatives', 'derivatives_orig'],
        ['func', 'standard'],
        ['all-conds', 'none', 'occluder', 'unoccluded'],
        ['z-score', 'mean-center'],
        ['pearson']#, 'spearman']
    ):

        norm_dir = f'norm-{norm}'
        if norm != 'none':
            norm_dir += f'_{norm_method}'
        analysis_dir = (f'exp?/{derdir}/RSA/*_space-{space}/'
                        f'{norm_dir}/'
                        f'{similarity}/object_completion')
        condwise = sorted(glob.glob(
            f'{analysis_dir}/cond-wise_sims_all-ROIs_long.png'))
        OCI = sorted(glob.glob(
            f'{analysis_dir}/OCI_norm_group_all-ROIs.png'))
        OCI_ind = sorted(glob.glob(
            f'{analysis_dir}/OCI_norm_ind_all-ROIs.png'))

        image_paths = condwise + OCI + OCI_ind
        print(image_paths)
        if image_paths:
            outpath = f'{outdir}/{derdir}_space-{space}_{norm}_' \
                      f'{norm_method}_{similarity}.png'
            if not op.isfile(outpath) or overwrite:
                tile(image_paths, outpath, num_cols=3, base_gap=0, colgap=0,
                     colgapfreq=0, rowgap=0, rowgapfreq=0, by_col=True)

if __name__ == "__main__":

    collate_RSA_plots(overwrite=True)
