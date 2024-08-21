# /usr/bin/python
# Created by David Coggan on 2023 06 23
import os
import os.path as op
import json
import itertools
import glob
import shutil
from in_vivo.fMRI.scripts.utils.config import CFG


def get_ROIs(overwrite=False, derdir='derivatives'):

    subjects = json.load(open("participants.json", "r+"))
    os.makedirs(f"{derdir}/masks/MNI152_2mm", exist_ok=True)

    for subject, (region_label, region) in \
            itertools.product(subjects, CFG.region_mapping.items()):

        print(f'Getting {region_label} ROI for {subject}')

        # create transformation from standard to native func space
        subj_dir = f'{derdir}/FEAT/sub-{subject}'
        session = op.basename(glob.glob(f'{subj_dir}/ses-?')[0])
        feat_dir = glob.glob(f'{subj_dir}/{session}/task-occlusion*/'
                             f'run-1.feat')[0]
        ref_func = f'{feat_dir}/example_func.nii.gz'
        xform = f'{feat_dir}/reg/standard2example_func.mat'
        if not op.isfile(xform) or overwrite:
            xform_inverse = f'{op.dirname(xform)}/example_func2standard.mat'
            os.system(f'convert_xfm -omat {xform} -inverse {xform_inverse}')

        # load roi mask (converting to native func space if necessary)
        roi_path_orig = glob.glob(f'{CFG.mask_dir_std}/*/{region}.nii.gz')
        assert len(roi_path_orig) == 1, f'found {len(roi_path_orig)} masks ' \
                                        f'for {region}'
        roi_path_std = f'{derdir}/masks/MNI152_2mm/{region_label}.nii.gz'
        if not op.isfile(roi_path_std) or overwrite:
            try:
                shutil.copy(roi_path_orig[0], roi_path_std)
            except:
                pass
        roi_path_func = f'{derdir}/masks/sub-{subject}/func_space/' \
                        f'{region_label}.nii.gz'
        if not op.isfile(roi_path_func) or overwrite:
            os.makedirs(op.dirname(roi_path_func), exist_ok=True)
            os.system(f'flirt -in {roi_path_std} -ref {ref_func} -applyxfm '
                      f'-init {xform} -out {roi_path_func} -interp '
                      f'nearestneighbour')

        # ROI plots
        plot_dir = f'{derdir}/masks/plots'
        os.makedirs(plot_dir, exist_ok=True)
        ref_range = os.popen(f'fslstats {ref_func} -R')
        ref_max = float(ref_range.read().split()[1])
        coords = os.popen(f'fslstats {roi_path_func} -C').read()[:-2].split(' ')
        coords = [int(float(c)) for c in coords]
        plot_file = f'{plot_dir}/sub-{subject}_{region_label}.png'
        if not op.isfile(plot_file) or overwrite:
            cmd = f'fsleyes render --outfile {plot_file} --size 3200 600 ' \
                  f'--scene ortho --autoDisplay -vl {coords[0]} {coords[1]} ' \
                  f'{coords[2]} {ref_func} -dr 0 {ref_max} {roi_path_func} ' \
                  f'-dr 0 1 -cm greyscale'
            os.system(cmd)

        # whole brain mask
        brain_mask_std = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
        mask_link = f'{derdir}/masks/MNI152_2mm/brain.nii.gz'
        if not op.exists(mask_link):
            os.system(f'ln -s {brain_mask_std} {mask_link}')
        brain_mask_func = f'{feat_dir}/mask.nii.gz'
        mask_link = f'{derdir}/masks/sub-{subject}/func_space/brain.nii.gz'
        if not op.exists(mask_link):
            os.system(f'ln -s {brain_mask_func} {mask_link}')

        # make links to reference images
        local_func = f'{op.dirname(roi_path_func)}/ref_func.nii.gz'
        if not op.exists(local_func):
            os.system(f'ln -s {op.abspath(ref_func)} {local_func}')

