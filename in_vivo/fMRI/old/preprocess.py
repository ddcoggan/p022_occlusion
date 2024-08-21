#!/usr/bin/python
"""
runs all preprocessing
requires data arranged in BIDS format
"""

import os
import os.path as op
import sys
import glob
import json
import pandas as pd
import shutil
import time
import subprocess
sys.path.append(op.expanduser(f"~/david/masterScripts/fMRI"))
sys.path.append(op.expanduser('~/david/masterScripts/misc'))
from seconds_to_text import seconds_to_text


def preprocess(subjects):


    # mriqc (data quality measures)

    version="22.0.6"
    outdir = f"derivatives/mriqc-{version}"
    os.makedirs(outdir, exist_ok=True)

    # individual subjects
    for subject in subjects:
        if not op.isdir(f"{outdir}/sub-{subject}"):
            cmd = f"docker run --rm " \
                  f"--mount type=bind,src={op.abspath('')},dst=/data " \
                  f"--mount type=bind,src={op.abspath('')}/derivatives/mriqc-{version},dst=/out " \
                  f"--memory=32g " \
                  f"--memory-swap=64g " \
                  f"nipreps/mriqc:{version} " \
                  f"--nprocs 8 " \
                  f"--verbose-reports " \
                  f"/data /out participant participant_label {subject}"  # indir, outdir, analysis level
            os.system(cmd)

    # group level
    if not os.path.isfile(f"{outdir}/group_bold.html"):
        cmd = cmd[:-(len(" participant_label XXXX"))]  # remove subject label from command
        cmd = cmd.replace("participant", "group")  # change analysis level from participant to group
        os.system(cmd)


    # fMRIprep (preprocessing for anatomical)

    # These are performed on individual basis as fmriprep does not check which subjects are already processed
    # TODO: allow to run these in parallel if > 16 available cores, with 8 cores per subject
    # https://fmriprep.org/en/stable/faq.html#running-subjects-in-parallel
    indir = ""  # change to a derivative dataset as required, e.g. "derivatives/NORDIC"
    version = "21.0.4"
    outdir = f"derivatives/fmriprep-{version}"
    os.makedirs(outdir, exist_ok=True)
    workdir = f"derivatives/fmriprep_work"
    if op.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    for subject in subjects:
        if not op.isdir(f"{outdir}/sub-{subject}"):
            cmd = f"docker run --rm " \
                  f"--mount type=bind,src={op.abspath(indir)},dst=/data " \
                  f"--mount type=bind,src={op.abspath(op.dirname(outdir))},dst=/out " \
                  f"--mount type=bind,src={os.environ['SUBJECTS_DIR']},dst=/fs_subjects " \
                  f"--mount type=bind,src={op.abspath(workdir)},dst=/work " \
                  f"--memory=64g " \
                  f"--memory-swap=128g " \
                  f"nipreps/fmriprep:{version} " \
                  f"--clean-workdir " \
                  f"--resource-monitor " \
                  f"--nprocs 8 " \
                  f"--mem-mb 64000 " \
                  f"--fs-license-file /fs_subjects/license.txt " \
                  f"--fs-subjects-dir /fs_subjects " \
                  f"--output-spaces func " \
                  f"-w /work " \
                  f"/data /out/{op.basename(outdir)} participant --participant-label {subject}"  # indir, outdir, analysis level
            os.system(cmd)
    shutil.rmtree(workdir)


    # make link to freesurfer subject dir in local project dir
    os.makedirs('derivatives/freesurfer', exist_ok=True)
    fs_link = f'derivatives/freesurfer/sub-{subject}'
    if not op.exists(fs_link):
        os.system(f"ln -s $HOME/david/subjects/freesurfer/sub-{subject} {fs_link}")



    # make T1 3D model for subjects
    for subject in subjects:
        fs_subjs_dir = os.environ["SUBJECTS_DIR"]
        fs_subj_dir = sorted(glob.glob(f"{fs_subjs_dir}/sub-{subject}*"))[-1]
        subject_fs = op.basename(fs_subj_dir)
        if not op.isdir(op.expanduser(f"~/david/subjects/for_subjects/{subject_fs}/3D")):
            os.system(f"python /home/tonglab/david/masterScripts/fMRI/make3Dbrain.py {subject_fs}")


    """
    # preprocess b0 map (topup currently favoured over b0 correction)
    from b0_preprocess import b0_preprocess
    for subject in subjects:
        preproc_dir = f'derivatives/preprocessing/sub-{subject}/b0'
        os.makedirs(preproc_dir, exist_ok=True)
        mag_path = glob.glob(f'sub-{subject}/ses-*/fmap/*b0_magnitude.nii')[0]
        real_path = glob.glob(f'sub-{subject}/ses-*/fmap/*b0_fieldmap.nii')[0]
        b0_preprocess(preproc_dir, mag_path, real_path)
    """

    # apply topup
    from apply_topup import apply_topup
    for subject in subjects:
        topup_scans = sorted(glob.glob(f'sub-{subject}/ses-*/fmap/*topup*epi.nii'))
        for topup_scan in topup_scans:
            topup_json = json.load(open(f'{topup_scan[:-4]}.json', 'r+'))
            func_scan = f'sub-{subject}/{topup_json["IntendedFor"]}'
            scan_info = op.basename(func_scan).split('_')
            ses, task, _, run = scan_info[1:5]
            out_path = f'derivatives/preprocessing/sub-{subject}/{ses}/sub-{subject}_{ses}_{task}_{run}_desc-preproc_bold.nii.gz'
            if not op.isfile(out_path):
                os.makedirs(op.dirname(out_path), exist_ok=True)
                apply_topup(func_scan, topup_scan, 90, 270, out_path)



if __name__ == "__main__":

    os.chdir(op.expanduser("~/david/projects/p022_occlusion/in_vivo/fMRI/exp1_orig"))
    subjects = json.load(open("participants.json", "r+"))
    start = time.time()
    preprocess(subjects)
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')


