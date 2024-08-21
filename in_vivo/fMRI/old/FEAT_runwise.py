#!/usr/bin/python
'''
takes first level design files (each run) made prior to this script, edits and submits to feat
'''

import os
import os.path as op
import sys
import glob
import datetime
import json
import shutil
import multiprocessing as mp
import math
import subprocess
import time
#time.sleep(15000)
from utils.seconds_to_text import seconds_to_text


def FEAT_runwise(num_procs=None):

    design_paths = []

    for scan in [op.basename(x) for x in glob.glob(f"{os.getcwd()}/derivatives/FEAT/designs/runwise/*")]:

        # get data from template
        design_dir = f"{os.getcwd()}/derivatives/FEAT/designs/runwise/{scan}"
        template = open(f"{design_dir}/design.fsf", 'r+').read()
        template_outdir = [x for x in template.splitlines() if x.startswith('set fmri(outputdir)')][-1].split('"')[1]
        template_subject = [x for x in template_outdir.split("/") if x.startswith('sub-')][-1][4:]
        template_session = [x for x in template_outdir.split("/") if x.startswith('ses-')][-1]
        template_run_num = [x for x in template_outdir.split("/") if x.startswith('run-')][-1][:-5]
        num_contrasts = int([x for x in template.splitlines() if x.startswith('set fmri(ncon_orig)')][-1].split(" ")[-1])

        # find all runs of this scan
        run_paths = sorted(glob.glob(f"{os.getcwd()}/derivatives/preprocessing/**/*/*{scan}*_desc-preproc_bold.nii.gz", recursive=True))

        for run_path in run_paths:

            # parameters for this analysis
            subject = [x for x in op.basename(run_path).split("_") if x.startswith('sub-')][-1][4:]
            session = [x for x in op.basename(run_path).split("_") if x.startswith('ses-')][-1]
            run_num = [x for x in op.basename(run_path).split("_") if x.startswith('run-')][-1]

            # determine which analyses need to be (re)performed
            outdir = f"{os.getcwd()}/derivatives/FEAT/sub-{subject}/{session}/task-{scan}/{run_num}.feat"
            if os.path.isdir(outdir):
                if os.path.isfile(f"{outdir}/stats/zstat{num_contrasts}.nii.gz"):
                    print(f'{subject} | {session} | {scan} | {run_num} | Analysis found and appears complete, skipping...')
                    run_feat = False
                else:
                    print(f'{subject} | {session} | {scan} | {run_num} | Incomplete analysis found, deleting and adding to job list...')
                    shutil.rmtree(outdir)
                    run_feat = True
            else:
                print(f'{subject} | {session} | {scan} | {run_num} | Analysis not found, adding to job list...')
                run_feat = True

            # design analysis
            if run_feat:

                # make copy of template and replace parameters
                design = template
                design = design.replace('set fmri(outputdir)', '#')
                design = design.replace(template_subject, subject)
                design = design.replace(template_session, session)
                design = design.replace(template_run_num, run_num)
                design += f'\nset fmri(outputdir) {op.abspath(outdir)}'

                # write the file out with a unique filename
                design_path = f'{design_dir}/{subject}_{session}_{run_num}.fsf'
                with open(design_path, 'w') as file:
                    file.write(design)
                file.close()

                # store FEAT command for parallel processing
                design_paths.append(f'{design_path}')

    # run jobs in parallel
    if num_procs is None:
        num_procs = mp.cpu_count() - 4  # spare some cpus for other processes by default
    print(f'Total of {len(design_paths)} jobs, processing in batches of {num_procs}')
    for batch in range(math.ceil(len(design_paths) / num_procs)):
        processes = []
        first_job = batch * num_procs
        last_job = min(((batch + 1) * num_procs), len(design_paths))
        jobs = design_paths[first_job:last_job]
        for job in jobs:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {job}')
            p = subprocess.Popen(['feat', job])
            processes.append(p)
        for p in processes:
            p.communicate()

    # collate registrations for inspection
    all_reg_images = sorted(glob.glob(f'derivatives/FEAT/**/*.feat/reg/example_func2standard.png', recursive=True))
    reg_dir = 'derivatives/FEAT/runwise_reg_images'
    os.makedirs(reg_dir, exist_ok=True)
    for image in all_reg_images:
        dir_info = image.split('/')
        subject, session, task = dir_info[2:5]
        run = dir_info[5].split('.')[0]
        out_path = f'{reg_dir}/{subject}_{session}_{task}_{run}.png'
        if not op.isfile(out_path):
            shutil.copy(image, out_path)



if __name__ == "__main__":

    os.chdir(op.expanduser("~/david/projects/p022_occlusion/in_vivo/fMRI/exp1_orig"))
    num_procs = 10
    start = time.time()
    FEAT_runwise(num_procs)
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')
    sys.exit()
