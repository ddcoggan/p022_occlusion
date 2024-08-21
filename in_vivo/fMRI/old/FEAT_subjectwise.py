#!/usr/bin/python
# Created by David Coggan on 2022 10 11

'''
takes second level design files (combining analyses across runs) made prior to this script, edits and submits to feat
'''

import os
import os.path as op
import sys
import glob
import datetime
import numpy as np
import random
import shutil
import multiprocessing as mp
import subprocess
import math
import json
import time

from utils.seconds_to_text import seconds_to_text

def FEAT_subjectwise(subjects, num_procs=None):
    
    design_paths = []

    for subject in subjects:

        for scan in [op.basename(x) for x in glob.glob(f"derivatives/FEAT/designs/runwise/*")]:

            # get data from template
            design_dir = f"derivatives/FEAT/designs/subjectwise/{scan}"
            template = open(f"{design_dir}/design.fsf", 'r+').read()
            template_outdir = [x for x in template.splitlines() if x.startswith('set fmri(outputdir)')][-1].split('"')[1]
            template_subject = [x for x in template_outdir.split("/") if x.startswith('sub-')][-1][4:]
            num_contrasts = int([x for x in template.splitlines() if x.startswith('set fmri(ncopeinputs)')][-1].split(" ")[-1])
            runwise_dirs = sorted(glob.glob(f"derivatives/FEAT/sub-{subject}/*/task-{scan}/run-*.feat"))
            num_runs = len(runwise_dirs)
            outdir = f"derivatives/FEAT/sub-{subject}/task-{scan}.gfeat"

            # determine if this analysis needs to be (re)performed
            if os.path.isdir(outdir):
                if os.path.isfile(f"{outdir}/cope{num_contrasts}.feat/stats/zstat1.nii.gz"):
                    print(f'{subject} | {scan} | Analysis found and appears complete, skipping...')
                    run_feat = False
                else:
                    print(f'{subject} | {scan} | Incomplete analysis found, deleting and adding to job list...')
                    shutil.rmtree(outdir)
                    run_feat = True
            else:
                print(f'{subject} | {scan} | Analysis not found, adding to job list...')
                run_feat = True

            # design analysis
            if run_feat:
                
                # make copy of template and replace parameters
                design = template
                design = design.replace(template_outdir, op.abspath(outdir))
                design = design.replace(template_subject, subject)

                # clear info about numbers and locations of inputs
                design = design.replace('set feat_files(', '#')  # clears all current inputs
                design = design.replace(f'set fmri(npts)', '#')
                design = design.replace(f'set fmri(multiple)', '#')

                # input info about numbers and locations of inputs
                design += f'\nset fmri(npts) {num_runs}'
                design += f'\nset fmri(multiple) {num_runs}'
                for r, runwise_dir in enumerate(runwise_dirs):
                    design += f'\nset fmri(evg{r + 1}.1) 1.0'
                    design += f'\nset fmri(groupmem.{r + 1}) 1'
                    design += f'\nset feat_files({r + 1}) {op.abspath(runwise_dir)}'

                # write the file out with a unique filename
                design_path = f'{design_dir}/{subject}.fsf'
                with open(design_path, 'w') as file:
                    file.write(design)
                file.close()

                # store FEAT command for parallel processing
                design_paths.append(design_path)

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

if __name__ == "__main__":

    #time.sleep(6 * 60 * 60)
    os.chdir(op.expanduser("~/david/projects/p022_occlusion/in_vivo/fMRI/exp1_orig"))
    subjects = json.load(open("participants.json", "r+"))
    num_procs = 10
    start = time.time()
    FEAT_subjectwise(subjects, num_procs)
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')

