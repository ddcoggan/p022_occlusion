# /usr/bin/python
# Created by David Coggan on 2023 06 23

import multiprocessing as mp
import subprocess
import shutil
import os
import os.path as op
import glob
import math
import datetime
import json
import numpy as np
import random
from .config import PROJ_DIR

def FEAT_runwise(n_procs=None, derdir='', input_dir=''):

    design_paths = []

    for scan in [op.basename(x) for x in glob.glob(
            f'{derdir}/FEAT/designs/runwise/*')]:

        # get data from template
        design_dir = f"{derdir}/FEAT/designs/runwise/{scan}"
        template = open(f"{design_dir}/design.fsf", 'r+').read()
        template_outdir = [
            x for x in template.splitlines()
            if x.startswith('set fmri(outputdir)')][-1].split('"')[1]
        template_subject = [x for x in template_outdir.split("/") if
                            x.startswith('sub-')][-1][4:]
        template_session = [x for x in template_outdir.split("/") if
                            x.startswith('ses-')][-1]
        template_run_num = [x for x in template_outdir.split("/") if
                            x.startswith('run-')][-1][:-5]
        n_contrasts = int(
            [x for x in template.splitlines() if x.startswith(
                'set fmri(ncon_orig)')][-1].split(" ")[-1])

        # find all runs of this scan
        run_paths = sorted(glob.glob(f"{derdir}/{input_dir}/**/*/" \
                                     f"*{scan}*_desc-preproc" \
                                     f"_bold.nii.gz", recursive=True))
        assert len(run_paths)

        for run_path in run_paths:

            # parameters for this analysis
            subject = [x for x in op.basename(run_path).split("_") if
                       x.startswith('sub-')][-1][4:]
            session = [x for x in op.basename(run_path).split("_") if
                       x.startswith('ses-')][-1]
            run_num = [x for x in op.basename(run_path).split("_") if
                       x.startswith('run-')][-1]

            # determine which analyses need to be (re)performed
            outdir = f"{op.abspath(derdir)}/FEAT/sub-{subject}/" \
                     f"{session}/task-{scan}/{run_num}.feat"
            if op.isdir(outdir):
                if op.isfile(f"{outdir}/stats/zstat{n_contrasts}.nii.gz"):
                    print(f'{subject} | {session} | {scan} | {run_num} | '
                          f'Analysis found and appears complete, skipping...')
                    run_feat = False
                else:
                    print(f'{subject} | {session} | {scan} | {run_num} | '
                          f'Incomplete analysis found, deleting and adding '
                          f'to job list...')
                    shutil.rmtree(outdir)
                    run_feat = True
            else:
                print(f'{subject} | {session} | {scan} | {run_num} | '
                      f'Analysis not found, adding to job list...')
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
    if n_procs is None:
        n_procs = mp.cpu_count() - 4  # spare some cpus for other processes
    print(f'Total of {len(design_paths)} jobs, '
          f'processing in batches of {n_procs}')
    for batch in range(math.ceil(len(design_paths) / n_procs)):
        processes = []
        first_job = batch * n_procs
        last_job = min(((batch + 1) * n_procs), len(design_paths))
        jobs = design_paths[first_job:last_job]
        for job in jobs:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | '
                  f'{job}')
            p = subprocess.Popen(['feat', job])
            processes.append(p)
        for p in processes:
            p.communicate()



    # collate registrations for inspection
    all_reg_images = sorted(glob.glob(
        f'{derdir}/FEAT/**/*.feat/reg/example_func2standard.png',
        recursive=True))
    reg_dir = f'{derdir}/FEAT/runwise_reg_images'
    os.makedirs(reg_dir, exist_ok=True)
    for image in all_reg_images:
        dir_info = image.split('/')
        subject, session, task = dir_info[2:5]
        run = dir_info[5].split('.')[0]
        out_path = f'{reg_dir}/{subject}_{session}_{task}_{run}.png'
        if not op.isfile(out_path):
            shutil.copy(image, out_path)


def FEAT_subjectwise(n_procs=None, derdir='derivatives'):

    subjects = json.load(open('participants.json', 'r+'))
    design_paths = {}

    for subject in subjects:

        for scan in [op.basename(x) for x in
                     glob.glob(f'{derdir}/FEAT/designs/subjectwise/*')]:

            design_paths[f'{subject}_{scan}'] = []

            # get data from template
            design_dir = f"{derdir}/FEAT/designs/subjectwise/{scan}"
            template = open(f"{design_dir}/design.fsf", 'r+').read()
            template_outdir = [x for x in template.splitlines() if x.startswith(
                'set fmri(outputdir)')][-1].split('"')[
                1]
            template_subject = [x for x in template_outdir.split("/")
                                if x.startswith('sub-')][-1][4:]
            n_contrasts = int(
                [x for x in template.splitlines() if x.startswith(
                    'set fmri(ncopeinputs)')][-1].split(" ")[-1])


            # make sets of runs to analyse
            runs_all = sorted(glob.glob(f"{derdir}/FEAT/sub-{subject}/"
                                            f"*/task-{scan}/run-*.feat"))
            n_runs_all = len(runs_all)
            run_sets = {'all-runs': runs_all}
            if 'occlusion' in scan:
                n_splits = 8
                for split in range(n_splits):
                    runs_A = random.sample(runs_all, n_runs_all // 2)
                    runs_B = [x for x in runs_all if x not in runs_A]
                    run_sets[f'split-{split}A'] = runs_A
                    run_sets[f'split-{split}B'] = runs_B

            for label, runs in run_sets.items():

                n_runs = len(runs)
                outdir = f"{op.abspath(derdir)}/FEAT/sub-{subject}/task" \
                         f"-{scan}_{label}.gfeat"

                # determine if this analysis needs to be (re)performed
                prnt_str = f'{subject} | {scan} | {label} |'
                if os.path.isdir(outdir):
                    if os.path.isfile(f"{outdir}/cope{n_contrasts}.feat/stats/"
                                      f"zstat1.nii.gz"):
                        print(f'{prnt_str} Analysis found and appears '
                              f'complete, skipping...')
                        run_feat = False
                    else:
                        print(f'{prnt_str} Incomplete analysis found, '
                              f'deleting and adding to job list...')
                        shutil.rmtree(outdir)
                        run_feat = True
                else:
                    print(f'{prnt_str} Analysis not found, adding to '
                          f'job list...')
                    run_feat = True

                # design analysis
                if run_feat:

                    # make copy of template and replace parameters
                    design = template
                    design = design.replace(template_outdir, op.abspath(outdir))
                    design = design.replace(template_subject, subject)

                    # clear input info from template
                    for clear_string in ['set feat_files(', 'set fmri(npts)',
                                         'set fmri(multiple)']:
                        design = design.replace(f'{clear_string}', '#')

                    # input new input info
                    design += f'\nset fmri(npts) {n_runs}'
                    design += f'\nset fmri(multiple) {n_runs}'
                    for r, run in enumerate(runs):
                        design += f'\nset fmri(evg{r + 1}.1) 1.0'
                        design += f'\nset fmri(groupmem.{r + 1}) 1'
                        design += f'\nset feat_files({r + 1}) ' \
                                  f'{op.abspath(run)}'

                    # write the file out with a unique filename
                    design_path = f'{design_dir}/{subject}_{label}.fsf'
                    with open(design_path, 'w') as file:
                        file.write(design)
                    file.close()

                    # store FEAT command for parallel processing
                    design_paths[f'{subject}_{scan}'].append(design_path)


    # run jobs in parallel
    if n_procs is None:
        n_procs = mp.cpu_count() - 4  # spare some cpus for other processes
    total_jobs = sum([len(x) for x in design_paths.values()])
    batch_size = min(n_procs, len(design_paths))
    print(f'Total of {total_jobs} jobs, processing in batches of {batch_size}')
    job_hashes = list(design_paths.keys())
    start_idx = 0
    while sum([len(x) for x in design_paths.values()]):
        batch = []

        # ensure jobs from same subject/scan are processed in different batches
        # this gets tricky if some subjects have more scans than others as
        # final batches may be smaller
        job_hashes_ordered = job_hashes[start_idx:] + job_hashes[:start_idx]
        job_idx = 0

        while len(batch) < batch_size and job_idx < len(job_hashes_ordered):
            jobs = design_paths[job_hashes_ordered[job_idx]]
            job_idx += 1
            if len(jobs):
                batch.append(jobs.pop())
                start_idx += 1

        processes = []
        for job in batch:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} '
                  f'| {job}')
            p = subprocess.Popen(['feat', job])
            processes.append(p)
        for p in processes:
            p.communicate()


if __name__ == "__main__":

    os.chdir(f'{PROJ_DIR}/in_vivo/fMRI/exp1')
    n_procs = 10
    for orig, input_folder in zip(['', '_orig'], ['fmriprep-23.0.2',
                                                  'preprocessing']):
        FEAT_runwise(n_procs, f'derivatives{orig}/{input_folder}',
                     f'derivatives{orig}/FEAT')
        FEAT_subjectwise(n_procs, f'derivatives{orig}/FEAT')
