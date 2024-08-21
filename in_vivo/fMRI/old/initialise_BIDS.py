# /usr/bin/python
# Created by David Coggan on 2022 11 02
"""
prepares directory structure and extracts/copies raw data from sourcedata
For future projects, try to use dcm2bids (https://andysbrainbook.readthedocs.io/en/latest/OpenScience/OS/BIDS_Overview.html)
"""

import os
import os.path as op
import sys
import glob
import shutil
import json
import time
sys.path.append(op.expanduser('~/david/masterScripts/misc'))
from seconds_to_text import seconds_to_text

def initialiseBIDS():
    
    # make link to fMRI custom scripts dir
    if not op.exists(f"scripts/utils"):
        os.system(f"ln -s $HOME/david/masterScripts/fMRI scripts/utils")
    sys.path.append("scripts")
    
    from utils import philips_slice_timing
    from utils import make_anat_slices
    
    print(f"Initializing BIDS...")
    subjects = json.load(open("participants.json", "r+"))
    
    for subject in subjects:
    
        for s, session in enumerate(subjects[subject]):

            print(f'{subject} {session}')
            filetypes = ["nii","json"] # do not include other filetypes that may cause BIDS errors
            sourcedir = f"sourcedata/sub-{subject}/ses-{s+1}/raw_data"
            sessID = subjects[subject][session]["sessID"]

            # detect DICOM or NIFTI format for raw data
            if len(glob.glob(f"{sourcedir}/*.DCM")): # if DICOM format
                os.system(f"dcm2niix {op.abspath(sourcedir)}") # convert to nifti, json etc
                copy_or_move = shutil.move # move files, don't copy
            else: # if NIFTI format
                copy_or_move = shutil.copy # copy files, don't move


            ### ANAT ###

            anatscan = subjects[subject][session]["anat"]
            if anatscan is not None:

                anat_ses = s+1
                anatdir = f"sub-{subject}/ses-{anat_ses}/anat"
                os.makedirs(anatdir, exist_ok=True)

                # json file
                files = glob.glob(f"{sourcedir}/*{sessID}.{anatscan:02}*.json")
                assert len(files) == 1
                inpath = files[0]
                outpath = f"{anatdir}/sub-{subject}_ses-{s+1}_T1w.json"
                if not op.isfile(outpath):
                    copy_or_move(inpath, outpath)

                # nii file
                files = glob.glob(f"{sourcedir}/*{sessID}.{anatscan:02}*.nii")
                assert len(files) == 1
                inpath = files[0]

            else:

                anat_ses = 'anat'
                anatdir = f"sub-{subject}/ses-{anat_ses}/anat"
                os.makedirs(anatdir, exist_ok=True)

                # if no anatomical, use most recent anatomical

                # anatomical manually found for this subject
                if subject == 'F013':
                    inpath = 'sourcedata/sub-F013/ses-anat/Tong_341248.06.01.09-32-36.WIP_T1_LGN_SENSE.01.json'
                else:
                    proj_dirs = sorted(['p022_occlusion/in_vivo/fMRI/exp2'])
                    inpath = None
                    proj_counter = 0
                    while not inpath:
                        files = sorted(glob.glob(op.expanduser(f'~/david/projects/{proj_dirs[proj_counter]}/sub-{subject}/ses-*/anat/sub-{subject}_ses-*_T1w.json')))
                        if files:
                            inpath = files[-1]
                        proj_counter += 1

                outpath = f"{anatdir}/sub-{subject}_ses-anat_T1w.json"
                if not op.isfile(outpath):
                    shutil.copy(inpath, outpath)

                # nii file
                inpath = f'{inpath[:-5]}.nii'


            # deidentify anatomical image
            outpath = f"{anatdir}/sub-{subject}_ses-{anat_ses}_T1w.nii"
            if not op.isfile(outpath):
                os.system(f'mideface --i {inpath} --o {outpath}')

            # make T1 images for subject
            if not op.isdir(op.expanduser(f'~/david/subjects/for_subjects/{subject}/2D')):
                make_anat_slices(subject, inpath)


            ### FUNC ###

            funcdir = f"sub-{subject}/ses-{s + 1}/func"
            os.makedirs(funcdir, exist_ok=True)
            fmapdir = f"sub-{subject}/ses-{s + 1}/fmap"
            os.makedirs(fmapdir, exist_ok=True)
            topup_counter = 1  # BIDS doesn't like task names in topup files so set a run number that is unique across tasks

            for funcscan in subjects[subject][session]["func"]:
                for run, scan_num in enumerate(subjects[subject][session]["func"][funcscan]):

                    for filetype in filetypes:
                        files = glob.glob(f"{sourcedir}/*{sessID}.{scan_num:02}*.{filetype}")
                        assert len(files) == 1
                        inpath = files[0]
                        outpath = f"{funcdir}/sub-{subject}_ses-{s + 1}_task-{funcscan}_dir-AP_run-{run+1}_bold.{filetype}"
                        if not op.isfile(outpath):
                            copy_or_move(inpath,outpath)

                    # add required meta data to json file
                    scandata = json.load(open(outpath, "r+"))
                    if "TaskName" not in scandata:
                        scandata["TaskName"] = funcscan
                    if "PhaseEncodingDirection" not in scandata:
                        scandata["PhaseEncodingDirection"] = "j-"
                    if "SliceTiming" not in scandata:
                        scandata["SliceTiming"] = philips_slice_timing(outpath)
                    if "TotalReadoutTime" not in scandata:
                        scandata["TotalReadoutTime"] = scandata["EstimatedTotalReadoutTime"]
                    json.dump(scandata, open(outpath, "w+"), sort_keys=True, indent=4)


                    # repeat for top up file (assumes the next scan was the top up scan)

                    # find data
                    nifti_target = f"{outpath[:-5]}.nii" # need this path for the top up json file
                    for filetype in filetypes:
                        files = glob.glob(f"{sourcedir}/*{sessID}.{scan_num+1:02}*.{filetype}")
                        assert len(files) == 1
                        inpath = files[0]
                        outpath = f"{fmapdir}/sub-{subject}_ses-{s + 1}_acq-topup_dir-PA_run-{topup_counter}_epi.{filetype}"
                        if not op.isfile(outpath):
                            copy_or_move(inpath,outpath)
                    topup_counter += 1

                    # add required meta data to json file
                    scandata = json.load(open(outpath, "r+"))
                    if "PhaseEncodingDirection" not in scandata:
                        scandata["PhaseEncodingDirection"] = "j"
                    if "SliceTiming" not in scandata:
                        scandata["SliceTiming"] = philips_slice_timing(outpath)
                    if "TotalReadoutTime" not in scandata:
                        scandata["TotalReadoutTime"] = scandata["EstimatedTotalReadoutTime"]
                    if "IntendedFor" not in scandata:
                        scandata["IntendedFor"] = nifti_target[9:]
                    json.dump(scandata, open(outpath, "w+"), sort_keys=True, indent=4)


            ### FMAP ###

            # b0
            for c, component in enumerate(["magnitude", "fieldmap"]):
                for filetype in filetypes:
                    files = glob.glob(f"{sourcedir}/*{sessID}.{subjects[subject][session]['fmap']['b0']:02}*B0_shimmed*e{c+1}*.{filetype}")
                    assert len(files) == 1
                    inpath = files[0]
                    outpath = f"{fmapdir}/sub-{subject}_ses-{s+1}_acq-b0_{component}.{filetype}"
                    if not op.isfile(outpath):
                        copy_or_move(inpath, outpath)

                # add required meta data to json file
                scandata = json.load(open(outpath, "r+"))
                if "IntendedFor" not in scandata:
                    intendedscans = glob.glob(f"sub-{subject}/ses-{s+1}/func/*.nii")
                    intendedscans += glob.glob(f"sub-{subject}/ses-{s+1}/anat/*.nii")
                    scandata["IntendedFor"] = sorted([x[9:] for x in intendedscans])
                if component == "fieldmap" and "Units" not in scandata:
                    scandata["Units"] = "Hz"
                json.dump(scandata, open(outpath, "w+"), sort_keys=True, indent=4)


            # funcNoEPI
            for filetype in filetypes:
                files = glob.glob(f"{sourcedir}/*{sessID}.{subjects[subject][session]['fmap']['funcNoEPI']:02}*.{filetype}")
                assert len(files) == 1
                inpath = files[0]
                outpath = f"{fmapdir}/sub-{subject}_ses-{s+1}_acq-funcNoEPI_magnitude.{filetype}"
                if not op.isfile(outpath):
                    copy_or_move(inpath, outpath)


if __name__ == "__main__":

    start = time.time()
    os.chdir(op.expanduser("~/david/projects/p022_occlusion/in_vivo/fMRI/exp1_orig"))
    initialiseBIDS()
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')


