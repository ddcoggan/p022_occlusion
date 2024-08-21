#!/usr/bin/python

import os
import os.path as op
import sys
import json
import time
sys.path.append(op.expanduser(f"~/david/master_scripts/fMRI"))
sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from seconds_to_text import seconds_to_text
from in_vivo.fMRI.utils import registration_and_ROI_masks

os.chdir(op.expanduser("~/david/projects/p022_occlusion/in_vivo/fMRI/exp1_orig"))
subjects = json.load(open("participants.json", "r+"))
start = time.time()
registration_and_ROI_masks(subjects, 'FSL')
finish = time.time()
print(f'analysis took {seconds_to_text(finish - start)} to complete')


