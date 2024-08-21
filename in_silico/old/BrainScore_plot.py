

import sys
import os
import os.path as op
import glob
import functools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime
import numpy as np
from argparse import Namespace

sys.path.append(f'{os.path.expanduser("~")}/david/masterScripts/DNN')
sys.path.append(f'{os.path.expanduser("~")}/david/masterScripts')

from misc.plot_utils import export_legend, custom_defaults
plt.rcParams.update(custom_defaults)

occluders_train = ['unaltered','barHorz08_vis-50', 'behaviouralOccs_vis-mixedVis']
test_config = {'V1': ['movshon.FreemanZiemba2013public.V1-pls'],
              'V2': ['movshon.FreemanZiemba2013public.V2-pls'],
              'V4': ['dicarlo.MajajHong2015public.V4-pls'],
              'IT': ['dicarlo.MajajHong2015public.IT-pls']}
layers = list(test_config.keys())
figsize=(5,6)


outdir = 'in_silico/analysis/results/BrainScore'
scores_path = f'{outdir}/scores.csv'

scores = pd.read_csv(open(scores_path, 'r+'), index_col=0)

