# /usr/bin/python
# Created by David Coggan on 2024 02 07

import os
import os.path as op
import glob
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from eyelinkparser import parse, defaulttraceprocessor
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import pathlib as pl
from scipy.ndimage import zoom
import itertools
import pickle as pkl
from tqdm import tqdm
import sys
from itertools import product as itp
import torch
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from plot_utils import custom_defaults, distinct_colors, make_legend
plt.rcParams.update(custom_defaults)

DATA_DIR = '../data/in_vivo/behavioral/exp2'

# screen
SCRN_RES = (1920, 1080)
SCRN_WIDTH_CM = 53.3
SCRN_DIST_CM = 53.3
PIX_PER_DEG = SCRN_RES[0] / (
        np.rad2deg(np.arctan(SCRN_WIDTH_CM / (2 * SCRN_DIST_CM))) * 2)
BG_COL = (128, 128, 128)
FPS = 60

# stimulus
STIM_LOC = (512, 92)
STIM_SIZE_PIX = (896, 896)
STIM_SIZE_DEG = (p / PIX_PER_DEG for p in STIM_SIZE_PIX)
STIM_DUR = 6
STIM_DIR = op.join(DATA_DIR, 'stimuli/final')
NUM_FRAMES_ORIG = 360
FRAMES_ORIG = np.arange(NUM_FRAMES_ORIG) + 1
NUM_FRAMES = 17
FRAMES = np.linspace(1, NUM_FRAMES_ORIG, NUM_FRAMES).astype(int)
FRAME_BATCHES = [[f for f in FRAMES_ORIG if np.argmin(np.abs(f-FRAMES)) == b]
                 for b in range(NUM_FRAMES)]
VISIBILITIES = np.linspace(0, 1, NUM_FRAMES)
NUM_TRIALS = 1152
STIM_IDS = [f'{s}-{i+1:04}' for s, i in itp(['a', 'b'], range(576))]

# data
SUBJECT_DIRS = sorted(glob.glob('data/sub-*'))
SUBJECTS = [op.basename(d).split('_')[0] for d in SUBJECT_DIRS]
TRIALS_PATH = op.join(DATA_DIR, 'analysis/trials.parquet')
SAMPLE_RATE = 1000
STIM_FRAMES = STIM_DUR * FPS
SAMPLES_PER_FRAME = SAMPLE_RATE / FPS

TAB20 = matplotlib.cm.tab20.colors


# useful variables for importing to computational modeling scripts
class CFG:

    # nested conditions
    objects = {
        'animate': ['bear', 'bison', 'elephant', 'hare'],
        'inanimate': ['jeep', 'lamp', 'car', 'teapot']}
    occluders = {
        'real': ['natural', 'man-made'],
        'noise': ['coarse', 'fine', 'fine-ori', 'pink']}
    textures = {'uniform': ['dark', 'light']}

    # object classes and imagenet properties
    object_classes = ['bear', 'bison', 'elephant', 'hare',
               'jeep', 'lamp', 'car', 'teapot']
    animate_classes = object_classes[:4]
    inanimate_classes = object_classes[4:8]
    classes_orig = [
        'brown bear, bruin, Ursus arctos',
        'bison',
        'African elephant, Loxodonta africana',
        'hare',
        'jeep, landrover',
        'table lamp',
        'sports car, sport car',
        'teapot']
    class_idxs = [294, 347, 386, 331, 609, 846, 817, 849]
    class_dirs = ['n02132136', 'n02410509', 'n02504458', 'n02326432',
                  'n03594945', 'n04380533', 'n04285008', 'n04398044']

    # occluder classes
    occluder_classes = [
        'natural', 'man-made', 'coarse', 'fine', 'fine-ori', 'pink']
    occluder_labels = [
        'natural', 'human-made', 'coarse', 'fine', 'fine (oriented)',
        'pink noise']
    occluder_colors = ['black', 'white']
    occluder_conds = list(itp(occluder_classes, occluder_colors))
    occluder_conds_labels = list(itp(occluder_labels, occluder_colors))
    plot_colors = [TAB20[i] for i in [4, 5, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11]]
    plot_ecolors = [TAB20[i] for i in [5, 4, 3, 2, 1, 0, 7, 6, 9, 8, 11, 10]]
    plot_colors_stimset = plot_colors * 2
    plot_ecolors_stimset = ['k'] * 12 + ['tab:grey'] * 12


def main():

    os.chdir(DATA_DIR)
    process_raw_data()
    analyze_performance(overwrite=True)
    make_fixation_maps(overwrite=False, render_set='subset')
    save_modeling_data(overwrite=False)


def condwise_robustness_plot(df, outpath, metric_column, class_column,
                             color_column, sample_column=None, title=None,
                             acc_only=False, invert_y=True,
                             ylabel=None, legend_path=None):

    ylabel = metric_column if ylabel is None else ylabel
    if acc_only:
        df = df[df.accuracy == 1]

    if sample_column:
        df = (df.groupby(
            [sample_column, class_column, color_column])
            .agg('mean', numeric_only=True).reset_index())
        summary = (df.drop(columns=[sample_column])
                   .groupby([class_column, color_column])
                   .agg({metric_column: ['mean', 'sem']}).reset_index())
        summary.columns = [class_column, color_column, metric_column, 'sem']

    else:
        summary = (df.groupby([class_column, color_column])
                   .agg('mean', numeric_only=True).reset_index())


    fig, ax = plt.subplots(figsize=(3.5, 2))
    xpos = np.arange(len(CFG.occluder_conds), dtype=float)
    xpos[1::2] -= .15  # clustering
    for (x, (occ_class, occ_col), color, ecolor) in zip(
            xpos, CFG.occluder_conds, CFG.plot_colors, CFG.plot_ecolors):
        cond_mean = summary[
            (summary[class_column] == occ_class) &
            (summary[color_column] == occ_col)][metric_column]
        if sample_column:
            cond_err = summary[
                (summary[class_column] == occ_class) &
                (summary[color_column] == occ_col)]['sem']
            yvals = df[
                (df[class_column] == occ_class) &
                (df[color_column] == occ_col)]
            if invert_y:
                ax.bar(x, 1, bottom=cond_mean, color=color, #edgecolor=ecolor,
                       zorder=2)
            else:
                ax.bar(x, cond_mean, color=color, #edgecolor=ecolor,
                       zorder=2)
            #parts = ax.violinplot([yvals], [x], showextrema=False)
            #parts['bodies'][0].set_facecolor(color)
            #parts['bodies'][0].set_edgecolor(ecolor)
            #parts['bodies'][0].set_alpha(1)
            ax.errorbar(x, cond_mean, cond_err, color='k', capsize=2, zorder=4)
            #ax.scatter(x, cond_mean, color='w', s=2, zorder=3)
            sns.swarmplot(yvals, x=x, y=metric_column, native_scale=True,
                          alpha=.7, color='.5', dodge=True,
                          ax=ax, size=2, zorder=3)
            legend_marker = 'D'
        else:
            if invert_y:
                ax.bar(x, 1, bottom=cond_mean, color=color, zorder=2)#,
                # edgecolor=ecolor)
            else:
                ax.bar(x, cond_mean, color=color, zorder=2)#, edgecolor=ecolor,
                # zorder=1)
            legend_marker = 's'
    ax.tick_params(axis='both', which='both', left=False)
    plt.ylim(1, 0) if invert_y else plt.ylim(0, 1)
    plt.xticks([])
    # plt.xticks(xpos, conds_labels, rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel('')  # dep_var.replace('_', ' '))
    plt.xlim(-.7, 11.7)
    ax.grid(axis='y', linestyle='solid', alpha=.5,
            zorder=1, clip_on=False)
    #plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    if legend_path:
        make_legend(
            outpath=legend_path,
            labels=[', '.join(c) for c in CFG.occluder_conds_labels],
            markers=legend_marker,
            colors=CFG.plot_colors,
            markeredgecolors=CFG.plot_ecolors,
            linestyles='None')


def analyze_performance(overwrite=False):

    trials = load_trials()
    trials = trials[trials.subject.isin([f'sub-0{i}' for i in [1,3,5]]) ==
                    False].copy()
    metrics = ['accuracy', 'visibility']
    titles = dict(
        accuracy='classification accuracy',
        visibility='object visibility at RT')
    out_dir = 'analysis/performance'
    os.makedirs(out_dir, exist_ok=True)

    # main effects
    dep_vars = ['object_class', 'object_animacy',
                'occluder_class', 'occluder_superordinate',
                'occluder_color']

    for metric, dep_var in itertools.product(metrics, dep_vars):
        outpath = f'{out_dir}/{dep_var}_{metric}.png'
        if not op.isfile(outpath) or overwrite:
            print(f'Analyzing performance ({dep_var}, {metric})')
            conds = trials[dep_var].unique().tolist()
            if 'all' in conds:
                conds.remove('all')
            colors = list(distinct_colors.values())[:len(conds)]
            if metric == 'visibility':
                trial_set = trials[trials.accuracy == 1]
            else:
                trial_set = trials
            subject_means = trial_set.groupby(['subject'] + [dep_var]).agg(
                'mean', numeric_only=True).reset_index()
            summary = subject_means.drop(columns=['subject']).groupby(
                dep_var).agg(['mean', 'sem'])
            fig, ax = plt.subplots(figsize=(1 + len(conds)*.5,  4))
            if metric == 'accuracy':
                summary.plot(kind='bar',
                             y=(metric, 'mean'),
                             yerr=summary[(metric, 'sem')],
                             color=colors,
                             capsize=5,
                             legend=False,
                             ax=ax)
                plt.ylim(0,1)
            else:
                summary = summary.reset_index()
                for xpos, (cond, color) in enumerate(zip(conds, colors)):
                    yval = summary[summary[dep_var] == cond][(metric, 'mean')]
                    err = summary[summary[dep_var] == cond][(metric, 'sem')]
                    ax.bar(xpos, 1, bottom=yval, color=color)
                    ax.errorbar(xpos, yval, err, color='k')
                plt.ylim(1, 0)
            plt.ylabel(metric)
            plt.xticks(range(len(summary)), conds, rotation=90)
            #plt.xlabel(dep_var.replace('_', ' '))
            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()

            # make legend
            if metric == metrics[0]:  # don't repeat this for each metric
                make_legend(
                    outpath=f'{out_dir}/{dep_var}_legend.png', labels=conds,
                    markers='s', colors=colors, markeredgecolors=None,
                    linestyles='None')

    # condition-wise robustness
    for metric in metrics:
        outpath = (f'{out_dir}/occluder_class_x_occluder_color_{metric}.pdf')
        if not op.isfile(outpath) or overwrite:
            print(f'Analyzing occluder class * color performance ({metric})')
            condwise_robustness_plot(
                df=trials, outpath=outpath, metric_column=metric,
                class_column='occluder_class',
                color_column='occluder_color',
                sample_column='subject',
                title=titles[metric],
                legend_path=outpath.replace(metric, 'legend'))


    # noise ceiling (condition-wise, leave-one-participant-out)
    hum_vals = []
    trials_acc = trials[trials.accuracy == 1]
    for subject in trials_acc.subject.unique():
        hum_vals.append(trials_acc[trials_acc.subject == subject]
                        .groupby(['stimulus_set', 'occluder_class',
                                  'occluder_color'])
                        .mean(numeric_only=True)['visibility'].to_list())
    hum_vals = np.array(hum_vals)
    grp = np.mean(hum_vals, axis=0)
    subs = np.arange(hum_vals.shape[0])
    lwr, upr = [], []
    for s in subs:
        ind = hum_vals[s]
        rem_grp = np.mean(hum_vals[subs != s], axis=0)
        lwr.append(np.corrcoef(ind, rem_grp)[0, 1])
        upr.append(np.corrcoef(ind, grp)[0, 1])
    lwr, upr = np.mean(lwr), np.mean(upr)
    nc_df = pd.DataFrame({'analysis': ['condition-wise'],
                          'lwr': [lwr], 'upr': [upr]})

    # noise-ceiling (trial-wise, leave-one-participant-out)
    lwr, upr = [], []
    trials_acc = trials_acc.copy().sort_values(by='stimulus_id')
    for subject in trials_acc.subject.unique():
        trials_subj = trials_acc[trials_acc.subject == subject].copy()
        trials_grp_rem = trials_acc[trials_acc.subject != subject].groupby(
            'stimulus_id').mean(numeric_only=True).reset_index()
        common_trials = np.intersect1d(trials_subj.stimulus_id,
                                       trials_grp_rem.stimulus_id)
        trials_subj = trials_subj[trials_subj.stimulus_id.isin(
            common_trials)].copy()
        trials_grp_rem = trials_grp_rem[trials_grp_rem.stimulus_id.isin(
            common_trials)].copy()
        trials_grp = trials_acc[trials_acc.stimulus_id.isin(
            common_trials)].groupby('stimulus_id').mean(numeric_only=True)
        vis_subj = trials_subj.visibility.to_numpy()
        vis_grp_rem = trials_grp_rem.visibility.to_numpy()
        vis_grp = trials_grp.visibility.to_numpy()
        lwr.append(np.corrcoef(vis_subj, vis_grp_rem)[0, 1])
        upr.append(np.corrcoef(vis_subj, vis_grp)[0, 1])
    lwr, upr = np.mean(lwr), np.mean(upr)
    nc_df = (pd.concat([nc_df, pd.DataFrame(
        {'analysis': ['trial-wise'], 'lwr': [lwr], 'upr': [upr]})])
        .reset_index(drop=True))
    nc_df.to_csv(f'{out_dir}/noise_ceiling.csv', index=False)


def get_fixations(eye_data, trial_data):

    fx, fy, ff, fl, fd = [], [], [], [], []

    for t, row in trial_data.iterrows():

        # get indices for post-onset fixations that are not nan
        fix_starts = eye_data.fixstlist_stimulus[t]
        idcs = ~np.isnan(fix_starts) * fix_starts > 0

        # fixation locations
        fx.append((eye_data.fixxlist_stimulus[t][idcs].astype(int) -
                  STIM_LOC[0]).tolist())
        fy.append((eye_data.fixylist_stimulus[t][idcs].astype(int) -
                  STIM_LOC[1]).tolist())

        # fixation times
        firsts = eye_data.fixstlist_stimulus[t][idcs]
        ff.append(np.maximum(1,
            np.floor(firsts / SAMPLES_PER_FRAME)).astype(int).tolist())
        lasts = eye_data.fixetlist_stimulus[t][idcs]
        fl.append(np.minimum(NUM_FRAMES_ORIG,
            np.ceil(lasts / SAMPLES_PER_FRAME)).astype(int).tolist())
        fd.append((lasts - firsts).astype(int).tolist())

    # add to dataframe
    trial_data['fix_x'] = pd.Series(fx, dtype=object)
    trial_data['fix_y'] = pd.Series(fy, dtype=object)
    trial_data['fix_first_fr'] = pd.Series(ff, dtype=object)
    trial_data['fix_last_fr'] = pd.Series(fl, dtype=object)
    trial_data['fix_dur_ms'] = pd.Series(fd, dtype=object)

    return trial_data


def object_on_bg(image_path, alpha=None, format='numpy'):

    bg = Image.fromarray(
        np.tile(np.array(BG_COL, dtype=np.uint8), list(SCRN_RES)[::-1] + [1]))
    stimulus = Image.open(image_path).resize(STIM_SIZE_PIX).convert('RGB')
    image = bg.copy()
    image.paste(stimulus, STIM_LOC)

    # if alpha, use current image as RGB for the top layer, and place over bg
    if alpha is not None:
        if type(alpha) == np.ndarray:
            alpha = Image.fromarray(alpha, 'L')
        elif type(alpha) is int:
            alpha = Image.fromarray(
                np.ones(list(SCRN_RES)[::-1], dtype=np.uint8) * alpha, 'L')
        image.putalpha(alpha)  # stimulus needs to be same size as alpha
        bg.paste(image, (0, 0), image)
        image = bg

    if format == 'numpy':
        return np.array(image)
    elif format == 'pil':
        return image
    else:
        Exception(f'"{format}" is invalid format. Use "numpy" or "pil".')


def load_trials():
    if op.isfile(TRIALS_PATH):
        trials_group = pd.read_parquet(TRIALS_PATH)
    else:
        trials_group = pd.DataFrame()
    return trials_group


def process_raw_data():

    new_subjects = []
    trials_group = load_trials()

    for subject_dir in SUBJECT_DIRS:

        subject = op.basename(subject_dir).split('_')[0]

        if trials_group.empty or subject not in trials_group.subject.unique():

            new_subjects.append(subject)

            # convert edf file to asc file
            edf_file = f'{subject_dir}/eye_data.edf'
            asc_file = edf_file.replace('.edf', '.asc')
            if not op.isfile(asc_file):
                print(f'Converting edf file to asc for {subject}')
                os.system(f'edf2asc {edf_file}')

            # parse asc file
            print(f'Parsing eye data for {subject}')
            eye_data = parse(
                folder=subject_dir,
                ext='asc',
                multiprocess=8,
                traceprocessor=defaulttraceprocessor(
                    blinkreconstruct=True,  # interp pupil size during blinks
                    mode='advanced'))  # advanced mode is much quicker

            # get trial data, rename conditions, levels etc and add fixations
            print(f'\nProcessing fixations for {subject}')
            trials_ind = pd.read_csv(f'{subject_dir}/trials.csv')
            trials_ind.drop(columns=[
                'texture_superordinate', 'texture_subordinate',
                'texture_path'], inplace=True)
            trials_ind.rename(columns={
                'response': 'prediction',
                'object_superordinate': 'object_animacy',
                'object_ordinate': 'object_class',
                'occluder_ordinate': 'occluder_class',
                'texture_ordinate': 'occluder_color'},
                inplace=True)
            trials_ind.occluder_color = trials_ind.occluder_color.map(dict(
                dark='black', light='white'))
            trials_ind = get_fixations(eye_data, trials_ind)

            # collate trials
            trials_ind['subject'] = subject
            trials_ind['accuracy'] = trials_ind['accuracy'].astype(int)
            trials_ind['stimulus_set'] = trials_ind['stimulus_id'].str[0]
            trials_group = pd.concat([trials_group, trials_ind])
            trials_group.to_parquet(TRIALS_PATH, index=False)


def make_fixation_maps(overwrite=False, render_set='subset'):

    trials = load_trials()
    stim_ids = sorted(trials.stimulus_id.unique())
    map_types = ['rgb', 'heat', 'point', 'visibility']
    heatmap_lut = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8), cv2.COLORMAP_INFERNO)[128:]
    heatmap_lut = cv2.resize(heatmap_lut, (1, 256))

    def get_fixations(stimulus_id):

        """ saves fixation maps for human analysis. Saving these in full
        spatiotemporal resolution is not practical due to file size.
        Fortunately, this only needs to be run once, in theory. """

        map_res = np.array(SCRN_RES[::-1], dtype=np.uint16) #// downsample
        fixations = np.zeros(list(map_res) + [NUM_FRAMES_ORIG], dtype=np.uint8)
        fix_data = trials[
            (trials.stimulus_id == stimulus_id) &
            (trials.accuracy == 1)]
        for _, row in fix_data.iterrows():  # each accurate subject
            for f, (fx, fy, ff, fl) in enumerate(zip(  # each fixation
                    row.fix_x, row.fix_y, row.fix_first_fr,
                    row.fix_last_fr)):
                fx_adj = (fx + STIM_LOC[0]) #// downsample
                fy_adj = (fy + STIM_LOC[1]) #// downsample
                if fx_adj >= map_res[1] or fy_adj >= map_res[0]:
                    print(f'Fixation outside of screen bounds, skipping')
                else:
                    fixations[fy_adj, fx_adj, ff:(fl+1)] += 1
        return fixations

    def render_fixation_maps(fixations, out_dir):

        map_type, stimulus_id = out_dir.split('/')[-2:]
        image_paths = sorted(glob.glob(
            f'{STIM_DIR}/set-?/frames/{stimulus_id}*/*.png'))
        assert len(image_paths) == NUM_FRAMES_ORIG
        video = cv2.VideoWriter(f'{out_dir}/fixations.mp4',
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps=FPS, frameSize=SCRN_RES)
        os.makedirs(f'{out_dir}/frames', exist_ok=True)
        fixations = fixations / fixations.max() * 255

        for f, image_path in enumerate(tqdm(image_paths)):

            fix_f = fixations[:, :, f]
            if fix_f.max() == 0:
                alpha = int(map_type != 'visibility') * 255
                image_pil = object_on_bg(image_path, alpha=alpha, format='pil')
                image_np = np.array(image_pil)

            elif map_type == 'point':
                image_np = object_on_bg(image_path, format='numpy')
                fix_col = (255, 0, 0)
                for y, x in zip(*np.where(fix_f > 0)):
                    fix_size = int(fix_f[y, x] / 32)
                    image_np = cv2.circle(image_np, (x, y), fix_size, fix_col, -1)
                image_pil = Image.fromarray(image_np)

            elif map_type == 'heat':
                hm_data = cv2.GaussianBlur(fix_f, (255, 255), 31, 31)
                hm_data = (hm_data / hm_data.max() * 255).astype(np.uint8)
                image_pil = object_on_bg(image_path, format='pil')
                heatmap = cv2.cvtColor(hm_data, cv2.COLOR_GRAY2BGR)
                heatmap = cv2.LUT(heatmap, heatmap_lut)[:,:,::-1]
                heatmap = Image.fromarray(heatmap, 'RGB')  # numpy to PIL
                heatmap.putalpha(Image.fromarray(hm_data, 'L'))  # transp
                image_pil.paste(heatmap, (0, 0), heatmap)  # paste heatmap
                image_np = np.array(image_pil)

            elif map_type == 'visibility':
                hm_data = cv2.GaussianBlur(fix_f, (511, 511), 63, 63)
                hm_data = (hm_data / hm_data.max() * 255).astype(np.uint8)
                image_pil = object_on_bg(
                    image_path, alpha=hm_data, format='pil')
                image_np = np.array(image_pil)

            elif map_type == 'rgb':
                hm_data = cv2.GaussianBlur(fix_f, (255, 255), 31, 31)
                hm_data = (hm_data / hm_data.max() * 255).astype(np.uint8)
                image_np = object_on_bg(image_path, format='numpy')
                image_np = show_cam_on_image(image_np.astype(
                    np.float32) / 255, hm_data, use_rgb=True)
                image_pil = Image.fromarray(image_np)

            image_pil.save(f'{out_dir}/frames/{f:03d}.png')
            video.write(image_np[:,:, ::-1])  # RGB to BGR
        cv2.destroyAllWindows()
        video.release()

    # render maps
    stims_render = (np.arange(len(stim_ids)) if render_set == 'full' else
        np.arange(0, len(stim_ids), 32))
    for stim_idx, map_type in itp(stims_render, map_types):
        stimulus_id = stim_ids[stim_idx]
        print(f'Making fixation {map_type} maps for {stimulus_id}')
        fixations = get_fixations(stimulus_id)
        out_dir = f'analysis/fixation_maps/{map_type}/{stimulus_id}'
        os.makedirs(out_dir, exist_ok=True)
        render_fixation_maps(fixations, out_dir)


def save_modeling_data(overwrite=False):

    """ saves fixation maps, occluder masks, and other objects useful for
    computational modeling. Maps are generated at lower spatiotemporal
    resolution and only within image bounds for practicality. """
    out_dir = 'analysis/modeling_data'
    out_path = op.join(out_dir, 'fixation_maps.npy')
    if not op.isfile(out_path) or overwrite:
        print('Saving fixation maps')
        trials = load_trials()
        fix_maps = np.zeros([NUM_TRIALS, NUM_FRAMES, 224, 224], dtype=np.uint8)
        for s, stim_id in enumerate(STIM_IDS):
            total_fix, valid_fix = 0, 0
            for _, row in trials[
                (trials.stimulus_id == stim_id) &
                (trials.accuracy == 1)].iterrows():
                for f, (fx, fy, ff, fl) in enumerate(zip(  # each fixation
                        row.fix_x, row.fix_y, row.fix_first_fr,
                        row.fix_last_fr)):
                    fx_ds, fy_ds = int(fx // 4), int(fy // 4)
                    if fx_ds in np.arange(224) and fy_ds in np.arange(224):
                        frame_batches = [
                            f for f, batch in enumerate(FRAME_BATCHES) if
                            len(np.intersect1d(np.arange(ff, fl + 1), batch))]
                        fix_maps[s, frame_batches, fy_ds, fx_ds] += 1
                        valid_fix += 1
                    total_fix += 1
            valid_pcnt = 100 * (valid_fix / total_fix)
            print(f'Stimulus {row.stimulus_id}, {valid_fix}/{total_fix} '
                  f'({int(valid_pcnt)}%) fixations valid')
        with open(out_path, 'wb') as f:
            np.save(f, fix_maps)

        # smoothed version
        smoothed_maps = (transforms.GaussianBlur(127, 15)(
            torch.Tensor(fix_maps)
            .reshape(NUM_TRIALS * NUM_FRAMES, 1, 224, 224))
            .reshape(NUM_TRIALS, NUM_FRAMES, 224, 224)
            .numpy())
        with open(out_path.replace('.npy', '_smooth.npy'), 'wb') as f:
            np.save(f, smoothed_maps)

    # plot number of fixations as a function of visibility
    out_path = op.join(out_dir, 'total_fixations_time.png')
    if not op.isfile(out_path) or overwrite:
        with open(op.join(out_dir, 'fixation_maps.npy'), 'rb') as f:
            fix_maps = np.load(f)
        means = (torch.Tensor(fix_maps).permute(0, 2, 3, 1)
                 .sum(dim=(1, 2)).mean(dim=0).numpy())
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(VISIBILITIES, means, zorder=1)
        ax.scatter(VISIBILITIES, means)
        plt.xticks((0, 1))
        plt.xlabel('visibility')
        plt.ylabel('number of subjects fixating')
        plt.tight_layout()
        fig.savefig(
            op.join(op.dirname(out_path), 'total_fixations_time.png'))
        plt.close()

    # save an array of object_pixels for each trial
    out_path = op.join(out_dir, 'object_pixels.npy')
    if not op.isfile(out_path) or overwrite:
        trials = load_trials()
        object_pixels = np.empty((NUM_TRIALS, NUM_FRAMES, 224, 224), dtype=bool)
        for s, stim_id in enumerate(STIM_IDS):
            row = trials[trials.stimulus_id == stim_id].loc[0]
            for f, frame in enumerate(FRAMES):
                occluder_path = (f'stimuli/'
                                 f'{row.occluder_path}/{frame:03}.png')
                pix_obj = (np.array(Image.open(occluder_path).convert('RGBA')
                     .resize(size=(224,224), resample=Image.NEAREST)
                     )[:,:, 3] < 128)
                object_pixels[s, f] = pix_obj
        with open(f'stimuli/object_pixels.npy', 'wb') as f:
            np.save(f, object_pixels)


if __name__ == '__main__':
    main()






