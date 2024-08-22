# Image Classification experiment

import os, glob, sys, csv
import os.path as op
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats
from scipy.stats import norm
import math
import itertools
import matplotlib
TAB20 = matplotlib.cm.tab20.colors
sys.path.append(op.expanduser('~/david/master_scripts'))
from misc.plot_utils import custom_defaults, make_legend
from misc.math_functions import sigmoid
from itertools import product as itp
plt.rcParams.update(custom_defaults)
from scipy.optimize import curve_fit

data_dir = op.expanduser('~/david/projects/p022_occlusion/data/in_vivo/behavioral/exp1')
all_trials_path = f'{data_dir}/analysis/trials.parquet'
curve_params_path = f'{data_dir}/analysis/robustness_curves.parquet'
noise_ceiling_path = f'{data_dir}/analysis/noise_ceiling.csv'

def main():
    collate_trials()
    fit_performance_curves()
    make_plots()
    calculate_noise_ceiling()


class CFG:

    """
    This class contains useful details about the stimulus conditions and
    commonly used objects and constants, e.g. condition-color mappings, to keep
    plots consistent across human and model analysis pipelines.
    """

    # object classes
    classes_orig = [
        'brown bear, bruin, Ursus arctos',
        'bison',
        'African elephant, Loxodonta africana',
        'hare',
        'jeep, landrover',
        'table lamp',
        'sports car, sport car',
        'teapot']
    object_classes = ['bear', 'bison', 'elephant', 'hare',
               'jeep', 'lamp', 'car', 'teapot']
    class_idxs = [294, 347, 386, 331, 609, 846, 817, 849]
    class_dirs = ['n02132136', 'n02410509', 'n02504458', 'n02326432',
                  'n03594945', 'n04380533', 'n04285008', 'n04398044']
    animate_classes = object_classes[:4]
    inanimate_classes = object_classes[4:8]

    # occluders
    occluder_classes = [
        'barHorz04', 'barVert04', 'barObl04',
        'crossBarCardinal', 'crossBarOblique', 'mudSplash', 'polkadot',
        'polkasquare', 'naturalUntexturedCropped2']
    occluder_labels = [
        'bars (H)', 'bars (V)', 'bars (O)',
        'crossbars (C)', 'crossbars (O)', 'mud splash', 'polkadot',
        'polkasquare', 'natural']

    visibilities = [.1, .2, .4, .6, .8]
    occluder_colors = ['black', 'white']
    plot_colors = [TAB20[i] for i in [
        0, 1, 2, 3, 6, 7, 16, 17, 10, 11, 18, 19, 8, 9, 12, 13, 4, 5]]
    plot_ecolors = [TAB20[i] for i in [
        1, 0, 3, 2, 7, 6, 17, 16, 11, 10, 19, 18, 9, 8, 13, 12, 5, 4]]

    # all condition combinations
    occ_vis_combos = itp(occluder_classes, visibilities)
    occ_vis_combos = [*[('none', 1.)], *occ_vis_combos]
    occ_col_combos = itp(occluder_classes, occluder_colors)
    occ_col_labels = [', '.join([o, c]) for o, c in itp(occluder_labels,
                                                        occluder_colors)]
    cond_combos = itertools.product(occluder_classes, occluder_colors, visibilities)
    cond_combos = [*[('none', 'none', 1.)], *cond_combos]
    """
    scripts to make labels by loading imagenet label file
    label_data = pd.read_csv(open(op.expanduser(
        '~/david/datasets/images/ILSVRC2012/labels.csv'), 'r+'))
    class_idxs = [
        label_data['index'][label_data['directory'] == class_dir].item() for
        class_dir in class_dirs]
    class_labels_alt = [
        label_data['label'][label_data['directory'] == class_dir].item() for
        class_dir in class_dirs]
    """

    subjects = sorted([int(op.basename(x)) for x in glob.glob(
        f'{data_dir}data/*') if op.isdir(x)])


def collate_trials():

    """
    This function combines the trial data from all subjects into a single
    dataframe, with some minor adjustments to align with broader conventions
    across experiments. The resulting dataframe is saved as a parquet file.
    """

    if not op.isfile(all_trials_path):
        all_trials = pd.DataFrame()
        for subject in CFG.subjects:
            trials_path = f'{data_dir}/data/{subject}/trials.csv'
            trials = pd.read_csv(trials_path, index_col=0)
            trials['subject'] = subject
            trials['trial'] = trials.index + 1
            trials.rename(columns={'response': 'prediction',
                                   'occluder': 'occluder_class',
                                   'colour': 'occluder_color',
                                   'class': 'object_class'}, inplace=True)
            trials.occluder_class = trials.occluder_class.replace(
                {'unoccluded': 'none'})
            trials.occluder_color = trials.occluder_color.replace(
                {np.nan: 'none'})
            all_trials = pd.concat([all_trials, trials])
        all_trials.reset_index(drop=True, inplace=True)
        all_trials['stimulus_id'] = [f'{t+1:05}' for t in range(len(all_trials))]
        all_trials['visibility'] = all_trials['visibility'].round(1)
        all_trials['object_animacy'] = ['animate' if c in CFG.animate_classes
            else 'inanimate' for c in all_trials['object_class']]
        all_trials.to_parquet(all_trials_path, index=False)


def fit_performance_curves():

    if not op.isfile(curve_params_path):
        all_trials = pd.read_parquet(all_trials_path)
        curve_params = pd.DataFrame()

        for metric, (level, subject_sample) in itp(
                ['accuracy', 'RT'], zip(['individual', 'group'],
                                        [CFG.subjects, ['group']])):

            if level == 'individual':
                performance = (all_trials.groupby(
                    ['subject', 'occluder_class', 'occluder_color',
                     'visibility'], dropna=False)
                   .agg('mean', numeric_only=True).reset_index())
            else:  # if level == 'group'
                performance = (all_trials.groupby(
                    ['occluder_class', 'occluder_color', 'visibility'], dropna=False)
                    .agg('mean', numeric_only=True).reset_index())
                performance['subject'] = 'group'

            # condition-wise curve params and thresholds
            for subject, occluder_class, occluder_color in itertools.product(
                    subject_sample, CFG.occluder_classes, CFG.occluder_colors):

                perf_unocc = performance[
                    (performance.visibility == 1) &
                    (performance.subject == subject)][metric].item()

                perf_occ = (performance[
                    (performance.occluder_class == occluder_class) &
                    (performance.occluder_color == occluder_color) &
                    (performance.subject == subject)]
                    .sort_values('visibility')[metric].to_list())

                if metric == 'accuracy':
                    xvals = [0] + CFG.visibilities + [1]
                    yvals = np.array(
                        [1/8] + list(perf_occ) + [perf_unocc])
                else:  # if metric == 'RT'
                    xvals = CFG.visibilities + [1]
                    yvals = np.array(list(perf_occ) + [perf_unocc])
                init_params = [max(yvals), np.median(xvals), 1, 0]
                curve_x = np.linspace(0, 1, 1000)
                try:
                    popt, pcov = curve_fit(
                        sigmoid, xvals, yvals, init_params, maxfev=100000)
                    curve_y = sigmoid(curve_x, *popt)
                    threshold = sum(curve_y < .5) / 1000
                except:
                    popt = [np.nan] * 4
                    threshold = np.nan

                curve_params = pd.concat(
                    [curve_params, pd.DataFrame({
                        'subject': [str(subject)],
                        'occluder_class': [occluder_class],
                        'occluder_color': [occluder_color],
                        'metric': [metric],
                        'L': [popt[0]],
                        'x0': [popt[1]],
                        'k': [popt[2]],
                        'b': [popt[3]],
                        'threshold_50': [threshold],
                        'mean': [np.mean(perf_occ)]
                    })]).reset_index(drop=True)

            # mean curve across all conditions
            for subject in subject_sample:

                perf = (performance[performance.subject == subject]
                    .sort_values('visibility')[metric].to_list())

                if metric == 'accuracy':
                    xvals = [0] + CFG.visibilities + [1]
                    yvals = np.array([1/8] + perf)
                else:  # if metric == 'RT'
                    xvals = CFG.visibilities + [1]
                    yvals = np.array(perf)
                init_params = [max(yvals), np.median(xvals), 1, 0]
                curve_x = np.linspace(0, 1, 1000)
                try:
                    popt, pcov = curve_fit(
                        sigmoid, xvals, yvals, init_params, maxfev=100000)
                    curve_y = sigmoid(curve_x, *popt)
                    threshold = sum(curve_y < .5) / 1000
                except:
                    popt = [np.nan] * 4
                    threshold = np.nan

                curve_params = pd.concat(
                    [curve_params, pd.DataFrame({
                        'subject': [str(subject)],
                        'occluder_class': ['all'],
                        'occluder_color': ['all'],
                        'metric': [metric],
                        'L': [popt[0]],
                        'x0': [popt[1]],
                        'k': [popt[2]],
                        'b': [popt[3]],
                        'threshold_50': [threshold],
                        'mean': [np.mean(perf[:-1])]
                    })]).reset_index(drop=True)

        curve_params.subject = curve_params.subject.astype('category')
        curve_params.to_parquet(curve_params_path, index=False)


def condwise_robustness_plot_array(df, metric,
                                    outpath, ylabel, df_curves=None,
                                    yticks=(0,1), ylims=(0,1),
                                    chance=None, legend_path=None):

    fig, axes = plt.subplots(
        3, 3, figsize=(3.5, 3.5), sharex=True, sharey=True)

    xvals = CFG.visibilities
    perf_unocc = df[df.visibility == 1][metric].mean()

    for o, occluder_class in enumerate(CFG.occluder_classes):
        ax = axes.flatten()[o]

        for c, occluder_color in enumerate(CFG.occluder_colors):
            face_color = CFG.plot_colors[o * 2 + c]
            edge_color = CFG.plot_colors[o * 2 + c]

            # plot curve function underneath
            if df_curves is not None:
                if 'subject' in df_curves.columns:
                    df_curves = df_curves[df_curves.subject == 'group']
                popt = (df_curves[
                    (df_curves.metric == metric) &
                    (df_curves.occluder_class == occluder_class) &
                    (df_curves.occluder_color == occluder_color)]
                    [['L', 'x0', 'k', 'b']]).values[0]
                if np.isfinite(popt).all():
                    curve_x = np.linspace(0, 1, 1000)
                    curve_y = sigmoid(curve_x, *popt)
                    ax.plot(curve_x, curve_y, color=edge_color, zorder=1)

            # plot accuracies on top
            perf_occ = df[
                (df.occluder_class == occluder_class) &
                (df.occluder_color == occluder_color)]
            yvals = perf_occ.groupby('visibility')[metric].mean().to_list()
            ax.scatter(xvals, yvals, facecolor=face_color, clip_on=False,
                       edgecolor=edge_color, zorder=2)

        # plot unoccluded
        ax.scatter(1, perf_unocc, color='white',
                   edgecolor='k', zorder=3, clip_on=False)

        # format plot
        #ax.set_title(CFG.occluder_labels[o], size=7)
        ax.set_xticks((0, 1))
        ax.set_xlim((0, 1))
        ax.set_yticks(yticks)
        ax.set_ylim(ylims)
        ax.tick_params(axis='both', which='major', labelsize=7)
        # ax.axhline(y=acc1unalt, color=colors[0], linestyle='dashed')
        if chance is not None:
            ax.axhline(y=chance, color='k', linestyle='dotted')
        if o == 7:
            ax.set_xlabel('visibility', size=10)
        if o == 3:
            ax.set_ylabel(ylabel, size=10)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    if legend_path:
        make_legend(
            outpath=legend_path,
            labels=['unoccluded'] + CFG.occ_col_labels,
            markers='o',
            colors=['w'] + CFG.plot_colors,
            markeredgecolors=None,
            linestyles='None')


def make_plots():

    all_trials = pd.read_parquet(all_trials_path)
    performance = (all_trials.groupby(
        ['subject', 'occluder_class', 'occluder_color', 'visibility'])
        .agg('mean', numeric_only=True).reset_index())
    for metric in ['accuracy', 'RT']:

        if metric == 'accuracy':
            ylabel = 'classification accuracy'
            ylims = (0, 1)
            yticks = (0, 1)
            chance = 1 / 8
        elif metric == 'RT':
            ylabel = 'RT (s)'
            yticks = (0, 1, 2)
            ylims = (0, 2)
            chance = None

        curve_params = pd.read_parquet(curve_params_path)

        outpath = f'{data_dir}/analysis/plots/group_mean_{metric}.pdf'
        condwise_robustness_plot_array(
            df=performance,
            df_curves=curve_params,
            metric=metric,
            outpath=outpath,
            ylabel=ylabel,
            yticks=yticks,
            ylims=ylims,
            chance=chance,
            legend_path=outpath.replace(metric, 'legend'))


def calculate_noise_ceiling():

    """
    This function calculates the noise-ceiling, i.e., between-subject
    reliability of condition-wise performance. There are several versions,
    which include different combinations of dependent variables.
    """

    all_trials = pd.read_parquet(all_trials_path)
    performance = (all_trials.groupby(
        ['subject', 'occluder_class', 'occluder_color', 'visibility'])
        .agg('mean', numeric_only=True).reset_index())

    nc_df = pd.DataFrame()
    performance_occ = performance[performance.visibility < 1]
    for grouping_vars, level in zip(
            ['occluder_class', ['occluder_class', 'occluder_color'],
             ['occluder_class', 'occluder_color', 'visibility']],
            ['occluder_class', 'occluder_class_x_occluder_color',
             'occluder_class_x_occluder_color_x_visibility']):
        hum_vals = []
        for subject in CFG.subjects:
            hum_vals.append(performance_occ[
                (performance_occ.subject == subject)]
                .groupby(grouping_vars)
                .mean(numeric_only=True).accuracy.to_list())
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
        nc_df = pd.concat([nc_df, pd.DataFrame({
            'level': [level],
            'lwr': [lwr],
            'upr': [upr]})])

    # for correlation across visibilities, including unoccluded,
    # separately for each occluder class/colour combination
    lwr, upr = [], []
    for occluder_class, occluder_color in itp(CFG.occluder_classes,
                                              CFG.occluder_colors):
        hum_vals = []
        for subject in performance.subject.unique():
            hum_vals.append(performance[
                (performance.subject == subject) &
                (performance.occluder_class.isin(
                    [occluder_class, 'none'])) &
                (performance.occluder_color == occluder_color)].groupby(
                'visibility')
                .accuracy.mean().to_list())
        hum_vals = np.array(hum_vals)
        grp = np.mean(hum_vals, axis=0)
        subs = np.arange(hum_vals.shape[0])
        for s in subs:
            ind = hum_vals[s]
            rem_grp = np.mean(hum_vals[subs != s], axis=0)
            lwr.append(np.corrcoef(ind, rem_grp)[0, 1])
            upr.append(np.corrcoef(ind, grp)[0, 1])
    lwr, upr = np.mean(lwr), np.mean(upr)
    nc_df = pd.concat([nc_df, pd.DataFrame({
        'level': ['visibility_corr_occluder_class_x_occluder_color'],
        'lwr': [lwr],
        'upr': [upr]})])

    nc_df.to_csv(noise_ceiling_path, index=False)


if __name__ == '__main__':
    main()





    



