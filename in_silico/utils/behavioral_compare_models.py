'''
This scripts makes plots and runs statistical analyses comparing model
performance in the behavioral benchmarks.
'''

import os
import os.path as op
import glob
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import seaborn as sns
from scipy import stats
from scipy import special
import pickle as pkl
import pandas as pd
from scipy.optimize import curve_fit
import math
import time
from types import SimpleNamespace
from itertools import product as itp
import pingouin as pg
from datetime import datetime
from tqdm import tqdm
import warnings

from .model_contrasts import (model_base, model_contrasts, model_dirs,
    effect_colors, region_to_layer)

sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from math_functions import sigmoid
from plot_utils import make_legend, custom_defaults
plt.rcParams.update(custom_defaults)

sys.path.append(f'../in_vivo/behavioral')
from exp1.analysis import CFG as EXP1
from exp2.analysis import CFG as EXP2
from .behavioral_benchmark import load_trials, reshape_metrics
from .helper_functions import now

EXPS = ['exp1', 'exp2']
METRICS = ['accuracy', 'true_class_prob', 'entropy']
OBJ_VARIABLES = ['object_animacy', 'object_class']
OBJ_ANIMACIES = ['animate', 'inanimate']
OBJ_CLASSES = EXP1.object_classes
OBJ_CLS_IDCS = EXP1.class_idxs
OBJ_CLS_DIRS = EXP1.class_dirs
OCC_VARIABLES = ['visibility', 'occluder_class', 'occluder_color']
OCC_COLORS = ['black', 'white']
OCC_CLASSES = {'exp1': EXP1.occluder_classes,
               'exp2': EXP2.occluder_classes}
HUMAN_COLOR = 'tab:gray'
UNOCC_COLOR = (.8, .8, .8)
NUM_FRAMES = 17
VISIBILITIES = {'exp1': EXP1.visibilities,
                'exp2': np.linspace(0, 1, NUM_FRAMES)}
FRAMES = np.linspace(1, 360, NUM_FRAMES).astype(int)
T = SimpleNamespace(classification=True, nGPUs=-1, batch_size=64)


def get_group_counts(config, df, metric):
    group_counts = {}
    if 'humans' in df.model.unique() and df[df.model == 'humans'][metric].any():
        group_counts['humans'] = 1
    for m, (label, info) in enumerate(config.items()):
        group = info['group']
        if group in df.group.to_list() and group != 'humans':
            if group in group_counts:
                group_counts[group] += 1
            else:
                group_counts[group] = 1
    return group_counts


def complete_model_config(model_config):

    for m, (label, info) in enumerate(model_config.items()):
        if 'group' not in info:
            info['group'] = 'model'
        if 'color' not in info:
            info['color'] = colormaps.tab20.colors[m]
        if 'xpos' not in info:
            info['xpos'] = m

    if 'humans' not in model_config:
        model_config['humans'] = {'group': 'humans',
                                  'color': HUMAN_COLOR,
                                  'xpos': 0}

    return model_config


def collate_data_exp1(config, model_contrast, transfer_method):

    exp = 'exp1'

    # human data
    human_dir = f'../data/in_vivo/behavioral/{exp}'
    groupbys = ['subject', 'occluder_class', 'occluder_color', 'visibility']
    robustness = load_trials(exp, drop_human=False)
    robustness = (robustness
        .groupby(groupbys, dropna=False)
        .agg('mean', numeric_only=True).reset_index())
    robustness['model'] = 'humans'
    robustness['group'] = 'humans'
    robustness['layer'] = 'humans'
    robustness['cycle'] = 0
    curves = pd.read_parquet(f'{human_dir}/analysis/robustness_curves.parquet')
    curves['model'] = 'humans'
    noise_ceiling = pd.read_csv(f'{human_dir}/analysis/noise_ceiling.csv')
    likeness = pd.DataFrame()

    # model data
    for m, (label, info) in enumerate(config.items()):

        path, color = info['path'], info['color']
        model_name = path.split('/')[0]
        model_dir = f'{model_base}/{path}'
        data_dir = f'{model_dir}/behavioral/{exp}'
        group = info['group'] if 'group' in info else model_contrast

        # check whether this transfer method was applied with this model
        df = load_trials(exp, model_dir=model_dir)
        if transfer_method == 'output':
            layer = 'output'
        else:
            layer = region_to_layer[model_name]['IT']
        if layer not in df.layer.unique():
            continue

        # use reconstruction accuracy for pix2pix models
        if 'pix2pix' in model_dir:
            df = pd.read_parquet(f'{model_dir}/behavioral/exp1/'
                                 f'reconstruction.parquet')
            layer = 'reconstruction'

        # robustness
        df = (df[df.layer == layer]
            .groupby(groupbys + ['cycle'], dropna=False)
            .agg('mean', numeric_only=True).reset_index())
        df['model'] = label
        df['layer'] = layer
        df['group'] = group
        robustness = pd.concat([robustness, df])

        # visibility curve functions
        df = pd.read_parquet(f'{data_dir}/robustness_curves.parquet')
        df = df[df.layer == layer]
        df['model'] = label
        df['layer'] = layer
        df['group'] = group
        curves = pd.concat([curves, df])

        # human likeness
        df = pd.read_parquet(f'{data_dir}/human_likeness.parquet')
        df = df[df.layer == layer]
        df['model'] = label
        df['layer'] = layer
        df['group'] = group
        likeness = pd.concat([likeness, df])

    return robustness, curves, likeness, noise_ceiling


def collate_data_exp2(config, model_contrast, transfer_method):

    exp = 'exp2'

    # human data
    groupbys = ['occluder_class', 'occluder_color']
    trials_h = load_trials(exp, drop_human=False)
    trials_h = (trials_h
        .groupby(groupbys + ['subject'])
        .agg('mean', numeric_only=True)
        .reset_index())
    trials_h['model'] = 'humans'
    trials_h['group'] = 'humans'
    trials_h['layer'] = 'humans'
    trials_h['cycle'] = 0
    noise_ceiling = pd.read_csv('../data/in_vivo/behavioral/exp2/analysis/'
                                'performance/noise_ceiling.csv')

    # model data
    trials_m = pd.DataFrame()
    trials_m_rt = pd.DataFrame()
    curves_m = pd.DataFrame()
    likeness = pd.DataFrame()

    for m, (label, info) in enumerate(config.items()):
        path, color = info['path'], info['color']
        model_dir = f'{model_base}/{path}'
        data_dir = f'{model_dir}/behavioral/{exp}'
        group = info['group'] if 'group' in info else model_contrast

        if ('task-cont' in model_dir and 'transfer' not in model_dir and
                transfer_method == 'output'):
            continue

        # trials
        df = load_trials(exp, model_dir=model_dir)
        if transfer_method == 'output':
            layer = 'output'
        else:
            layer = [l for l in df.layer.unique() if l != 'output'][0]
        df = df[df.layer == layer]
        df = (df
              .groupby(groupbys + ['visibility', 'cycle'])
              .agg('mean', numeric_only=True)
              .reset_index())
        df['model'] = label
        df['layer'] = layer
        df['group'] = group
        #df = reshape_metrics(df, 'long')
        trials_m = pd.concat([trials_m, df])

        # estimated RTs
        df = pd.read_parquet(f'{data_dir}/trials_RT.parquet')
        df = df[df.layer == layer]
        df['model'] = label
        df['layer'] = layer
        df['group'] = group
        trials_m_rt = pd.concat([trials_m_rt, df])

        # visibility curve functions
        df = pd.read_parquet(f'{data_dir}/robustness_curves.parquet')
        df = df[df.layer == layer]
        df = (df
              .groupby(groupbys + ['cycle'])
              .agg('mean', numeric_only=True)
              .reset_index())
        df['model'] = label
        df['layer'] = layer
        df['group'] = group
        curves_m = pd.concat([curves_m, df])

        # human-likeness
        df = pd.read_parquet(f'{data_dir}/human_likeness.parquet')
        df = df[df.layer == layer]
        df['model'] = label
        df['layer'] = layer
        df['group'] = group
        likeness = pd.concat([likeness, df])

    return trials_h, trials_m, trials_m_rt, likeness, noise_ceiling


def split_bar(num_cycles, xbase):
    bar_gap = .2
    width = (1 - bar_gap) / num_cycles
    left_bar = xbase - .5 + bar_gap / 2 + width / 2
    right_bar = xbase + .5 - bar_gap / 2 - width / 2
    xposs = np.linspace(left_bar, right_bar, num_cycles)
    return xposs, width


def bar_plot_single_cycle(model_config, df, metric, samples, plot_config,
                          plot_file, noise_ceiling=None):

    group_counts = get_group_counts(model_config, df, metric)
    humans = int('humans' in group_counts)
    invert_y = 'invert_y' in plot_config and plot_config['invert_y']
    edgecolor = 'white' if 'edgecolor' not in samples else samples['edgecolor']
    linewidth = 0 if 'linewidth' not in samples else samples['linewidth']
    figsize = (2 + ((len(model_config) + humans) / 4), 3)
    fig, axes = plt.subplots(
        ncols=len(group_counts), sharey='row', figsize=figsize,
        gridspec_kw={'width_ratios': list(group_counts.values())})
    legend_colors, legend_labels = [], []

    for g, group in enumerate(group_counts):

        ax = axes[g]
        df_group = df[df.group == group]
        for m, model in enumerate(df_group.model.unique()):
            xpos = model_config[model]['xpos']
            color = model_config[model]['color']
            if color not in legend_colors:
                legend_colors.append(color)
                legend_labels.append(model)
            df_model = (df_group[df_group.model == model]
                .groupby(samples['column'], dropna=False)
                .agg('mean', numeric_only=True)
                .reset_index())
            yvals = df_model.sort_values(by=samples['column'])[metric].to_list()
            sns.stripplot(x=xpos, y=yvals, color=samples['color'], zorder=3,
                          size=samples['size'], alpha=samples['alpha'],
                          edgecolor=edgecolor, linewidth=linewidth,
                          clip_on=False,
                          native_scale=True, dodge=True, ax=ax)
            if 'errorbar' in samples:
                cond_mean, cond_err = np.mean(yvals), stats.sem(yvals)
                ax.errorbar(xpos, cond_mean, cond_err, color='k', capsize=2,
                            zorder=3)
            if 'occluder_class' in df_model.columns:
                yval = df_model[(df_model.occluder_class != 'unoccluded') &
                                (~df_model.occluder_color.isna())][metric].mean()
            else:
                yval = np.mean(yvals)
            bottom = yval if invert_y else 0
            yval = 1 if invert_y else yval
            ax.bar(xpos, yval, bottom=bottom, color=color, zorder=1)
        ax.set_title(df_group.group.values[0], pad=20, size=12)
        ax.set_xticks([])
        ax.set_xlim(-.5, m + .5)
        if 'chance' in plot_config:
            ax.axhline(y=plot_config['chance'], color='k', linestyle='dotted')
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=4, clip_on=False)
        if 'ylims' in plot_config:
            ax.set_ylim(plot_config['ylims'])
            if min(plot_config['ylims']) < 0:
                ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False)
        if g == 0:
            ax.set_ylabel(plot_config['ylabel'])
            ax.set_yticks(plot_config['yticks'])
        else:
            ax.spines['left'].set_visible(False)
        if noise_ceiling is not None:
            ax.fill_between(np.arange(-1, 50), noise_ceiling[0],
                            noise_ceiling[1], color='tab:gray', lw=0,
                            zorder=0)
    plt.tight_layout()
    fig.savefig(plot_file)
    plt.close()

    # legend
    make_legend(
        outpath=f'{op.dirname(plot_file)}/legend.pdf',
        labels=legend_labels,
        markers='s',
        colors=legend_colors,
        markeredgecolors=None,
        linestyles='None')


def bar_plot_multi_cycle(model_config, df, metric, plot_config, plot_file):

    group_counts = get_group_counts(model_config, df, metric)
    humans = int('humans' in df.columns)
    invert_y = 'invert_y' in plot_config and plot_config['invert_y']
    figsize = (2 + ((len(model_config) + humans) / 4), 3)
    fig, axes = plt.subplots(
        ncols=len(group_counts), sharey='row', figsize=figsize,
        gridspec_kw={'width_ratios': list(group_counts.values())})

    for g, group in enumerate(group_counts):

        ax = axes[g]
        df_group = df[df.group == group]
        for m, model in enumerate(df_group.model.unique()):
            xbase = model_config[model]['xpos']
            color = model_config[model]['color']
            df_model = df_group[df_group.model == model]
            cycles = df_model.cycle.unique()
            xposs, width = split_bar(len(cycles), xbase)
            for xpos, cycle in zip(xposs, cycles):

                if 'visibility' in df_model.columns:
                    unocc = df_model[
                        (df_model.visibility == 1) &
                        (df_model.cycle == cycle)][metric].mean()
                    ax.bar(xpos, unocc, color=UNOCC_COLOR, width=width)
                    yval = df_model[
                        (df_model.visibility < 1) &
                        (df_model.cycle == cycle)][metric].mean()
                else:
                    yval = df_model[df_model.cycle == cycle][metric].mean()
                bottom = yval if invert_y else 0
                yval = 1 if invert_y else yval
                ax.bar(xpos, yval, bottom=bottom, color=color, width=width,
                       zorder=2)
        ax.set_title(df_group.group.values[0], pad=20)
        ax.set_xticks([])
        ax.set_xlim(-.5, m + .5)
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=4, clip_on=False)
        ax.set_yticks(plot_config['yticks'])
        if 'ylims' in plot_config:
            ax.set_ylim(plot_config['ylims'])
            if min(plot_config['ylims']) < 0:
                ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False)
        if g == 0:
            ax.set_ylabel(plot_config['ylabel'])
            ax.set_yticks(plot_config['yticks'])
        else:
            ax.spines['left'].set_visible(False)
    plt.tight_layout()
    fig.savefig(plot_file)
    plt.close()


def add_unoccluded_exp1(df):
    df = df.copy()
    df.occluder_class = pd.Categorical(df.occluder_class,
        categories=EXP1.occluder_classes + ['unoccluded'], ordered=True)
    df.occluder_class.fillna('unoccluded')
    df.occluder_color = pd.Categorical(df.occluder_color,
        categories=OCC_COLORS + ['unoccluded'], ordered=True)
    df.occluder_color.fillna('unoccluded')
    return df


def compare_models_exp1(overwrite=False):

    def _human_likeness_plots(df, model_config, plot_configs, noise_ceiling,
                             results_dir, overwrite):

        analysis, metric, metric_sim, level = df.name
        out_dir = f'{results_dir}/human_likeness/{analysis}/{level}'
        os.makedirs(out_dir, exist_ok=True)

        if level.replace('subject_x_', '') in noise_ceiling.level.unique():
            nc = (noise_ceiling[
                noise_ceiling.level == level.replace('subject_x_', '')][
                ['lwr', 'upr']]).values[0]
        else:
            nc = None

        # all cycles
        plot_file = f'{out_dir}/{metric_sim}_{metric}_cyclewise.pdf'
        if df.cycle.max() > 0 and (not op.isfile(plot_file) or overwrite):
            bar_plot_multi_cycle(model_config, df, 'value',
                                 plot_configs[analysis][metric_sim],
                                 plot_file)

        # final cycle
        plot_file = f'{out_dir}/{metric_sim}_{metric}.pdf'
        if not op.isfile(plot_file) or overwrite:
            df_plot = (
                df
                .groupby(['subject', 'model', 'layer'])
                .apply(lambda d: d[d.cycle == d.cycle.max()])
                .reset_index(drop=True))
            plot_config = plot_configs[analysis][metric_sim]
            samples = {'column': 'subject',
                       'color': HUMAN_COLOR,
                       'size': 3,
                       'alpha': .5}
            #if metric == 'accuracy':
            #    df_plot.accuracy[
            #        df_plot.group == 'Pix2Pix\nreconstruction'
            #        ] = robustness_fc.accuracy_v_reconstruction_loss[
            #        df_plot.group == 'Pix2Pix\nreconstruction']
            bar_plot_single_cycle(
                model_config=model_config,
                df=df_plot,
                metric='value',
                samples=samples,
                plot_config=plot_config,
                plot_file=plot_file,
                noise_ceiling=nc)

            # ANOVA
            df_plot['architecture'] = (
                df_plot['model'].apply(lambda r: r.split(', ')[0]))
            df_plot['dataset'] = (
                df_plot['model'].apply(lambda r: r.split(', ')[1]))
            df_plot['task'] = (
                df_plot['model'].apply(lambda r: r.split(', ')[2]))
            anova = pg.rm_anova(
                dv='value', within=['task', 'dataset'],
                subject='subject', data=df_plot,
                detailed=True)
            try:
                post_hocs = pg.pairwise_tests(
                    dv='value', within=['architecture', 'task', 'dataset'],
                    subject='subject', data=df_plot, padjust='bonf')
            except:
                post_hocs = (df_plot
                    .groupby(['architecture'])
                    .apply(lambda df: pg.pairwise_tests(
                        dv='value', within='dataset', subject='subject',
                        data=df, padjust='bonf'))
                    .reset_index())
                post_hocs = pd.concat([post_hocs, (df_plot
                    .groupby(['architecture', 'task'])
                    .apply(lambda df: pg.pairwise_tests(
                        dv='value', within='dataset', subject='subject',
                        data=df, padjust='bonf'))
                    .reset_index())])
            anova.to_csv(f'{out_dir}/{metric}_anova.csv')
            post_hocs.to_csv(f'{out_dir}/{metric}_posthocs.csv')

            # save performance profile
            summary = (df_plot
                       .drop(columns='subject')[['model', 'value']]
                       .groupby('model').agg(['mean', 'sem'], numeric_only=True)
                       .reset_index())
            summary.columns = ['model', 'mean', 'sem']
            summary = pd.concat([
                summary, pd.DataFrame({
                    'model': ['nc_lwr', 'nc_upr'],
                    'mean': nc,
                    'sem': [np.nan, np.nan]})])
            summary.to_csv(f'{out_dir}/{metric}_summary.csv')


    for (model_contrast, model_config), transfer_method in \
            itp(model_contrasts.items(), ['SVC', 'output']):

        print(f'{now()} | Comparing models (exp1) | {model_contrast} |'
              f' {transfer_method}')
        config = complete_model_config(model_config.copy())
        results_dir = (f'../data/in_silico/analysis/'
                       f'{model_contrast}/behavior/exp1/{transfer_method}')
        robustness, curves, likeness, noise_ceiling = collate_data_exp1(
            model_config, model_contrast, transfer_method)

        # robustness plots
        plot_configs = {
            'accuracy': {
                'title': 'classification accuracy',
                'ylabel': 'classification accuracy',
                'yticks': np.arange(0, 2, .2),
                'ylims': (0, 1),
                'chance': 1 / 8},
            'true_class_prob': {
                'title': 'probability estimate for true class',
                'ylabel': 'probability',
                'yticks': np.arange(0, 2, .2),
                'ylims': (0, 1),
                'chance': 1 / 8},
            'entropy': {
                'title': 'uncertainty across 8 classes',
                'ylabel': 'shannon entropy',
                'yticks': np.arange(0, 4, 1),
                'ylims': (0, 3)}}

        out_dir = f'{results_dir}/occlusion_robustness'
        os.makedirs(out_dir, exist_ok=True)

        for metric in METRICS:

            # all cycles
            plot_file = f'{out_dir}/{metric}_barplot_cyclewise.pdf'
            if robustness.cycle.max() > 0 and \
                    (not op.isfile(plot_file) or overwrite):
                bar_plot_multi_cycle(config, robustness, metric,
                                     plot_configs[metric], plot_file)

            # final cycle only, with anova and posthocs
            plot_file = f'{out_dir}/{metric}_barplot.pdf'
            if not op.isfile(plot_file) or overwrite:
                robustness_fc = (robustness
                                 .groupby(['model', 'layer'])
                                 .apply(lambda d: d[d.cycle == d.cycle.max()])
                                 .reset_index(drop=True))
                robustness_fc = add_unoccluded_exp1(robustness_fc)
                samples = {'column': ['occluder_class', 'occluder_color'],
                           'color': EXP1.plot_colors + ['w'],
                           'edgecolor': EXP1.plot_colors + ['k'],
                           'linewidth': [0] * len(EXP1.plot_colors) + [1],
                           'size': 5,
                           'alpha': 1}
                #if metric == 'accuracy':
                #    robustness_fc.accuracy[
                #        robustness_fc.group == 'Pix2Pix\nreconstruction'
                #        ] = robustness_fc.reconstruction_loss[
                #        robustness_fc.group == 'Pix2Pix\nreconstruction']
                bar_plot_single_cycle(config, robustness_fc, metric,
                    samples, plot_configs[metric], plot_file)
            anova_path = f'{out_dir}/{metric}_anova.csv'
            ph_path = anova_path.replace('anova', 'posthocs')
            if not op.isfile(anova_path) or not op.isfile(ph_path) or overwrite:
                model_perf = robustness_fc[
                    robustness_fc.visibility < 1].groupby(
                    ['model', 'subject']).agg(
                    'mean', numeric_only=True).reset_index()
                anova = pg.rm_anova(
                    dv=metric, within='model', subject='subject',
                    data=model_perf, detailed=True)
                anova.to_csv(anova_path)

                post_hocs = pd.DataFrame()
                try:
                    post_hocs = pd.concat([post_hocs, pg.pairwise_tests(
                        dv='value', within=['architecture', 'task', 'dataset'],
                        subject='subject', data=robustness_fc, padjust='bonf')])
                    success = True
                except:
                    success = False
                if not success:
                    try:
                        post_hocs = pd.concat([post_hocs, (robustness_fc
                            .groupby(['architecture'])
                            .apply(lambda df: pg.pairwise_tests(
                                dv='value', within=['task', 'dataset'],
                                subject='subject', data=df, padjust='bonf'))
                            .reset_index())])
                        success = True
                    except:
                        pass
                if not success:
                    try:
                        post_hocs = pd.concat([post_hocs, (robustness_fc
                            .groupby(['architecture', 'task'])
                            .apply(lambda df: pg.pairwise_tests(
                                dv='value', within='dataset', subject='subject',
                                data=df, padjust='bonf'))
                            .reset_index())])
                        success = True
                    except:
                        pass
                if success:
                    post_hocs.to_csv(ph_path)

            """
            # performance curves
            outpath = f'{out_dir}/{metric}_curves.pdf'
            if not op.isfile(outpath) or overwrite:

                fig, ax = plt.subplots(figsize=(4, 4))
                curve_x = np.linspace(0, 1, 1000)
                xvals = VISIBILITIES[exp] + [1]

                # plot human performance
                if metric == 'accuracy':
                    # curve function
                    popt = curves_h[
                        (curves_h.metric == 'accuracy') &
                        (curves_h.subject == 'group') &
                        (curves_h.occluder_class == 'all') &
                        (curves_h.occluder_color == 'all')][
                        ['L', 'x0', 'k', 'b']].values[0]
                    curve_y = sigmoid(curve_x, *popt)
                    ax.plot(curve_x, curve_y, color='tab:gray')

                    # plot human data
                    yvals = (robustness_h.drop(columns='subject')
                             .groupby('visibility')
                             .agg('mean',
                                  numeric_only=True).accuracy.to_list())
                    ax.scatter(xvals, yvals, color='tab:gray', s=2)

                # models
                curves_m_last_cycle = (curves_m.groupby(['model', 'layer'])
                .apply(
                    lambda d: d[d.cycle == d.cycle.max()]))
                for m, (label, info) in enumerate(config.items()):
                    color = info['color']

                    # plot model curve function
                    popt = (curves_m_last_cycle[
                                (curves_m_last_cycle.model == label) &
                                (curves_m_last_cycle.metric == metric) &
                                (
                                            curves_m_last_cycle.occluder_class == 'all') &
                                (
                                            curves_m_last_cycle.occluder_color == 'all')]
                            [['L', 'x0', 'k', 'b']].values)

                    if not len(popt):
                        continue
                    assert len(popt) == 1, 'wrong number of rows selected'
                    curve_y = sigmoid(curve_x, *popt[0])
                    ax.plot(curve_x, curve_y, color=color)

                    # plot model performance
                    yvals = (robustness_m_last_cycle[
                                 robustness_m_last_cycle['model'] == label]
                             .groupby('visibility').agg('mean',
                                                        numeric_only=True)
                             [metric].to_list())
                    ax.scatter(xvals, yvals, color=color, s=2)

                # format plot
                # ax.set_title(EXP1.occluder_labels[o], size=7)
                ax.set_xticks(xvals)
                ax.set_xlim((0, 1.02))
                ax.set_yticks(plot_config['yticks'])
                ax.set_ylim(plot_config['ylims'])
                ax.tick_params(axis='both', which='major', labelsize=7)
                if plot_config['chance']:
                    ax.axhline(y=plot_config['chance'], color='k',
                               ls='dotted')
                ax.set_xlabel('visibility')
                ax.set_ylabel(metric)
                plt.tight_layout()
                fig.savefig(outpath)
                plt.close()
                """

        # human likeness plots
        plot_configs = {
            'condition-wise': {
                'pearson_r': {
                    'title': 'accuracy similarity to humans\n(condition-wise)',
                    'ylabel': r"Pearson's $\it{r}$",  #r"correlation ($\it{r}$)"
                    'yticks': np.arange(-2, 2, .5),
                    'ylims': (-.7, 1),
                    'chance': 0}},
            'trial-wise': {
                'c_inacc': {
                    'title': 'inaccurate consistency with humans\n(trial-wise)',
                    'ylabel': 'inaccurate consistency',
                    'yticks': np.arange(0, 1, .1),
                    'ylims': (0, .5),
                    'chance': 1 / 7},
                'c_obs': {
                    'title': 'observed consistency with humans\n(trial-wise)',
                    'ylabel': 'observed consistency',
                    'yticks': np.arange(0, 2, .5),
                    'ylims': (0, 1)},
                'c_err': {
                    'title': 'error consistency with humans\n(trial-wise)',
                    'ylabel': "Cohen's $\\kappa$",
                    'yticks': np.arange(0, 1, .2),
                    'ylims': (-.1, .6)}}}

        likeness.groupby(['analysis', 'metric', 'metric_sim', 'level']).apply(
            _human_likeness_plots, model_config, plot_configs, noise_ceiling,
                         results_dir, overwrite)


    # save occluder legend separately
    outpath = f'{op.dirname(results_dir)}/occluder_types_legend.pdf'
    if not op.isfile(outpath) or overwrite:
        make_legend(
            outpath=outpath,
            labels=[f'{o} {t}' for o, t in itp(
                EXP1.occluder_labels, OCC_COLORS)],
            markers='o',
            colors=EXP1.plot_colors,
            markeredgecolors=None,
            linestyles='None')

    # save occluder legend with unoccluded bar separately
    outpath = f'{op.dirname(results_dir)}/occluder_types_legend_w-unocc.pdf'
    if not op.isfile(outpath) or overwrite:
        make_legend(
            outpath=outpath,
            labels=['unoccluded'] + [f'{o} {t}' for o, t in itp(
                EXP1.occluder_labels, OCC_COLORS)],
            markers='o',
            colors=['w'].extend(EXP1.plot_colors),
            markeredgecolors=['k'] + [None] * 18,
            linestyles='None')


def compare_models_exp2(overwrite=False):

    def _human_likeness_plots(df, model_config, plot_configs, noise_ceiling,
                             results_dir, overwrite):

        analysis, metric, level = df.name
        out_dir = f'{results_dir}/human_likeness/{analysis}'
        os.makedirs(out_dir, exist_ok=True)

        nc = noise_ceiling[noise_ceiling.analysis == analysis][[
            'lwr', 'upr']].values[0]
        level = 'occluder_class_x_occluder_color'

        # plot all cycles (barplot)
        plot_file = f'{out_dir}/{level.replace("*", "-")}_barplot_cyclewise.pdf'
        if df.cycle.max() > 0 and metric in plot_configs[analysis] and (
                not op.isfile(plot_file) or overwrite):
            bar_plot_multi_cycle(model_config, df, 'value',
                                 plot_configs[analysis][metric], plot_file)

        # final cycle
        plot_file = f'{out_dir}/{level.replace("*", "-")}_barplot.pdf'
        if metric in plot_configs[analysis] and (
                not op.isfile(plot_file) or overwrite):
            df_plot = (df
                       .groupby(['subject', 'model', 'layer'])
                       .apply(lambda d: d[d.cycle == d.cycle.max()])
                       .reset_index(drop=True))
            plot_config = plot_configs[analysis][metric]
            samples = {'column': 'subject',
                       'color': HUMAN_COLOR,
                       'size': 3,
                       'alpha': .5}
            bar_plot_single_cycle(
                model_config=model_config,
                df=df_plot,
                metric='value',
                samples=samples,
                plot_config=plot_config,
                plot_file=plot_file,
                noise_ceiling=nc)

            # ANOVA
            df_plot['architecture'] = (
                df_plot['model'].apply(lambda r: r.split(', ')[0]))
            df_plot['dataset'] = (
                df_plot['model'].apply(lambda r: r.split(', ')[1]))
            df_plot['task'] = (
                df_plot['model'].apply(lambda r: r.split(', ')[2]))
            anova = pg.rm_anova(
                dv='value', within=['task', 'dataset'],
                subject='subject', data=df_plot,
                detailed=True)
            anova.to_csv(f'{out_dir}/{metric}_anova.csv')

            post_hocs = pd.DataFrame()
            try:
                post_hocs = pd.concat([post_hocs, pg.pairwise_tests(
                    dv='value', within=['architecture', 'task', 'dataset'],
                    subject='subject', data=df_plot, padjust='bonf')])
                success = True
            except:
                success = False
            if not success:
                try:
                    post_hocs = pd.concat([post_hocs, (df_plot
                        .groupby(['architecture'])
                        .apply(lambda df: pg.pairwise_tests(
                            dv='value', within=['task', 'dataset'],
                            subject='subject', data=df, padjust='bonf'))
                        .reset_index())])
                    success = True
                except:
                    pass
            if not success:
                try:
                    post_hocs = pd.concat([post_hocs, (df_plot
                        .groupby(['architecture', 'task'])
                        .apply(lambda df: pg.pairwise_tests(
                            dv='value', within='dataset', subject='subject',
                            data=df, padjust='bonf'))
                        .reset_index())])
                    success = True
                except:
                    pass
            if success:
                post_hocs.to_csv(f'{out_dir}/{metric}_posthocs.csv')

            # save performance profile
            summary = (df_plot
                .drop(columns='subject')[['model', 'value']]
                .groupby('model').agg(['mean', 'sem'], numeric_only=True)
                .reset_index())
            summary.columns = ['model', 'mean', 'sem']
            summary = pd.concat([
                summary, pd.DataFrame({
                    'model': ['nc_lwr', 'nc_upr'],
                    'mean': nc,
                    'sem': [np.nan, np.nan]})])
            summary.to_csv(f'{out_dir}/{metric}_summary.csv')

    for (model_contrast, model_config), transfer_method in \
            itp(model_contrasts.items(), ['SVC', 'output']):

        print(f'{now()} | Comparing models (exp2) | {model_contrast} | {transfer_method}')
        config = complete_model_config(model_config.copy())
        results_dir = (f'../data/in_silico/analysis/'
                       f'{model_contrast}/behavior/exp2/{transfer_method}')
        trials_h, trials_m, trials_m_rt, likeness, noise_ceiling\
            = collate_data_exp2(model_config, model_contrast, transfer_method)

        # occlusion robustness measures
        plot_configs = {
            'condition-wise': {
                'accuracy': {
                    'title': 'visibility threshold for\n'
                             '50% classification accuracy',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert_y': True},
                'entropy': {
                    'title': 'visibility threshold for\nentropy < .5',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert_y': True},
                'true_class_prob': {
                    'title': 'visibility threshold for\n'
                             '50% confidence in true class',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert_y': True}},
            'trial-wise': {
                'accuracy': {
                    'title': 'visibility at first accurate response',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert_y': True},
                'entropy': {
                    'title': 'visibility at first entropy < .5',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert_y': True},
                'true_class_prob': {
                    'title': 'visibility at first confidence > 50%',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert_y': True}}
        }

        for analysis, metric in itp(plot_configs, METRICS):

            out_dir = f'{results_dir}/occlusion_robustness/{analysis}'
            os.makedirs(out_dir, exist_ok=True)

            # plot all cycles (barplot)
            plot_file = f'{out_dir}/{metric}_barplot_cyclewise.pdf'
            if trials_m.cycle.max() > 0 and \
                    (not op.isfile(plot_file) or overwrite):

                groupbys = ['group', 'model', 'layer', 'cycle']
                data_human = (trials_h[trials_h.accuracy == 1]
                    [groupbys + ['visibility']]
                    .groupby(groupbys)
                    .visibility.mean()
                    .reset_index())
                data_human.rename(columns={'visibility': 'value'}, inplace=True)

                if analysis == 'condition-wise':
                    data_model = (trials_m
                        [groupbys + ['visibility', metric]]
                        .groupby(groupbys)
                        .apply(lambda d: d[d[metric] > .5].visibility.min())
                        .reset_index()
                        .rename(columns={0: 'value'}))

                else:  # elif analysis == 'trial-wise':
                    data_model = (trials_m_rt[
                        (trials_m_rt.metric == metric) &
                        (trials_m_rt.accuracy == 1)]
                        .groupby(groupbys)
                        .agg({'visibility': 'mean'})
                        .reset_index()
                        .rename(columns={'visibility': 'value'}))

                robustness = pd.concat([data_human, data_model])
                bar_plot_multi_cycle(config, robustness, 'value',
                    plot_configs[analysis][metric], plot_file)

            # final cycle only
            plot_file = f'{out_dir}/{metric}_barplot.pdf'
            if not op.isfile(plot_file) or overwrite:
                groupbys = ['group', 'model', 'layer', 'occluder_class', 'occluder_color']
                data_human = (trials_h[trials_h.accuracy == 1]
                    .rename(columns={'visibility': 'value'}))

                if analysis == 'condition-wise':
                    data_model = (trials_m
                        [groupbys + ['cycle', 'visibility', metric]]
                        .groupby(groupbys)
                        .apply(lambda d: d[d.cycle == d.cycle.max()])
                        .reset_index(drop=True)
                        .groupby(groupbys)
                        .apply(lambda d: d[d[metric] > .5].visibility.min())
                        .reset_index()
                        .rename(columns={0: 'value'}))
                    data_model.value.fillna(1, inplace=True)

                else:  # elif analysis == 'trial-wise':
                    data_model = (trials_m_rt[
                        (trials_m_rt.metric == metric) &
                        (trials_m_rt.accuracy == 1)]
                        .groupby(groupbys)
                        .apply(lambda d: d[d.cycle == d.cycle.max()])
                        .reset_index(drop=True)
                        .groupby(groupbys)
                        .agg({'visibility': 'mean'})
                        .reset_index()
                        .dropna()
                        .rename(columns={'visibility': 'value'}))

                robustness_fc = pd.concat([data_human, data_model])
                samples = {'column': ['occluder_class', 'occluder_color'],
                           'color': EXP2.plot_colors, 'size': 5, 'alpha': 1}
                bar_plot_single_cycle(
                    model_config=config,
                    df=robustness_fc,
                    metric='value',
                    samples=samples,
                    plot_config=plot_configs[analysis][metric],
                    plot_file=plot_file)

                """
                # performance curves
                if analysis == 'condition-wise':
                    outpath = f'{out_dir}/{metric}_curves.pdf'
                    if not op.isfile(outpath) or overwrite:

                        fig, ax = plt.subplots(figsize=(4, 4))
                        curve_x = np.linspace(0, 1, 1000)
                        xvals = VISIBILITIES[exp]

                        # models
                        curves_m_last_cycle = (curves_m.groupby('model')
                        .apply(
                            lambda d: d[d.cycle == d.cycle.max()]))
                        for m, (label, info) in enumerate(config.items()):
                            color = info['color']

                            # plot model curve function
                            popt = curves_m_last_cycle[
                                (curves_m_last_cycle.metric == metric) &
                                (curves_m_last_cycle.model == label) &
                                (curves_m_last_cycle.occluder_class == 'all') &
                                (curves_m_last_cycle.occluder_color == 'all') &
                                (curves_m_last_cycle.stimulus_set == 'all')][
                                ['L', 'x0', 'k', 'b']].values
                            assert len(
                                popt) == 1, 'wrong number of rows selected'
                            curve_y = sigmoid(curve_x, *popt[0])
                            ax.plot(curve_x, curve_y, color=color,
                                    clip_on=False)

                            # plot model performance
                            yvals = (trials_m_last_cycle[
                                         (trials_m_last_cycle.model == label) &
                                         (trials_m_last_cycle.metric == metric)]
                                     .groupby('visibility')
                                     .agg('mean', numeric_only=True)
                                     .sort_values(by='visibility')
                                     .value.to_list())
                            ax.scatter(xvals, yvals, color=color, s=2)

                        # format plot
                        # ax.set_title(EXP1.occluder_labels[o], size=7)
                        ax.set_xticks((0, 1))
                        ax.set_xlim((-.02, 1.02))
                        ax.set_yticks(yticks)
                        ax.set_ylim(ylims)
                        ax.tick_params(axis='both', which='major', labelsize=7)
                        ax.axhline(y=chance, color='k', linestyle='dotted')
                        ax.set_xlabel('visibility')
                        ax.set_ylabel(metric)
                        plt.tight_layout()
                        fig.savefig(outpath)
                        plt.close()
                        """


        # human likeness measures
        plot_configs = {
            'condition-wise': {
                'accuracy': {
                    'title': 'accuracy similarity to humans\n(condition-wise)',
                    'ylabel': r"Pearson's $\it{r}$",#"correlation ($\it{r}$)",
                    'yticks': np.arange(-1, 2, .5),
                    'ylims': (-.5, 1)}},
            # 'entropy': {
            #    'title': 'entropy similarity to humans\n(condition-wise)',
            #    'ylabel': "correlation ($\it{r}$)",
            #    'yticks': np.arange(-1, 2, .2),
            #    'ylims': (-1, 1)},
            # 'true_class_prob': {
            #    'title': 'true class probability similarity to humans\n'
            #             '(condition-wise)',
            #    'ylabel': "correlation ($\it{r}$)",
            #    'yticks': np.arange(-1, 2, .2),
            #    'ylims': (0, 1)}},
            'trial-wise': {
                'accuracy': {
                    'title': 'RT similarity to humans\n'
                             '(trial-wise, accuracy-based)',
                    'ylabel': r"Pearson's $\it{r}$",#"correlation ($\it{r}$)",
                    'yticks': np.arange(-1, 2, .5),
                    'ylims': (-.5, 1)}}}
            # 'entropy': {
            #    'title': 'RT similarity to humans\n'
            #             '(trial-wise, entropy-based)',
            #    'ylabel': "correlation ($\it{r}$)",
            #    'yticks': np.arange(-1, 2, .2),
            #    'ylims': (0, 1)},
            # 'true_class_prob': {
            #    'title': 'RT similarity to humans\n'
            #             '(trial-wise, true class probability-based)',
            #    'ylabel': "correlation ($\it{r}$)",
            #    'yticks': np.arange(-1, 2, .2),
            #    'ylims': (0, 1)}}}

        likeness.groupby(['analysis', 'metric', 'level']).apply(
            _human_likeness_plots, model_config, plot_configs, noise_ceiling,
            results_dir, overwrite)

        # legend
        legend_file = f'{out_dir}/legend.pdf'
        if model_contrast in effect_colors:
            leg_labels = list(effect_colors[model_contrast].keys())
            leg_colors = list(effect_colors[model_contrast].values())
        else:
            leg_colors = [m['color'] for m in config.values()]
            leg_labels = list(config.keys())
        if not op.isfile(legend_file) or overwrite:
            make_legend(
                outpath=legend_file,
                labels=leg_labels,
                markers='s',
                colors=leg_colors,
                markeredgecolors=None,
                linestyles='None')

        # panel legend with one label per dataset
        if model_contrast == 'CCN_2024_poster':
            legend_file = f'{out_dir}/legend_panel.pdf'
            if 'group' in likeness.columns and (
                    not op.isfile(legend_file) or overwrite):
                leg_colors = []
                leg_labels = []
                for k, v in config.items():
                    if v['color'] not in leg_colors:
                        leg_colors.append(v['color'])
                        try:
                            leg_labels.append(k.split(', ')[-2])
                        except:
                            leg_labels.append(k)
                make_legend(
                    outpath=legend_file,
                    labels=leg_labels,
                    markers='s',
                    colors=leg_colors,
                    markeredgecolors=None,
                    linestyles='None')

    # save occluder legend separately
    outpath = f'{op.dirname(results_dir)}/occluder_types_legend.pdf'
    if not op.isfile(outpath) or overwrite:
        make_legend(
            outpath=outpath,
            labels=[f'{o} {t}' for o, t in itp(EXP2.occluder_labels,
                                               OCC_COLORS)],
            markers='o',
            colors=EXP2.plot_colors,
            markeredgecolors=None,
            linestyles='None')


def compare_models(overwrite=False):

    compare_models_exp1(overwrite)
    compare_models_exp2(overwrite)


if __name__ == "__main__":

    from seconds_to_text import seconds_to_text

    start = time.time()
    compare_models(overwrite=False)
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')
