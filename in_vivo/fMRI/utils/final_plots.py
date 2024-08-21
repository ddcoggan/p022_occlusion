# /usr/bin/python
# Created by David Coggan on 2023 06 23

"""
script for mkaing final plots for manuscript
"""

import os
import os.path as op
import itertools
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import sys

sys.path.append(os.path.expanduser("~/david/master_scripts"))
from misc.plot_utils import export_legend, custom_defaults, make_legend
plt.rcParams.update(custom_defaults)

from .config import CFG, PROJ_DIR
from .RSA import RSA

figdir = f'{PROJ_DIR}/in_vivo/fMRI/figures'
space = 'standard'
norm = 'all-conds'
norm_method = 'z-score'
similarity = 'pearson'
similarity_label = CFG.similarities[similarity]
region_set = 'EVC_ventral'
regions = CFG.region_sets[region_set]

def main(overwrite=False):

    make_ROI_plots(overwrite=overwrite)
    make_RSM_model_plots(overwrite=overwrite)
    for exp in ['exp1','exp2']:
        tasks = [task for task in CFG.scan_params[exp] if 'occlusion' in task]
        for task in tasks:
            make_RSM_plots(exp, task, overwrite=True)
            make_MDS_plots(exp, task, overwrite=False)
            make_condwise_plots(exp, task, overwrite=False)
            make_index_plots(exp, task, overwrite=False)


def RGB_tuple_to_string(colour):
    return f'{colour[0]},{colour[1]},{colour[2]}'

def make_ROI_plots(overwrite=False):

    hemis = ['lh', 'rh']
    views = ['inferior']
    fs_dir = f'{os.environ["SUBJECTS_DIR"]}/fsaverage'
    out_dir = f'{figdir}/ROIs'
    os.makedirs(out_dir, exist_ok=True)

    for ROI_set, hemi, view in itertools.product(
            ['EVC_ventral'], hemis, views):

        regions = CFG.region_sets[ROI_set]
        cmap = matplotlib.colormaps['plasma'].colors
        colours = [cmap[c] for c in np.linspace(0,255,len(regions), dtype=int)]

        image_path = (f'{out_dir}/{ROI_set}_{view}_{hemi}.png')
        if not op.isfile(image_path) or overwrite:

            cmd_plot = (f'freeview -f '
                        f'{fs_dir}/surf/{hemi}.inflated'
                        f':curvature_method=binary')

            for region, colour in zip(regions, colours):

                # get existing surface labels
                if not 'ventral' in region:
                    region_wang = CFG.regions[region]
                    if region in ['V1','V2','V3']:
                        subregions = [f'{region_wang}d', f'{region_wang}v']
                    else:
                        subregions = [region_wang]
                    labels = [f'{fs_dir}/surf/{hemi}.wang15_mplbl.{reg}.label' \
                              for reg in subregions]

                # make remaining surface labels
                else:
                    volume = (f'derivatives/ROIs/MNI152_2mm/'
                              f'{ROI_set}/{region}.nii.gz')
                    surface = f'{volume[:-7]}_{hemi}.mgh'
                    if not op.isfile(surface):
                        cmd = (f'mri_vol2surf '
                               f'--mov {volume} '
                               f'--out {surface} '
                               f'--regheader fsaverage '
                               f'--hemi {hemi}')
                        os.system(cmd)
                    label = f'{volume[:-7]}_{hemi}.label'
                    if not op.isfile(label):
                        cmd = (f'mri_cor2label '
                               f'--i {surface} '
                               f'--surf fsaverage {hemi} '
                               f'--id 1 '
                               f'--l {label}')
                        os.system(cmd)
                    labels = [label]


                col_str = ','.join([str(int(255 * c)) for c in colour])
                for label in labels:
                    cmd_plot += f':label={label}:label_color={col_str}'

            cmd_plot += (f' -layout 1 -viewport 3d -view {view} '
                         f'-ss {image_path} 2 autotrim')
            os.system(cmd_plot)


def make_RSM_model_plots(overwrite=False):

    out_dir = f'{figdir}/RSMs'
    os.makedirs(out_dir, exist_ok=True)
    for model_label, model in CFG.RSM_models['matrices'].items():
        outpath = f'{out_dir}/model_{model_label}.pdf'
        if not op.isfile(outpath) or overwrite:
            RSA(RSM=model).plot_RSM(cmap='cividis', vmin=0, vmax=1,
                                    fancy=True,
                                    title=f'model: {model_label}',
                                    outpath=outpath)

    colours = ['white'] + CFG.occlusion_robustness_analyses[
        'object_completion']['colours']
    colours += CFG.occlusion_robustness_analyses[
        'occlusion_invariance']['colours']
    cmap = matplotlib.colors.ListedColormap(colours)
    for label, contrast_mat in CFG.contrast_mats.items():
        outpath = f'{out_dir}/contrasts_{label}.pdf'
        if not op.isfile(outpath) or overwrite:
            RSA(RSM=contrast_mat).plot_RSM(vmin=0, vmax=8,
                                           title=f'contrasts',
                                           cmap=cmap, outpath=outpath,
                                           fancy=True)


def make_RSM_plots(exp, task, overwrite=False):

    out_dir = f'{figdir}/RSMs'
    os.makedirs(out_dir, exist_ok=True)
    for region in ['V1', 'V2', 'hV4', 'ventral_stream_sub_ret']:
        outpath = f'{out_dir}/{exp}_{task}_{CFG.regions[region]}.pdf'
        if not op.isfile(outpath) or overwrite:
            RSMs = pkl.load(
                open(f'{PROJ_DIR}/in_vivo/fMRI/{exp}/derivatives/RSA/'
                     f'{task}_space-{space}/norm-{norm}_{norm_method}/'
                     f'{similarity}/RSA.pkl', 'rb'))
            print(f'Plotting RSM...')
            RSM = RSMs[region]['group'].RSM
            imx = f'{PROJ_DIR}/in_vivo/fMRI/RSM_pictures_x.png'
            imy = f'{PROJ_DIR}/in_vivo/fMRI/RSM_pictures_y.png'
            picx = plt.imread(imx)
            picy = plt.imread(imy)

            fig, unused_ax = plt.subplots(figsize=(7.5, 5.25))
            unused_ax.axis('off')
            ax = fig.add_axes([.1, .22, .75, .75])
            im = ax.imshow(RSM, vmin=-.3, vmax=.3, cmap='rainbow')
            ax.tick_params(**{'length': 0})
            ax.set_xticks(np.arange(RSM.shape[0]))
            ax.set_yticks(np.arange(RSM.shape[1]))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(direction='in')
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            cbar = fig.colorbar(im, fraction=0.0453, pad=0.1, ax=ax)
            cbar.set_ticks([-.3, 0, .3], labels=['-0.3', '0.0',
                           '0.3'], fontsize=16)
            ax.set_title('')
            plt.text(25, 15, "Correlation ($\it{r}$)", rotation='vertical',
                     fontsize=16)
            newax = fig.add_axes([-.27, 0.2, .79, .79])
            newax.imshow(picy)
            newax.tick_params(**{'length': 0})
            newax.spines['bottom'].set_visible(False)
            newax.spines['left'].set_visible(False)
            newax.set_xticks([])
            newax.set_yticks([])
            hw_ratio = (.79*(5.25/7.5))
            newax2 = fig.add_axes([.2, -0.19, hw_ratio, hw_ratio+.02])
            newax2.imshow(picx)
            newax2.tick_params(**{'length': 0})
            newax2.spines['bottom'].set_visible(False)
            newax2.spines['left'].set_visible(False)
            newax2.set_xticks([])
            newax2.set_yticks([])
            plt.savefig(outpath)
            plt.close()


def make_MDS_plots(exp, task, overwrite=False):
    out_dir = f'{figdir}/MDS'
    os.makedirs(out_dir, exist_ok=True)
    for region in ['V1', 'ventral_stream_sub_ret']:
        outpath = f'{out_dir}/{exp}_{task}_{CFG.regions[region]}.pdf'
        if not op.isfile(outpath) or overwrite:
            print(f'Plotting MDS...')
            RSMs = pkl.load(
                open(f'{PROJ_DIR}/in_vivo/fMRI/{exp}/derivatives/RSA/'
                     f'{task}_space-{space}/norm-{norm}_{norm_method}/'
                     f'{similarity}/RSA.pkl', 'rb'))
            RSMs[region]['group'].plot_MDS(outpath=outpath)


def make_condwise_plots(exp, task, overwrite=False):

    out_dir = f'{figdir}/condwise_sims'
    os.makedirs(out_dir, exist_ok=True)
    figsize = (4.2, 2.5) if exp == 'exp1' else (4.5, 2)  # figure size

    # conditions-wise similarities
    for analysis, params in CFG.occlusion_robustness_analyses.items():

        df_summary = pd.read_csv(
            f'{PROJ_DIR}/in_vivo/fMRI/{exp}/derivatives/RSA/'
            f'{task}_space-{space}/norm-{norm}_{norm_method}/'
            f'{similarity}/{analysis}/cond-wise_sims_summary.csv')
        os.makedirs(out_dir, exist_ok=True)

        # region-wise plot for each region_set
        outpath = f'{out_dir}/{exp}_{task}_{analysis}.pdf'
        if not op.isfile(outpath) or overwrite:

            plot_df = df_summary[
                (df_summary.analysis == analysis) &
                (df_summary.region.isin(regions))].copy(deep=True)
            reg_ord = [r for r in regions if r in plot_df.region.unique()]
            plot_df.region = pd.Categorical(
                plot_df.region, categories=reg_ord, ordered=True)
            plot_df.level = pd.Categorical(
                plot_df.level, categories=params['conds'], ordered=True)
            plot_df.sort_values(['region', 'level'], inplace=True)
            ylabel = similarity_label

            if exp == 'exp1':
                yticks = np.arange(0,1,.1)
                if analysis == 'object_completion':
                    ylims = (-.025, .23)
                else:
                    ylims = (-.04, .31)
            else:
                yticks = np.arange(0, 1, .1)
                if analysis == 'object_completion':
                    ylims = (-.02, .15)
                else:
                    ylims = (-.04, .25)

            x_tick_labels = [CFG.regions[r] for r in reg_ord]

            # all conditions as bars with errors
            df_means = plot_df.pivot(index='region', columns='level', values='value')
            df_sems = plot_df.pivot(index='region', columns='level',
                               values='error').values
            fig, ax = plt.subplots(figsize=figsize)
            df_means.plot(
                kind='bar',
                ylabel=ylabel,
                yerr=df_sems.transpose(),
                rot=0,
                figsize=figsize,
                color=params['colours'],
                legend=False,
                width=.8,
                ax=ax)
            plt.yticks(yticks)
            plt.ylim(ylims)
            plt.xlabel(None)
            plt.xticks(np.arange(len(x_tick_labels)), x_tick_labels)
            ax.xaxis.set_ticks_position('none')  # no ticks
            ax.spines['bottom'].set_color('none')  # no x-axis
            plt.tight_layout()
            fig.savefig(outpath)
            plt.close()

        # legend
        outpath = f'{out_dir}/legend_{analysis}.png'
        if not op.isfile(outpath) or overwrite:
            f = lambda m, c: \
                plt.plot([], [], marker=m, markerfacecolor=c, color='white')[0]
            handles = [f('s', colour) for colour in params['colours']]
            legend = plt.legend(handles, params['labels'], loc=3)
            export_legend(legend, filename=outpath)
            plt.close()


def make_index_plots(exp, task, overwrite=False):


    out_dir = f'{figdir}/indices'
    os.makedirs(out_dir, exist_ok=True)
    figsize = (4.2, 2.5) if exp == 'exp1' else (4.5, 2) # figure size
    yticks = (0, .5, 1)
    ytick_labels = ['0.0', '0.5', '1.0']

    for analysis, params in CFG.occlusion_robustness_analyses.items():

        outpath = f'{out_dir}/{exp}_{task}_{analysis}.pdf'
        if exp == 'exp1' or analysis == 'occlusion_invariance':
            ylims = (0, 1.1)
        else:
            ylims = (0, 1.58)
        if not op.isfile(outpath) or overwrite:

            # data
            RSA_dir = (f'{PROJ_DIR}/in_vivo/fMRI/{exp}/derivatives/RSA/'
                f'{task}_space-{space}/norm-{norm}_{norm_method}/'
                f'{similarity}/{analysis}')
            df = pd.read_csv(f'{RSA_dir}/indices.csv')
            plot_df = df[
                (df.level == 'group') &
                (df.subtype == 'norm') &
                (df.region.isin(regions))].copy(
                deep=True).reset_index(drop=True).drop(columns=[
                'level', 'subtype', 'subject'])
            plot_df['error'] = np.nan

            reg_ord = [r for r in regions if
                       r in plot_df.region.unique()]
            plot_df.region = pd.Categorical(
                plot_df.region, categories=reg_ord, ordered=True)
            plot_df.sort_values('region', inplace=True)

            # remove unqualified ROIs and set significance bars
            ph = pd.read_csv(f'{RSA_dir}/{region_set}/cond-wise_sims_posthocs.csv')
            ceil_sigs, floor_sigs, batches, b = [], [], [], []
            for r, region in enumerate(reg_ord):

                if region in ph.region.unique():
                    floor_sig = ph['p-corr'][
                        (ph.region == region) &
                        (ph.A == params['conds'][3]) &
                        (ph.B == params['conds'][1])].item() < .05
                    ceil_sig = ph['p-corr'][
                        (ph.region == region) &
                        (ph.A == params['conds'][1]) &
                        (ph.B == params['conds'][0])].item() < .05
                    #floor_v_ceil = ph['p-corr'][
                    #    (ph.region == region) &
                    #    (ph.A == params['conds'][3]) &
                    #    (ph.B == params['conds'][0])].item() < .05
                    #floor_v_ceil_lib = ph['p-corr'][
                    #    (ph.region == region) &
                    #    (ph.A == params['conds'][3]) &
                    #    (ph.B == params['conds'][0])].item() < .1
                    floor_sigs.append(floor_sig)
                    ceil_sigs.append(ceil_sig)
                    if exp == 'exp1' or region not in [
                            'VO1','VO2','PHC1','PHC2']:
                        b.append(r)
                    else:
                        if len(b):
                            batches.append(b)
                            b = []
                else:
                    if len(b):
                        batches.append(b)
                        b = []
            if len(b):
                batches.append(b)


            # plot
            x_tick_labels = [CFG.regions[r] for r in reg_ord]
            n_x_ticks = len(x_tick_labels)
            os.makedirs(op.dirname(outpath), exist_ok=True)
            fig, ax = plt.subplots(figsize=figsize)
            for batch in batches:
                ax.plot(batch,
                        plot_df['value'].values[batch],
                        color=params['colours'][1],
                        marker='o')
            for x_pos in range(n_x_ticks):
                xmin = (x_pos + .2) / n_x_ticks
                xmax = xmin + (.6 / n_x_ticks)
                if x_pos in [b for batch in batches for b in batch]:
                    if floor_sigs[x_pos]:
                        ax.axhline(y=.08, xmin=xmin, xmax=xmax, lw=1,
                                   color=params['colours'][3])
                    if ceil_sigs[x_pos]:
                        ax.axhline(y=.92, xmin=xmin, xmax=xmax, lw=1,
                                   color=params['colours'][0])
                #else:
                    #ax.axhline(y=.25, xmin=xmin, xmax=xmax, lw=1,
                    #           color='#d3d3d3')

            lwr, upr = 1., max(ylims)
            cl_x = (-.5, n_x_ticks - .5)
            ax.fill_between(cl_x, lwr, upr, color='#e4e4e4', lw=0)
            ax.set_xticks(np.arange(n_x_ticks), labels=x_tick_labels)
            if exp == 'exp2':
                for label in ax.get_xticklabels()[4:8]:
                    label.set_color('#d3d3d3')
            ax.set_yticks(yticks, labels=ytick_labels)
            ax.set_ylabel(params['index_label'])
            ax.set_ylim(ylims)
            #ax.set_title(f'{analysis.replace("_", " ")} index')
            ax.set_xlim((-.5, n_x_ticks - .5))
            plt.tight_layout()
            plt.savefig(outpath, dpi=300)
            plt.close()

        # legend
        outpath = f'{out_dir}/{analysis}_legend.pdf'
        if not op.isfile(outpath) or overwrite:
            labels = [
                'index < ceiling (p corr. < .05)',
                'index > floor (p corr. < .05)',
                #r'ceiling $\approx$ floor',
                #'index not calculated'
            ]
            markers = [None] * 3
            colors = [params['colours'][0], params['colours'][3], '#d3d3d3']
            markeredgecolors = [None] * 3
            linestyles = ['solid'] * 3
            make_legend(outpath, labels, markers, colors, markeredgecolors,
                        linestyles)


if __name__ == "__main__":
   main()