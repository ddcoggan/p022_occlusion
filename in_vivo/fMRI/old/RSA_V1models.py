import os
import os.path as op
import sys
import glob
import time
import pickle as pkl
import numpy as np
from types import SimpleNamespace
import shutil
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy.stats import kendalltau, sem, gamma
from frrsa import frrsa
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import gc
from torch.profiler import profile, record_function, ProfilerActivity
from scipy import optimize


sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from plot_utils import make_legend, custom_defaults
plt.rcParams.update(custom_defaults)

from in_vivo.fMRI.utils import CFG as fMRI
from in_vivo.fMRI.utils import RSA_dataset

sys.path.append(op.expanduser('~/david/repos/vonenet'))
from vonenet.vonenet import VOneNet

PROJ_DIR = op.expanduser('~/david/projects/p022_occlusion')
FMRI_DIR = op.join(PROJ_DIR, 'data/in_vivo/fMRI')
SIMILARITY = 'pearson'
NORM = 'all-conds'
NORM_METHOD = 'z-score'
ylabel = "Correlation ($\it{r}$)"
IMAGE_SIZE_PIX = 224
IMAGE_SIZE_DEG = 9
PPD = IMAGE_SIZE_PIX / IMAGE_SIZE_DEG
REPS = 64
FIX_STDS_DEG = [0.291409, 0.197803]  # [0, .125, .25, .5, 1]
NUMS_FILTERS = [512, 32]  # 512
VIS_ANGS = [3.2]  # [36, 18, 9, 4.5, 2.25]
KERNEL_SIZES = [99]  # [7, 15, 29, 59, 119]
MEAN_V1_FWHM_DEG = 0.7396

fix_stds_labels = [str(f) + r"$\degree$" for f in FIX_STDS_DEG]
vis_angs_labels = [str(v) + r"$\degree$" for v in VIS_ANGS]
max_human_prf_sigma_pix = .3658 * 224 / 9
vis_ang_ratios = [.25, .5, 1, 2, 4]
target_rfs_sigma_pix = [max_human_prf_sigma_pix * e for e in
                        vis_ang_ratios]  # [51*5]


def main(overwrite=False):
    for num_filters in NUMS_FILTERS:
        analysis_dir = f'{FMRI_DIR}/V1_models/{num_filters}_filters'
        os.makedirs(analysis_dir, exist_ok=True)
        os.chdir(analysis_dir)

        calculate_visual_angle(num_filters, overwrite=True)
        evaluate_models(VIS_ANGS, KERNEL_SIZES, num_filters, overwrite=False)
        # compare_single_var(overwrite=False)
        compare_fix_std_vis_ang(overwrite=True)


# code for visualizing receptive fields
def gaussian(x, amplitude, xo, sigma, offset):
    xo = float(xo)
    a = 1 / (2 * sigma ** 2)
    g = offset + amplitude * np.exp(-a * ((x - xo) ** 2))
    return g


def circular_gaussian(xy, amplitude, xo, yo, sigma, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma ** 2) + (np.sin(theta) ** 2) / (
                2 * sigma ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma ** 2) + (np.sin(2 * theta)) / (
                4 * sigma ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma ** 2) + (np.cos(theta) ** 2) / (
                2 * sigma ** 2)
    g = offset + amplitude * np.exp(
        - (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
           + c * ((y - yo) ** 2)))
    return g.ravel()


def fit_gaussian(filter: np.array) -> SimpleNamespace:
    w, h = filter.shape
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    initial_guess = (filter.max(), w // 2, h // 2, 1, 0, 0)
    popt, pcov = optimize.curve_fit(circular_gaussian, (x, y), filter.ravel(),
                                    p0=initial_guess)
    return SimpleNamespace(amplitude=popt[0], xo=popt[1], yo=popt[2],
                           sigma=popt[3], theta=popt[4], offset=popt[5])


def get_rf_params(vis_angs, kernel_sizes, num_filters):
    # get sample unit RF sizes
    rf_params = pd.DataFrame()
    for vis_ang, kernel_size in zip(vis_angs, kernel_sizes):
        model = VOneNet(model_arch=None, visual_degrees=vis_ang,
                        ksize=kernel_size, gabor_seed=0)

        # get mean of selected filters and fit Gaussian
        filter_idcs = np.argsort(model.gabor_params['sf'])[:num_filters]
        params_q0 = model.simple_conv_q0.weight[filter_idcs].abs().mean(
            dim=(0, 1))
        params_q1 = model.simple_conv_q1.weight[filter_idcs].abs().mean(
            dim=(0, 1))
        mean_filter = torch.stack([params_q0, params_q1]).mean(0).numpy()
        image = np.zeros([IMAGE_SIZE_PIX] * 2)
        center = IMAGE_SIZE_PIX // 2
        start = center - kernel_size // 2
        image[start:start + kernel_size,
        start:start + kernel_size] = mean_filter
        rf = fit_gaussian(image)

        # 2D plot of RF
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        FWHM = rf.sigma * 2.355
        FWHM_circle = plt.Circle((center, center), FWHM / 2, color='tab:red',
                                 fill=False)
        ax.add_patch(FWHM_circle)
        kernel = plt.Rectangle((start, start), kernel_size, kernel_size,
                               color='tab:orange', fill=False)
        ax.add_patch(kernel)
        ax.spines[['left', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(f'receptive_fields/2d_rf_size_ang-{vis_ang:.2f}_kern-'
                    f'{kernel_size:03}.png')
        plt.close()

        # 1D plot of RF
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(image[center])
        ax.plot(gaussian(np.arange(IMAGE_SIZE_PIX), mean_filter.max(),
                         center, rf.sigma, rf.offset)),
        ax.axvline(center - FWHM / 2, color='tab:red')
        ax.axvline(center + FWHM / 2, color='tab:red')
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel('pixels')
        ax.set_xlim((0, IMAGE_SIZE_PIX))
        ax.set_xticks(np.arange(0, IMAGE_SIZE_PIX, 28))
        plt.tight_layout()
        plt.savefig(f'receptive_fields/1d_rf_size_ang-{vis_ang:.2f}_kern-'
                    f'{kernel_size:03}.png')
        plt.close()

        # save RF params
        pix_per_plot_cm = IMAGE_SIZE_PIX / 4.5
        df = pd.DataFrame(dict(
            visual_angle=[vis_ang],
            kernel_size_pix=[kernel_size],
            sigma_pix=[rf.sigma],
            sigma_plot_cm=[rf.sigma / pix_per_plot_cm],
            FWHM_pix=[FWHM],
            FWHM_plot_cm=[FWHM / pix_per_plot_cm],
            amplitude=[rf.amplitude],
            xo=[rf.xo],
            yo=[rf.yo],
        ))
        rf_params = pd.concat([rf_params, df]).reset_index(drop=True)

    return rf_params


def calculate_visual_angle(num_filters, overwrite=True):
    outdir = 'receptive_fields'
    os.makedirs(outdir, exist_ok=True)
    outpath = f'{outdir}/rf_params.csv'
    if not op.isfile(outpath) or overwrite:

        # get raw unit sizes at various visual angles
        vis_angs = [9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8]
        kernel_sizes = [25] + [int(25 * 8 / v) for v in vis_angs[1:]]
        vis_angs += VIS_ANGS
        kernel_sizes += KERNEL_SIZES
        for k, ksize in enumerate(kernel_sizes):
            if ksize % 2 == 0:
                kernel_sizes[k] += 1
        rf_params = get_rf_params(vis_angs, kernel_sizes, num_filters)

        # calculate relationship between visual angle and RF size
        def func(x, a, b, c):
            return a * np.exp(-b * x) + c * -x

        popt, pcov = optimize.curve_fit(
            func, rf_params.visual_angle.values, rf_params.sigma_pix.values)

        rf_params.plot(x='visual_angle', y='sigma_pix')
        X = [.125, .25, .5] + list(range(1, 37))
        plt.plot(X, [func(x, *popt) for x in X])
        plt.xlim(0, 36)
        plt.ylim(0, None)
        plt.show()

        # get target visual angles and kernel sizes
        # target_vis_ang = [gamma(s, *popt) for s in target_rfs_sigma_pix]
        # target_kern_sizes = [int(IMAGE_SIZE_PIX // t) for t in target_vis_ang]
        # for t in target_kern_sizes:
        #    if t % 2 == 0:
        #        t += 1

        # target_rf_params = get_rf_params(target_vis_ang, target_kern_sizes)
        # rf_params = pd.concat([rf_params, target_rf_params]).reset_index(
        # drop=True)
        rf_params.to_csv(outpath, index=False)


def evaluate_models(vis_angs, kernel_sizes, num_filters, overwrite=False):

    if not op.isfile('conditions.csv') or overwrite:

        # prepare data objects
        torch.no_grad()
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        image_dir = f'{FMRI_DIR}/exp1/stimuli/images'
        images = sorted(glob.glob(f'{image_dir}/*'))
        dataset = torch.empty((len(images), 3, 224, 224))
        image_counter = 0
        for cond in fMRI.cond_labels['exp1']:
            cond_label = cond.replace('none', 'complete')
            for im, image in enumerate(images):
                if cond_label in image:
                    image_PIL = Image.open(image).convert('RGB')
                    dataset[image_counter] = transforms.ToTensor()(image_PIL)
                    image_counter += 1
        df_conds = pd.DataFrame()
        df_indices = pd.DataFrame()

        # loop over different levels of fixation stability and visual angle
        for fix_std, (vis_ang, kernel_size) in itertools.product(
                FIX_STDS_DEG, zip(vis_angs, kernel_sizes)):

            print(f'Measuring responses, fix_std:{fix_std}, vis_ang:{vis_ang}')

            model = VOneNet(model_arch=None, visual_degrees=vis_ang,
                            ksize=kernel_size, rand_param=False).cuda()
            out_dir = f'{PROJ_DIR}/in_silico/models/VOneNet/fMRI'
            filter_idcs = np.argsort(model.gabor_params['sf'])[:num_filters]

            # no reps if fix_std == 0
            if fix_std == 0:
                inputs = normalize(dataset).cuda()
                responses_all = model(inputs)[:, filter_idcs].flatten(
                    start_dim=1).detach().cpu().numpy()
                unit_means = np.nanmean(responses_all, axis=0)
                unit_stds = np.nanstd(unit_means)
                response_z = unit_means / unit_stds
                selected_units = response_z > 3.1
                responses_final = responses_all[:, selected_units]

            # loop over reps for fix_std > 0
            else:
                responses_all = np.empty((REPS, 24, num_filters * 56 ** 2))
                sample_dir = f'sample_inputs_fix-{fix_std:.3f}'
                os.makedirs(sample_dir, exist_ok=True)
                sample_inputs = torch.zeros(REPS, 2, 3, 224, 224)
                for r in range(REPS):
                    #print(f'Measurement {r+1}/{REPS}')
                    inputs = dataset.clone()
                    for i, image in enumerate(dataset):
                        ecc = np.random.normal(loc=0, scale=fix_std * PPD)
                        ang = np.random.uniform(low=0, high=2 * np.pi)
                        x = int(ecc * np.cos(ang))
                        y = int(ecc * np.sin(ang))
                        inputs[i] = transforms.functional.affine(
                            img=image, angle=0, translate=(x, y),
                            scale=1., shear=0., fill=.5)
                    #inputs = normalize(inputs).cuda()
                    inputs = inputs.cuda()
                    responses_all[r] = (
                        model(inputs)[:, filter_idcs].detach().cpu().flatten(
                        start_dim=1).numpy())

                    # presentation of inputs and responses
                    sample_input = inputs[1:3].clone()
                    sample_inputs[r] = sample_input
                    for i in range(2):
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(transforms.ToPILImage()(sample_input[i]))
                        ax.spines[['left', 'bottom']].set_visible(False)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        plt.tight_layout()
                        plt.savefig(op.join(sample_dir, f'input_{i}_{r}.png'))
                        plt.close()

                sample_means = sample_inputs.mean(0)
                for i in range(2):
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(transforms.ToPILImage()(sample_means[i]))
                    ax.spines[['left', 'bottom']].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.tight_layout()
                    plt.savefig(op.join(sample_dir, f'input_{i}_mean.png'))
                    plt.close()

                # select units with a z-score > 3.1
                unit_means = responses_all.mean(axis=(0,1))
                selected_units = unit_means > (unit_means.std() * 3.1)
                #responses_all.std(axis=(0,1))
                #response_z = unit_means / unit_stds
                #selected_units = response_z > 3.1
                print(f'Number of selected units: {selected_units.sum()}/'
                      f'{np.prod(unit_means.shape)}')

                # average across measurements
                responses_final = (responses_all[:, :, selected_units]
                    .reshape(2, REPS//2, 24, -1).mean(axis=1))

            # make RSA dataset and calculate RSM
            print(f'Calculating RSM')
            os.makedirs(f'{out_dir}', exist_ok=True)
            RSM = RSA_dataset(responses=responses_final).calculate_RSM(
                NORM, NORM_METHOD, SIMILARITY)

            # plot RSM
            print(f'Plotting RSM ')
            fix_vis_label = f'fixrng-{fix_std:.3f}_visang-{vis_ang:.3f}'
            RSM.plot_RSM(
                vmin=-1, vmax=1,
                fancy=True,
                title=f'fixation std:{fix_std}, visual angle:{vis_ang}',
                labels=fMRI.cond_labels['exp1'],
                outpath=(f'{out_dir}/RSMs/{fix_vis_label}.pdf'),
                measure=ylabel)

            # MDS
            print(f'Plotting MDS')
            outpath = (f'{out_dir}/MDS/{fix_vis_label}.pdf')
            RSM.plot_MDS(title=None, outpath=outpath)

            # perform contrasts
            RSM.RSM_to_table()
            RSM.calculate_occlusion_robustness()
            RSM.fit_models()
            df = RSM.occlusion_robustness.copy(deep=True)
            df['vis_ang'] = vis_ang
            df['fix_std'] = fix_std
            df['num_units'] = selected_units.sum()
            df_indices = pd.concat([df_indices, df]).reset_index(drop=True)

            # condition-wise similarities
            df = RSM.RSM_table.copy(deep=True)
            df = df.drop(columns=['exemplar_a', 'exemplar_b', 'occluder_a',
                'occluder_b']).groupby(['analysis', 'level']).agg('mean'). \
                dropna().reset_index()
            df['vis_ang'] = vis_ang
            df['fix_std'] = fix_std
            df['num_units'] = selected_units.sum()
            df_conds = pd.concat([df_conds, df.copy(deep=True)]).reset_index(
                drop=True)

        df_conds.to_csv('conditions.csv')
        df_indices.to_csv('indices.csv')


# compare fix_stds for each model
def compare_single_var(overwrite=False):

    df_indices = pd.read_csv('indices.csv', index_col=0)
    df_conds = pd.read_csv('conditions.csv', index_col=0)

    # ensure levels are ordered correctly
    level_order = fMRI.occlusion_robustness_analyses[
                      'object_completion']['conds'] + \
                  fMRI.occlusion_robustness_analyses[
                      'occlusion_invariance']['conds']
    #df_conds.level = df_conds.level.astype('category').cat.reorder_categories(
    #    level_order)

    for variable, (analysis, params) in itertools.product(
            ['fix_std', 'vis_ang'], fMRI.occlusion_robustness_analyses.items()):

        out_dir = f'{analysis}'
        os.makedirs(out_dir, exist_ok=True)

        # condition-wise similarities
        outpath = (f'{out_dir}/condition-wise_similarities_{variable}.pdf')

        if not op.isfile(outpath) or overwrite:

            if variable == 'fix_std':
                df_analysis = df_conds[
                    (df_conds.analysis == analysis) &
                    (df_conds.vis_ang == 9)]
                xlabel = r"spatial jitter $\sigma$ ($\degree$)"
                xticklabels = fix_stds_labels
                df_analysis = df_analysis.sort_values(by=variable)
            else:
                df_analysis = df_conds[
                    (df_conds.analysis == analysis) &
                    (df_conds.fix_std == 0)]
                xlabel = r"visual angle of stimulus ($\degree$)"
                xticklabels = vis_angs_labels
                df_analysis = df_analysis.sort_values(by=variable,
                                                      ascending=False)
            df_analysis.level = df_analysis.level.astype(
                'category').cat.reorder_categories(
                fMRI.occlusion_robustness_analyses[
                    analysis]['conds'])

            df_means = df_analysis.pivot(index=variable, columns='level',
                                         values='similarity')

            fig, ax = plt.subplots(figsize=(4, 3))
            df_means.plot(
                kind='bar',
                rot=0,
                color=params['colours'],
                legend=False,
                ax=ax,
                width=.8)
            ax.set_yticks(np.arange(-1, 1.1, 1))
            ax.set_ylim((-.5, 1))
            ax.set_ylabel(ylabel)
            ax.set_xticks(labels=xticklabels, ticks=range(len(xticklabels)))
            #plt.text(index_x, index_y, index_texts, fontsize=16)
            plt.xlabel(xlabel)
            plt.tight_layout()
            fig.savefig(outpath)
            plt.close()

        # robustness indices
        outpath = f'{out_dir}/{params["index_label"]}_{variable}.pdf'
        if not op.isfile(outpath) or overwrite:

            # model indices
            if variable == 'fix_std':
                df_index = df_indices[
                    (df_indices.analysis == analysis) &
                    (df_indices.vis_ang == 9) &
                    (df_indices.subtype == 'norm')]
                xlabel = r"spatial jitter $\sigma$ ($\degree$)"
                df_index = df_index.sort_values(by=variable)
            else:
                df_index = df_indices[
                    (df_indices.analysis == analysis) &
                    (df_indices.fix_std == 0) &
                    (df_indices.subtype == 'norm')]
                xlabel = r"visual angle of stimulus ($\degree$)"
                df_index = df_index.sort_values(by=variable,
                                                      ascending=False)
            index_values = df_index.value.values

            # human indices
            fMRI_vals = []
            linestyles = ['solid', 'dashed', 'dotted']
            for exp, task in zip(
                    ['exp1', 'exp2', 'exp2'],
                    ['occlusion', 'occlusionAttnOn', 'occlusionAttnOff']):
                fMRI_data = pd.read_csv(
                    f'{FMRI_DIR}/{exp}/derivatives/RSA/'
                    f'{task}_space-standard/norm-all-conds_z-score/'
                    f'pearson/{analysis}/indices.csv')
                fMRI_vals.append(fMRI_data['value'][
                    (fMRI_data.level == 'group') &
                    (fMRI_data.subtype == 'norm') &
                    (fMRI_data.region == 'V1')].item())
            fig, ax = plt.subplots(figsize=(3.5, 2))
            x_pos = np.arange(len(index_values))
            ax.plot(x_pos, index_values, color='tab:purple', marker='o')
            ceiling_x = np.arange(-.5, len(FIX_STDS_DEG) + .5)
            ax.fill_between(ceiling_x, 1.0, 2, color='black', alpha=.2, lw=0)
            ax.fill_between(ceiling_x, 0, -1, color='black', alpha=.2, lw=0)
            ax.set_yticks((0, .5, 1), labels=['0','.5','1'])
            ax.set_ylabel(params['index_label'])
            ax.set_ylim((-0.1, 1.2))
            for fMRI_val, ls in zip(fMRI_vals, linestyles):
                ax.axhline(y=fMRI_val, xmin=-.5, xmax=len(FIX_STDS_DEG) + .5,
                           color='tab:blue', ls=ls)
            ax.set_xticks(ticks=x_pos, labels=df_index[variable].values)
            ax.set_xlim((-.5, len(FIX_STDS_DEG) - .5))
            #ax.set_title(f"{analysis.replace('_', ' ')} index")
            ax.set_xlabel(xlabel)
            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()

            # legend
            labels = [
                'Human V1 (Exp. 1)',
                'Human V1 (Exp. 2, attended)',
                'Human V1 (Exp. 2, unattended)',
                'VOneNet']
            markers = [None, None, None, 'o']
            lss = ['solid', 'dashed', 'dotted', None]
            cols = ['tab:blue']*3 + ['tab:purple']
            make_legend(outpath=f'model_legend.pdf', labels=labels,
                        markers=markers, linestyles=lss, colors=cols)


def compare_fix_std_vis_ang(overwrite=False):

    df_indices = pd.read_csv('indices.csv', index_col=0)
    df_conds = pd.read_csv('conditions.csv', index_col=0)

    # ensure levels are ordered correctly
    level_order = fMRI.occlusion_robustness_analyses[
                      'object_completion']['conds'] + \
                  fMRI.occlusion_robustness_analyses[
                      'occlusion_invariance']['conds']
    df_conds.level = df_conds.level.astype('category').cat.reorder_categories(
        level_order)

    for analysis, params in fMRI.occlusion_robustness_analyses.items():

        out_dir = f'{analysis}'
        os.makedirs(out_dir, exist_ok=True)

        """
        # condition-wise similarities (5x5)
        outpath = (f'{out_dir}/condition-wise_similarities_fix_std_vis_ang.pdf')
        if not op.isfile(outpath) or overwrite:

            fig, axes = plt.subplots(5, 5, figsize=(6, 6), sharex=True,
                                     sharey=True)
            for (f, fix_std), (v, vis_ang) in itertools.product(
                    enumerate(FIX_STDS_DEG), enumerate(VIS_ANGS)):
                ax = axes[f, v]
                df = df_conds[(df_conds.analysis == analysis) &
                              (df_conds.fix_std == fix_std) &
                              (df_conds.vis_ang == vis_ang)]
                df.plot(
                    x='level', y='similarity',
                    kind='bar',
                    rot=0,
                    color=params['colours'],
                    legend=False,
                    ax=ax,
                    width=1)
                ax.set_yticks(np.arange(-1, 1.1, 1))
                ax.set_ylim(-.5, 1)
                ax.set_xlim(-1.5, 4.5)
                if f == 2 and v == 0:
                    ax.set_ylabel("Correlation ($\it{r}$)", fontsize=12)
                ax.set_xticks([])
                ax.tick_params(axis='both', which='minor', bottom=False,
                               left=False)
                # plt.text(index_x, index_y, index_texts, fontsize=16)
                if f == 4:
                    ax.set_xlabel(f'{vis_ang}'+r"$\degree$", fontsize=12,
                                  labelpad=4)
            fig.subplots_adjust(left=0.25)
            left_pos = -38
            plt.text(left_pos, 8.5, r"spatial jitter $\sigma$", fontsize=12,
                     ha='center')
            for vert_pos, fix_std_label in zip(
                    np.linspace(7.5, 0.25, 5), fix_stds_labels):
                plt.text(left_pos, vert_pos, fix_std_label, fontsize=12,
                         ha='center')
            plt.text(-13.5, -1.3, "visual angle of stimulus", fontsize=12,
                     ha='center')
            #plt.tight_layout()
            fig.savefig(outpath)
            plt.close()
        """

        # condition-wise similarities (1x2)
        outpath = (
            f'{out_dir}/condition-wise_similarities_fix_std_vis_ang.pdf')
        if not op.isfile(outpath) or overwrite:

            fig, axes = plt.subplots(1, 2, figsize=(2.5, 3.5), sharey=True)
            fix_std_labels = (r"0.29$\degree$", r"0.20$\degree$")
            for (f, fix_std), (v, vis_ang) in itertools.product(
                    enumerate(FIX_STDS_DEG), enumerate(VIS_ANGS)):
                ax = axes[f]
                df = df_conds[(df_conds.analysis == analysis) &
                              (df_conds.fix_std == fix_std) &
                              (df_conds.vis_ang == vis_ang)]
                df.plot(
                    x='level', y='similarity',
                    kind='bar',
                    rot=0,
                    color=params['colours'],
                    legend=False,
                    ax=ax,
                    width=1)
                ax.set_yticks(np.arange(-1, 1.1, .5))
                ax.set_ylim(-.1, 1)
                ax.set_xlim(-1.5, 4.5)
                if f == 0:
                    ax.set_ylabel("Correlation ($\it{r}$)", fontsize=12)
                else:
                    ax.spines['left'].set_visible(False)
                    ax.tick_params(axis='both', which='both', bottom=False,
                                   left=False)
                ax.tick_params(axis='both', which='minor', bottom=False,
                               left=False)
                ax.set_xticks([])
                # plt.text(index_x, index_y, index_texts, fontsize=16)
                ax.set_xlabel(fix_std_labels[f], fontsize=12, labelpad=4)
            fig.text(.58, .03, r"Spatial jitter $\sigma$", fontsize=12,
                     ha='center', va='center')
            fig.subplots_adjust(bottom=0.13, left=.25)
            #plt.tight_layout()
            fig.savefig(outpath)
            plt.close()

        # robustness indices

        # human indices
        fMRI_vals = []
        linestyles = ['solid', 'dashed', 'dotted']
        for exp, task in zip(
                ['exp1', 'exp2', 'exp2'],
                ['occlusion', 'occlusionAttnOn', 'occlusionAttnOff']):
            fMRI_data = pd.read_csv(
                f'{FMRI_DIR}/{exp}/derivatives/RSA/'
                f'{task}_space-standard/norm-all-conds_z-score/'
                f'pearson/{analysis}/indices.csv')
            fMRI_vals.append(fMRI_data['value'][
                                 (fMRI_data.level == 'group') &
                                 (fMRI_data.subtype == 'norm') &
                                 (fMRI_data.region == 'V1')].item())

        # matrix plot
        outpath = f'{out_dir}/{params["index_label"]}_fix_std_vis_ang.pdf'
        if not op.isfile(outpath) or overwrite:
            index_values = np.empty((len(FIX_STDS_DEG), len(VIS_ANGS)))
            for (f, fix_std), (v, vis_ang) in itertools.product(
                    enumerate(FIX_STDS_DEG), enumerate(VIS_ANGS)):
                index_values[f,v] = df_indices[
                    (df_indices.analysis == analysis) &
                    (df_indices.fix_std == fix_std) &
                    (df_indices.vis_ang == vis_ang) &
                    (df_indices.subtype == 'norm')].value.item()
            fig, ax = plt.subplots(figsize=(6, 3))
            im = ax.imshow(index_values, vmin=0, vmax=1, cmap='viridis')
            ax.tick_params(**{'length': 0})
            ax.set_xticks(np.arange(len(VIS_ANGS)), vis_angs_labels)
            ax.set_yticks(np.arange(len(FIX_STDS_DEG)), fix_stds_labels)
            ax.tick_params(direction='in')
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.set_xlabel(r"visual angle of stimulus")
            ax.set_ylabel(r"spatial jitter $\sigma$")
            fig.subplots_adjust(right=0.7)
            cbar = fig.colorbar(im, fraction=0.0453, ax=ax)
            cbar.set_ticks(ticks=[0, 1] + fMRI_vals,
                           labels=['0', '1'] + [
                'Human V1 (Exp. 1)',
                'Human V1 (Exp. 2, attended)',
                'Human V1 (Exp. 2, unattended)'], fontsize=8)
            for (f, fix_std), (v, vis_ang) in itertools.product(
                    enumerate(FIX_STDS_DEG), enumerate(VIS_ANGS)):
                plt.text(v, f, f'{np.abs(index_values[f,v]):.2f}'[1:],
                fontsize=8, ha='center', va='center', color='white')
            ax.set_title(f"{analysis.replace('_', ' ')} index")
            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()


        # bar plot
        outpath = (f'{out_dir}/'
                   f'{params["index_label"]}_fix_std_vis_ang_barplot.pdf')
        if not op.isfile(outpath) or overwrite:

            colors = matplotlib.cm.tab20.colors
            text_offset = .03
            fontsize = 10
            fig, axes = plt.subplots(
                nrows=1, ncols=2, figsize=(2.5, 3.5), sharey=True,
                gridspec_kw={'width_ratios': [3,2]})

            # human panel
            ax = axes[0]
            human_colors = [colors[i] for i in [4, 4, 5]]
            ax.bar(np.arange(3), fMRI_vals, color=human_colors)
            ax.set_xticks([])
            ax.set_yticks(np.arange(0, 1.1, .5))
            ax.set_ylim(0, 1)
            ax.set_ylabel(f'{analysis.replace("_", " ")} index', size=12)
            ax.set_title("Human V1")
            ax.text(0, text_offset, 'Exp. 1', fontsize=fontsize, ha='center',
                    rotation=90, c='w')
            ax.text(1, text_offset, 'Exp. 2, object discrimination',
                    fontsize=fontsize,
                    ha='center', rotation=90, c='w')
            ax.text(2, text_offset, 'Exp. 2, letter detection',
                    fontsize=fontsize,
                    ha='center', rotation=90, c='w')

            # model panel
            ax = axes[1]
            model_vals = []
            for (f, fix_std), (v, vis_ang) in itertools.product(
                    enumerate(FIX_STDS_DEG), enumerate(VIS_ANGS)):
                model_vals.append(df_indices[
                    (df_indices.analysis == analysis) &
                    (df_indices.fix_std == fix_std) &
                    (df_indices.vis_ang == vis_ang) &
                    (df_indices.subtype == 'norm')].value.item())
            model_colors = [colors[i] for i in [0, 1]]
            ax.bar(np.arange(2), model_vals, color=model_colors)
            ax.set_xticks([])
            ax.set_title("VOneNet")
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='both', left=False)
            ax.text(0, text_offset, r"$\sigma$" + ' = 0.29' + r"$\degree$",
                    fontsize=fontsize, ha='center', rotation=90, c='w')
            ax.text(1, text_offset, r"$\sigma$" + ' = 0.20' + r"$\degree$",
                    fontsize=fontsize, ha='center', rotation=90, c='w')

            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()


if __name__ == '__main__':

    main()



