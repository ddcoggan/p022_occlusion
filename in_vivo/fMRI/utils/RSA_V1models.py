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
FIX_STDS_DEG = [0.291409, 0.197803]  # for the object and letter tasks
NUMS_FILTERS = [512, 32]  # sanity check: higher completion in largest filters?
VIS_ANG = 3.2  # this gives the same mean RF size as the mean human V1 pRF size
KERNEL_SIZE = 99  # ensure this is large enough to contain the filters
MEAN_V1_FWHM_DEG = 0.7396
fix_stds_labels = [f'{f:.2f}' + r"$\degree$" for f in FIX_STDS_DEG]

def main(overwrite=False):
    for num_filters in NUMS_FILTERS:
        analysis_dir = f'{FMRI_DIR}/V1_models/{num_filters}_filters'
        os.makedirs(analysis_dir, exist_ok=True)
        os.chdir(analysis_dir)
        get_rf_params(num_filters)
        evaluate_models(num_filters, overwrite=False)
        compare_fix_std_vis_ang(overwrite=True)


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


def get_rf_params(num_filters):
    outdir = 'receptive_fields'
    os.makedirs(outdir, exist_ok=True)
    outpath = f'{outdir}/rf_params.csv'

    model = VOneNet(model_arch=None, visual_degrees=VIS_ANG,
                    ksize=KERNEL_SIZE, gabor_seed=0)
    filter_idcs = np.argsort(model.gabor_params['sf'])[:num_filters]

    params_q0 = model.simple_conv_q0.weight[filter_idcs]
    params_q1 = model.simple_conv_q1.weight[filter_idcs]

    # plot all filters
    for params, params_str in zip([params_q0, params_q1], ['q0', 'q1']):
        outpath = f'{outdir}/{params_str}_filters.png'
        num_filters = params.shape[0]
        grid_size = np.ceil(np.sqrt(num_filters))
        montage_size = [int(KERNEL_SIZE * grid_size)] * 2
        montage = Image.new(size=montage_size, mode='RGB')
        for i in range(num_filters):
            image_array = np.array(params[i, :, :, :].permute(1, 2, 0))
            image_pos = image_array - image_array.min()  # rescale to between 0,255 for PIL
            image_scaled = image_pos * (255.0 / image_pos.max())
            image = Image.fromarray(image_scaled.astype(np.uint8))
            offset_x = int(i % grid_size) * KERNEL_SIZE
            offset_y = int(i / grid_size) * KERNEL_SIZE
            montage.paste(image, (offset_x, offset_y))
        montage.save(outpath)

    # get mean of selected filters and fit Gaussian
    params_q0_mean = params_q0.abs().mean(dim=(0, 1))
    params_q1_mean = params_q1.abs().mean(dim=(0, 1))
    mean_filter = torch.stack([params_q0_mean, params_q1_mean]).mean(0).numpy()
    image = np.zeros([IMAGE_SIZE_PIX] * 2)
    center = IMAGE_SIZE_PIX // 2
    start = center - KERNEL_SIZE // 2
    image[start:start + KERNEL_SIZE,
    start:start + KERNEL_SIZE] = mean_filter
    rf = fit_gaussian(image)

    # 2D plot of RF
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    FWHM = rf.sigma * 2.355
    FWHM_circle = plt.Circle((center, center), FWHM / 2, color='tab:red',
                             fill=False)
    ax.add_patch(FWHM_circle)
    kernel = plt.Rectangle((start, start), KERNEL_SIZE, KERNEL_SIZE,
                           color='tab:orange', fill=False)
    ax.add_patch(kernel)
    ax.spines[['left', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'receptive_fields/2d_rf_size.png')
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
    plt.savefig(f'receptive_fields/1d_rf_size.png')
    plt.close()

    # save RF params
    df = pd.DataFrame(dict(
        visual_angle=[VIS_ANG],
        kernel_size_pix=[KERNEL_SIZE],
        sigma_pix=[rf.sigma],
        sigma_deg=[rf.sigma / PPD],
        FWHM_pix=[FWHM],
        FWHM_deg=[FWHM / PPD],
        amplitude=[rf.amplitude],
        xo=[rf.xo],
        yo=[rf.yo],
    ))
    df.to_csv(outpath, index=False)


def evaluate_models(num_filters, overwrite=False):

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
        for fix_std in FIX_STDS_DEG:

            print(f'Measuring responses, fix_std:{fix_std}')

            model = VOneNet(model_arch=None, visual_degrees=VIS_ANG,
                            ksize=KERNEL_SIZE, rand_param=False).cuda()
            out_dir = op.expanduser(f'~/david/models/VOneNet/fMRI')
            filter_idcs = np.argsort(model.gabor_params['sf'])[:num_filters]


            # loop over reps
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
            fix_vis_label = f'fixrng-{fix_std:.3f}_visang-{VIS_ANG:.3f}'
            RSM.plot_RSM(
                vmin=-1, vmax=1,
                fancy=True,
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
            df['vis_ang'] = VIS_ANG
            df['fix_std'] = fix_std
            df['num_units'] = selected_units.sum()
            df_indices = pd.concat([df_indices, df]).reset_index(drop=True)

            # condition-wise similarities
            df = RSM.RSM_table.copy(deep=True)
            df = df.drop(columns=['exemplar_a', 'exemplar_b', 'occluder_a',
                'occluder_b']).groupby(['analysis', 'level']).agg('mean'). \
                dropna().reset_index()
            df['vis_ang'] = VIS_ANG
            df['fix_std'] = fix_std
            df['num_units'] = selected_units.sum()
            df_conds = pd.concat([df_conds, df.copy(deep=True)]).reset_index(
                drop=True)

        df_conds.to_csv('conditions.csv')
        df_indices.to_csv('indices.csv')


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

        # condition-wise similarities (1x2)
        outpath = (
            f'{out_dir}/condition-wise_similarities.pdf')
        if not op.isfile(outpath) or overwrite:

            fig, axes = plt.subplots(1, 2, figsize=(2.5, 3), sharey=True)
            for f, fix_std in enumerate(FIX_STDS_DEG):
                ax = axes[f]
                df = df_conds[(df_conds.analysis == analysis) &
                              (df_conds.fix_std == fix_std)]
                df.plot(
                    x='level', y='similarity',
                    kind='bar',
                    rot=0,
                    color=params['colours'],
                    legend=False,
                    ax=ax,
                    width=1)
                ax.set_yticks(np.arange(-1, 1.1, .5))
                ax.set_ylim(-.4, 1)
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
                ax.set_xlabel(fix_stds_labels[f], fontsize=12, labelpad=4)
            fig.text(.58, .03, r"Spatial jitter $\sigma$", fontsize=12,
                     ha='center', va='center')
            fig.subplots_adjust(bottom=0.13, left=.25)
            #plt.tight_layout()
            fig.savefig(outpath)
            plt.close()

        # robustness indices

        # human indices
        fMRI_vals = []
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

        # bar plot
        outpath = f'{out_dir}/{params["index_label"]}_barplot.pdf'
        if not op.isfile(outpath) or overwrite:

            colors = matplotlib.cm.tab20b.colors
            text_offset = .03
            fontsize = 10
            fig, axes = plt.subplots(
                nrows=1, ncols=2, figsize=(2.5, 3.5), sharey=True,
                gridspec_kw={'width_ratios': [3,2]})

            # human panel
            ax = axes[0]
            human_colors = [colors[i] for i in [0, 0, 1]]
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
            for f, fix_std in enumerate(FIX_STDS_DEG):
                model_vals.append(df_indices[
                    (df_indices.analysis == analysis) &
                    (df_indices.fix_std == fix_std) &
                    (df_indices.subtype == 'norm')].value.item())
            model_colors = [colors[i] for i in [12, 13]]
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

