'''
This scripts compares various pixel attribution maps for occluded images
'''

import os
import os.path as op
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import seaborn as sns
from scipy import stats
import pickle as pkl
import pandas as pd
from scipy.optimize import curve_fit
import math
import time
from types import SimpleNamespace
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import itertools
import pingouin as pg
from scipy.stats import zscore
from tqdm import tqdm
import torch
from PIL import Image
from itertools import product as itp


np.random.seed(42)
TABCOLS = list(mcolors.TABLEAU_COLORS.keys())

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # converts to order in nvidia-smi (not in cuda)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # which device(s) to use

from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    show_factorization_on_image)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import (GradCAM, GradCAMPlusPlus, HiResCAM, EigenGradCAM,
                              DeepFeatureFactorization)
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

from .model_contrasts import model_contrasts, region_to_layer, effect_colors
from .helper_functions import get_trained_model, reorg_dict, now
from .behavioral_benchmark import load_trials, list_images

sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from plot_utils import make_legend, custom_defaults

plt.rcParams.update(custom_defaults)

sys.path.append(op.expanduser('~/david/master_scripts'))
from DNN.utils import get_activations, predict, CustomDataSet

EXP_DIR = f'code/in_vivo/behavioral/exp2'
from in_vivo.behavioral.exp2.analysis import CFG as BEHAV
from in_vivo.behavioral.exp2.analysis import (
    NUM_TRIALS, NUM_FRAMES,  FRAMES, FRAME_BATCHES, STIM_IDS, VISIBILITIES)
np.random.seed(42)
SAMPLER = np.random.permutation(NUM_TRIALS * len(VISIBILITIES))

T = SimpleNamespace(classification=True, num_workers=8, nGPUs=-1, batch_size=8)
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=[0.445, 0.445, 0.445],
                         std=[0.269, 0.269, 0.269]),
])


class TargetDataset:
    dataset = pd.read_csv('utils/SVC_images.csv')
    image_paths = dataset.filepath.values
    object_classes = dataset['class'].values
    targets = [BEHAV.object_classes.index(oc) for oc in object_classes]
    targets_1000 = [BEHAV.class_idxs[t] for t in targets]
    sampler = np.random.permutation(len(image_paths))
TARGET_DATASET = TargetDataset()


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(
            *list(self.model.children())[:-1])

    def __call__(self, x):
        return self.feature_extractor(x).flatten(start_dim=1)



class FeatureStore:
    def __init__(self, key, features):
        setattr(self, key, features)

class FeatureExtractor(torch.nn.Module):

    def __init__(self, model, model_name, layer):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.model_name = model_name
        self.key = layer
        if 'cyc' in layer:
            self.layer = layer.split('_cyc')[0]
            self.cycle = layer.split(f'{layer}_')[-1]
        else:
            self.layer = layer
            self.cycle = None

    def __call__(self, x):
        activations = reorg_dict(get_activations(
            self.model, self.model_name, inputs=x, layers=self.layer,
            array_type='torch'))
        return FeatureStore(self.key, activations[self.key])



def get_target_features(model_dir, model_name, method, transform, out_dir):
    # if method == 'EigenCAM':
    #    return [ClassifierOutputTarget(BEHAV.class_idxs[t]) for t in range(8)]

    if method in ['GradCAM', 'GradCAM++', 'HiResCAM']:
        features_path = f'{out_dir}/target_features.pt'
        layer = region_to_layer[model_name]['IT']
        if not op.isfile(features_path):
            model = get_trained_model(model_dir)
            activations = reorg_dict(get_activations(
                model, model_name, TARGET_DATASET.image_paths, T=T,
                layers=[layer], transform=transform,
                sampler=TARGET_DATASET.sampler,
                array_type='torch'))
            features = {k: torch.Tensor(0) for k in activations.keys()}
            for key, act in activations.items():
                features_unsampled = torch.cat([act[i].unsqueeze(0) for i in
                                                np.argsort(
                                                    TARGET_DATASET.sampler)],
                                               dim=0).flatten(
                    start_dim=1)
                for t in range(8):
                    features_all = torch.concat([features_unsampled[np.where(
                        np.array(TARGET_DATASET.targets) == t)[0], ...]])
                    features_mean = features_all.mean(dim=0).unsqueeze(0)
                    features[key] = torch.cat([features[key], features_mean],
                                              dim=0)
            torch.save(features, features_path)
        else:
            features = torch.load(features_path)
            if type(features) is not dict:
                features = {layer: features}
        return features


def load_cam(model_dir, model_name, method):
    model = get_trained_model(model_dir).cuda()
    feature_layer = region_to_layer[model_name]['IT']
    feature_layer = feature_layer.replace('.output', '')
    feature_module = model
    for l in feature_layer.split('.'):
        feature_module = feature_module[int(l)] if l.isnumeric() else getattr(
            feature_module, l)
    output_module = list(model.children())[-1]
    cam_module = {
        'GradCAM': GradCAM,
        'GradCAM++': GradCAMPlusPlus,
        'HiResCAM': HiResCAM,
        'EigenGradCAM': EigenGradCAM}[method]
    cam = cam_module(model=model, target_layers=[feature_module])
    """
    elif method == 'all-classes':
        # cam = DeepFeatureFactorization(
        #    model=model, target_layer=feature_module,
        #    computation_on_concepts=output_module)
        #    reshape_transform=lambda t: torch.flatten(t, start_dim=1))
        # cam = DeepFeatureFactorization(
        #    model=model, target_layer=model.features[11],
        #    computation_on_concepts=model.classifier[6])

        import torchvision
        resnet = torchvision.models.resnet50(pretrained=True)
        cam = DeepFeatureFactorization(
            model=resnet, target_layer=resnet.layer4,
            computation_on_concepts=resnet.fc)
    
    else:
        raise ValueError(f'Unknown method: {method}')
    """
    return cam


def get_salience(cam, method, inputs, targets, target_features=None):
    if 'CAM' in method:
        if target_features is not None:
            target_batch = [SimilarityToConceptTarget(target_features[t])
                            for t in targets]
        else:
            target_batch = [ClassifierOutputTarget(t) for t in targets]
        salience = cam(input_tensor=inputs, targets=target_batch)
        return salience

    else:  # if type(cam) is DeepFeatureFactorization:
        concepts, batch_expl, concept_scores = cam(inputs, n_components=4)
        concepts = concepts[BEHAV.class_idxs].numpy()
        concept_scores = concept_scores[BEHAV.class_idxs].numpy()
        concepts = concepts[BEHAV.class_idxs].numpy()
        # visualization = show_factorization_on_image(
        #    inputs[0], batch_explanations[0], image_weight=0.3,
        #    concept_labels=BEHAV.object_classes)

        # result = np.hstack((img, visualization))


def pixel_attribution(model_dir, m=0, total_models=0,
                      num_procs=1, overwrite=False):
    # model
    model_info = model_dir.split('models/')[-1].split('/')
    model_name, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'

    # 2x2 design for different pixels attribution methods
    for method, salience_target in itp(
            ['GradCAM', 'GradCAM++', 'HiResCAM'], #, 'EigenGradCAM'],
            ['true_class', 'predicted_class']):

        out_dir = (f'{model_dir}/behavioral/exp2/pixel_attribution/'
                   f'{method}_{salience_target}')
        os.makedirs(out_dir, exist_ok=True)

        salience_maps_path = f'{out_dir}/salience_maps.npy'
        if not op.isfile(salience_maps_path) or overwrite:

            print(f'{now()} | Pixel attribution, model {m + 1}/{total_models},'
                  f' {model_name}/{identifier}/{transfer_dir}, {salience_target}')

            trials = load_trials('exp2', drop_human=True, model_dir=model_dir)
            trials = trials[trials.layer == 'output']
            trials = trials[trials.cycle == trials.cycle.max()]
            trials = (trials
                .sort_values(['visibility', 'stimulus_id'])
                .reset_index(drop=True))

            image_paths = list_images('exp2')
            num_images = len(image_paths)
            assert num_images == len(trials)

            target_features_all = get_target_features(
                model_dir, model_name, method, transform,
                f'{model_dir}/behavioral/exp2/pixel_attribution')
            feature_key = sorted(list(target_features_all.keys()))[0]
            #target_features = target_features_all[feature_key].cuda()

            dataset = CustomDataSet(image_paths, targets=trials.index.values,
                transform=transform)
            loader = DataLoader(
                dataset, batch_size=T.batch_size, num_workers=num_procs,
                sampler=SAMPLER, pin_memory=True)

            salience_maps = np.zeros((0, 224, 224), dtype=np.uint8)

            cam = load_cam(model_dir, model_name, method)

            for batch, (inputs, idcs) in enumerate(tqdm(
                    loader, unit=f"batch({T.batch_size})")):

                # get trial info for batch
                trial_batch = trials.iloc[idcs].copy().reset_index(drop=True)
                true_classes = trial_batch.object_class.to_list()
                pred_classes = trial_batch.prediction.to_list()

                # get salience maps
                sal_classes = (pred_classes if salience_target ==
                               'predicted_class' else true_classes)
                sal_idcs_8 = [BEHAV.object_classes.index(i) for i in
                              sal_classes]
                sal_idcs_1000 = [BEHAV.class_idxs[i] for i in sal_idcs_8]
                #salience_batch = get_salience(
                #    cam, method, inputs, sal_idcs_8, target_features)
                salience_batch = get_salience(
                    cam, method, inputs, sal_idcs_1000)

                # save sample inputs with salience maps for pred + targ classes
                if batch < 16:
                    sample_dir = f'{out_dir}/samples'
                    os.makedirs(sample_dir, exist_ok=True)
                    for i, (t, p) in enumerate(zip(
                            true_classes, pred_classes)):
                        if inputs[i].max() == inputs[i].min():
                            continue  # skip blank images
                        image = inputs[i] - inputs[i].min()
                        image /= image.max()
                        image = image.permute(1, 2, 0).numpy()
                        out_path = (
                            f'{sample_dir}/batch-{batch:02}_image-{i:02}_'
                            f'targ-{t}_pred-{p}.png')
                        cam_image = show_cam_on_image(
                            image, salience_batch[i], use_rgb=True)
                        Image.fromarray(cam_image).save(out_path)


                # convert salience map to np.uint8 and add to salience_maps
                salience_map = salience_batch - salience_batch.min()
                salience_map /= salience_map.max()
                salience_map *= 255
                salience_map = salience_map.astype(np.uint8)
                salience_maps = np.concatenate(
                    [salience_maps, salience_map], axis=0)

                # periodically reload cam to avoid memory leak
                if (batch + 1) % 100 == 0:
                    cam = load_cam(model_dir, model_name, method)

            salience_maps = salience_maps.reshape(
                (NUM_TRIALS, NUM_FRAMES, 224, 224))
            with open(salience_maps_path, 'wb') as f:
                np.save(f, salience_maps)


def normalize_salience_maps(salience_maps):
    normed_maps = (zscore(salience_maps
        .reshape(NUM_TRIALS, NUM_FRAMES, -1), axis=2)
        .reshape(NUM_TRIALS, NUM_FRAMES, 224, 224))
    return normed_maps


def calculate_oa(row, sal_maps, object_pixels):
    trial_idx = STIM_IDS.index(row.stimulus_id.values[0])
    vis_idx = np.where(VISIBILITIES == row.visibility.values[0])[0]
    pix_obj = object_pixels[trial_idx, vis_idx]
    sal = sal_maps[trial_idx, vis_idx]
    if pix_obj.max() == pix_obj.min():
        oa = np.nan
    else:
        oa = sal.mean(where=pix_obj)
    row['pix_obj_att'] = oa
    return row


def calculate_cc(row, fix_maps, sal_maps):
    trial_idx = STIM_IDS.index(row.name[0])
    vis_idx = np.where(VISIBILITIES == row.name[1])[0][0]
    fix = np.mean(fix_maps[trial_idx, :vis_idx+1], axis=0).flatten()
    sal = sal_maps[trial_idx, vis_idx].flatten()
    row['pix_corr_coef'] = np.corrcoef(fix, sal)[0, 1]
    return row


def calculate_oi(row, sal_maps):
    trial_idx = STIM_IDS.index(row.name[0])
    vis_idx = np.where(VISIBILITIES == row.name[1])[0][0]
    if row.name[1] == 1:
        row['pix_occ_inv'] = 1
    else:
        sal_current = sal_maps[trial_idx, -1].flatten()
        sal_final = sal_maps[trial_idx, vis_idx].flatten()
        row['pix_occ_inv'] = np.corrcoef(sal_current, sal_final)[0, 1]
    return row


def calculate_nss(df, sal_maps):
    df.sort_values('subject', inplace=True)
    result = pd.DataFrame(index=VISIBILITIES)
    result.index.name = 'visibility'
    for i, row in df.iterrows():
        if row.visibility != 1:
            continue  # DRY: fixations for all vis stored in each row
        saliences = [[] for _ in VISIBILITIES]
        trial_idx = STIM_IDS.index(row.stimulus_id)
        subject = row.subject
        for fix, (fx, fy, ff, fl) in enumerate(zip(  # each fixation
                row.fix_x, row.fix_y, row.fix_first_fr,
                row.fix_last_fr)):
            fx_ds, fy_ds = int(fx // 4), int(fy // 4)
            if fx_ds in np.arange(224) and fy_ds in np.arange(224):
                for f in np.arange(ff, fl + 1):
                    vis_idcs = np.arange([v for v, batch in enumerate(
                        FRAME_BATCHES) if f in batch][0])
                    for vis_idx in vis_idcs:  # all previous frames
                        saliences[vis_idx].append(
                            sal_maps[trial_idx, vis_idx, fy_ds, fx_ds])
        salience_means = [np.mean(s) if len(s) else np.nan for s
                          in saliences]
        result[('pix_norm_sal', subject)] = salience_means

    return result


def evaluate_salience(model_dir, m=0, total_models=0, overwrite=False):

    exp = 'exp2'
    model_info = model_dir.split('models/')[-1].split('/')
    model_name, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'
    trials_m = load_trials(exp, model_dir=model_dir)
    drop_cols = [c for c in trials_m.columns if c.startswith('pix_')]
    trials_m.drop(columns=['index'] + drop_cols, errors='ignore', inplace=True)
    trials_m.to_parquet(f'{model_dir}/behavioral/exp2/trials.parquet',
                        index=False)
    #salience_path = f'{out_dir}/salience_maps.npy'
    #if not op.isfile(salience_path):
    #    return
    for method, salience_target in itp(
            ['GradCAM', 'GradCAM++', 'HiResCAM'],  # , 'EigenGradCAM'],
            ['true_class', 'predicted_class']):
        out_dir = (f'{model_dir}/behavioral/exp2/pixel_attribution/'
                   f'{method}_{salience_target}')
        salience_path = f'{out_dir}/salience.parquet'
        if not op.isfile(salience_path) or overwrite:

            print(f'{now()} | Comparing with human fixations, model {m + 1}'
                  f'/{total_models}, {model_name}/{identifier}/{transfer_dir}')

            # load object pixels
            with open(f'{EXP_DIR}/analysis/modeling_data/object_pixels.npy',
                      'rb') as f:
                object_pixels = np.load(f)

            print(f'{now()} | Loading human fixation data')
            trials_h = load_trials(exp)
            with open(f'{EXP_DIR}/analysis/modeling_data/fixation_maps_smooth.npy',
                      'rb') as f:
                fixation_maps = np.load(f)

            print(f'{now()} | Preprocessing salience maps')
            layer = [l for l in trials_m.layer.unique() if l != 'output'][0]
            trials_m = load_trials(exp, model_dir=model_dir)
            trials_m = trials_m[trials_m.layer == layer]
            trials_m = (trials_m[trials_m.cycle == trials_m.cycle.max()]
                .reset_index(drop=True))  # same for every layer / cycle
            assert (len(trials_m) == len(STIM_IDS) * NUM_FRAMES)
            with open(f'{out_dir}/salience_maps.npy', 'rb') as f:
                salience_maps = np.load(f)
            salience_maps = normalize_salience_maps(salience_maps)

            print(f'{now()} | Calculating object attention index')
            trials_m = (trials_m
                .groupby(['stimulus_id', 'visibility'])
                .apply(calculate_oa, salience_maps, object_pixels)
                .reset_index(drop=True))

            print(f'{now()} | Calculating occlusion invariance')
            trials_m = (trials_m
                .groupby(['stimulus_id', 'visibility'])
                .apply(calculate_oi, salience_maps)
                .reset_index(drop=True))

            print(f'{now()} | Calculating correlation with human fixations')
            trials_m = (trials_m
                .groupby(['stimulus_id', 'visibility'])
                .apply(calculate_cc, fixation_maps, salience_maps)
                .reset_index(drop=True))

            print(f'{now()} | Calculating normalized scan path saliency')
            # requires model data (separate rows for each visibility level) to
            # be merged with human fixation data (one row for all visibility levels)
            mutual_cols = [c for c in trials_m.columns if c in trials_h.columns]
            [mutual_cols.remove(c) for c in ['accuracy','prediction', 'visibility']]
            trials = trials_h.merge(trials_m, on=mutual_cols, suffixes=('_h', ''))
            nss = (trials
                .groupby(['stimulus_id'])
                .apply(calculate_nss, salience_maps)
                .reset_index(['stimulus_id', 'visibility']))
            trials_m.sort_values(['stimulus_id', 'visibility'], inplace=True)
            trials_m.columns = pd.MultiIndex.from_product([trials_m.columns, ['']])
            salience = trials_m.merge(
                nss, on=[('stimulus_id', ''), ('visibility', '')])

            salience.to_parquet(salience_path)

def make_plots(model_dir, m=0, total_models=0, overwrite=False):
    exp = 'exp2'
    model_info = model_dir.split('models/')[-1].split('/')
    model_name, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'
    trials_m = load_trials(exp, model_dir=model_dir)
    methods = ['object_attention', 'normalized_scanpath_saliency',
               'correlation_coefficient']
    columns = ['pix_obj_att', 'pix_norm_sal', 'pix_corr_coef']
    for method, salience_target in itp(
            ['GradCAM', 'GradCAM++', 'HiResCAM'],  # , 'EigenGradCAM'],
            ['true_class', 'predicted_class']):
        out_dir = (f'{model_dir}/behavioral/exp2/pixel_attribution/'
                   f'{method}_{salience_target}')
        salience = pd.read_parquet(f'{out_dir}/salience.parquet')
        for metric, col in zip(methods, columns):
            out_path = f'{out_dir}/{metric}.png'
            if not op.isfile(out_path) or overwrite:

                fig, ax = plt.subplots(figsize=(4, 3))

                if metric == 'normalized_scanpath_saliency':
                    index_cols = [c for c in salience.columns if c[0] != col]
                    value_cols = [c for c in salience.columns if c[0] == col]
                    melted_df = salience.melt(
                        id_vars=index_cols,
                        value_vars=value_cols,
                        var_name=['metric', 'subject'])
                    melted_df.columns = [c[0] if type(c) is tuple else c for
                                         c in melted_df.columns]
                    vals = (melted_df
                        .groupby(['subject', 'visibility'])
                        .agg({'value': 'mean'})
                        .groupby('visibility')
                        .agg(['mean', 'sem'], numeric_only=True))

                else:  # correlation and object attention
                    vals = (salience.groupby('visibility')
                            .agg({(col, ''): ['mean', 'std']}))
                means, errors = vals.values.T
                ax.plot(VISIBILITIES, means, zorder=1)
                if errors is not None:
                    ax.errorbar(VISIBILITIES, means, errors, capsize=2,
                                zorder=2)
                ax.scatter(VISIBILITIES, means)
                # plt.yticks(plot_cfg['yticks'])
                ax.axhline(y=0, color='k', linestyle='dotted')
                plt.xticks((0, 1))
                plt.xlim(-.05,1.05)
                plt.xlabel('visibility')
                plt.ylabel(metric)
                #plt.ylim(plot_cfg['ylims'])
                #plt.title(plot_cfg['title'])
                plt.tight_layout()
                fig.savefig(out_path)
                plt.close()


def compare_models(overwrite=False):

    for model_contrast, config in model_contrasts.items():

        num_models = len(config)
        results_dir = (
            f'data/in_silico/analysis/{model_contrast}/pixel_attribution')

        plot_cfgs = dict(
            correlation_coefficient=dict(
                column='pix_corr_coef',
                title='Correlation between human fixation\n'
                      'and model salience maps',
                ylabel=r"$\it{r}$",
                yticks=np.arange(-1, 2, 1),
                ylims=(-1, 1),
                chance=0),
            normalized_scanpath_saliency=dict(
                column='pix_norm_sal',
                title='Normalized scanpath saliency',
                ylabel='Z',
                yticks=np.arange(-1, 2, 1),
                ylims=(-1, 1),
                chance=0),
            object_attention=dict(
                column='pix_obj_att',
                title='Salience of object pixels',
                ylabel='Z',
                yticks=np.arange(-1, 2, 1),
                ylims=(-1, 1),
                chance=0),
            occlusion_invariance=dict(
                column='pix_occ_inv',
                title=('Correlation with salience\n'
                      'at full visibility'),
                ylabel=r"$\it{r}$",
                yticks=np.arange(-1, 2, 1),
                ylims=(-1, 1),
                chance=0))

        for (metric, plot_cfg), method, salience_target in itp(
                plot_cfgs.items(),
                ['GradCAM', 'GradCAM++', 'HiResCAM'],  # ,'EigenGradCAM'],
                ['true_class', 'predicted_class']):
            out_dir = f'{results_dir}/{method}_{salience_target}'
            os.makedirs(out_dir, exist_ok=True)

            out_path = f'{out_dir}/{metric}.png'
            if not op.isfile(out_path) or overwrite:

                fig, ax = plt.subplots(figsize=(4, 3))

                for m, (label, info) in enumerate(config.items()):
                    path, color = info['path'], info['color']
                    xpos = m if 'xpos' not in info else info['xpos']
                    model_dir = f'in_silico/models/{path}'

                    # data
                    salience = pd.read_parquet(
                        f'{model_dir}/behavioral/exp2/pixel_attribution/'
                        f'{method}_{salience_target}/salience.parquet')
                    if metric == 'normalized_scanpath_saliency':
                        index_cols = [c for c in salience.columns if
                                      c[0] != plot_cfg['column']]
                        value_cols = [c for c in salience.columns if
                                      c[0] == plot_cfg['column']]
                        melted_df = salience.melt(
                            id_vars=index_cols,
                            value_vars=value_cols,
                            var_name=['metric', 'subject'])
                        melted_df.columns = [c[0] if type(c) is tuple else c for
                                             c in melted_df.columns]
                        vals = (melted_df
                                .groupby(['subject', 'visibility'])
                                .agg({'value': 'mean'})
                                .groupby('visibility')
                                .agg(['mean', 'sem'], numeric_only=True))

                    else:  # correlation and object attention
                        vals = (salience.groupby('visibility')
                            .agg({(plot_cfg['column'], ''): ['mean', 'std']}))
                    means, errors = vals.values.T
                    ax.plot(VISIBILITIES, means, color=color, zorder=1)
                    if errors is not None:
                        ax.errorbar(VISIBILITIES, means, errors,
                                    color=color, capsize=2,
                                    zorder=2)
                    ax.scatter(VISIBILITIES, means, s=.5, color=color)
                    # plt.yticks(plot_cfg['yticks'])
                ax.axhline(y=0, color='k', linestyle='dotted')
                plt.xticks((0, 1))
                plt.xlim(-.05, 1.05)
                plt.xlabel('visibility')
                plt.ylabel(plot_cfg['ylabel'])
                plt.ylim(plot_cfg['ylims'])
                plt.title(plot_cfg['title'])
                plt.tight_layout()
                fig.savefig(out_path)
                plt.close()

                # plt.yticks(plot_cfg['yticks'])
                ax.axhline(y=0, color='k', linestyle='dotted')
                plt.xticks([-2, -1])
                plt.xlim(-.5, xpos + .5)
                plt.ylabel(plot_cfg['ylabel'])
                plt.xlim(-.5, num_models - .5)
                #plt.ylim(plot_cfg['ylims'])
                plt.title(plot_cfg['title'])
                plt.tight_layout()
                fig.savefig(out_path)
                plt.close()

                # legend
                outpath = f'{out_dir}/legend.pdf'
                if model_contrast in effect_colors:
                    leg_labels = list(effect_colors[model_contrast].keys())
                    leg_colors = list(effect_colors[model_contrast].values())
                else:
                    leg_colors = [m['color'] for m in config.values()]
                    leg_labels = list(config.keys())
                if not op.isfile(outpath) or overwrite:
                    make_legend(
                        outpath=outpath,
                        labels=leg_labels,
                        markers='s',
                        colors=leg_colors,
                        markeredgecolors=None,
                        linestyles='None')


if __name__ == "__main__":
    from seconds_to_text import seconds_to_text

    start = time.time()

    os.chdir(op.expanduser('~/david/projects/p022_occlusion'))
    model_dir = op.expanduser('~/david/models/alexnet/pretrained')
    pix_at = pixel_attribution
    pix_at(model_dir, overwrite=True)

    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')
