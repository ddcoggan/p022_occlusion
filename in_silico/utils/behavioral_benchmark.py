'''
This script tests the accuracy of CNNs on classifying the exact images presented in the human behavioral experiment.
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
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from itertools import product as itp
import pingouin as pg
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

sys.path.append('../in_vivo/behavioral')
from in_vivo.behavioral.exp1.analysis import CFG as EXP1
from in_vivo.behavioral.exp1.analysis import condwise_robustness_plot_array
from in_vivo.behavioral.exp2.analysis import CFG as EXP2

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # converts to order in nvidia-smi (not in cuda)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # which device(s) to use

import torchvision.transforms.v2 as transforms
from torch import float32
from .model_contrasts import region_to_layer
from .helper_functions import get_trained_model, reorg_dict, now
from . import model_base

sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from math_functions import sigmoid
from plot_utils import make_legend, custom_defaults
plt.rcParams.update(custom_defaults)

sys.path.append(op.expanduser('~/david/master_scripts/DNN/utils'))
from get_activations import get_activations
from predict import predict


np.random.seed(42)

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
NUM_FRAMES = 17
VISIBILITIES = {'exp1': EXP1.visibilities,
                'exp2': np.linspace(0, 1, NUM_FRAMES)}
FRAMES = np.linspace(1, 360, NUM_FRAMES).astype(int)
T = SimpleNamespace(classification=True, nGPUs=-1, batch_size=64)


def get_transform(model_name):

    imsize = 256 if model_name == 'pix2pix' else 224
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(float32, scale=True),
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize(mean=[0.445, 0.445, 0.445],
                             std=[0.269, 0.269, 0.269]),
    ])
    return transform


def make_svc_dataset(overwrite=False):
    """ This function creates a set of images to train a linear support vector
    machine (SVC) classifier. The images are from the same classes as the
    behavioral experiments but are not the same images. """

    dataset_dir = op.expanduser(f'~/Datasets/ILSVRC2012')
    SVC_images_path = 'utils/SVC_images.csv'

    if not op.isfile(SVC_images_path) or overwrite:

        # get set of all images used in behavioral exps
        behavioral_set = set()
        for exp in EXPS:
            trials = load_trials(exp)
            behavioral_set.update([op.basename(x).split('.')[0]
                                   for x in trials.object_path.values])

        # create independent training set
        classes, images = [], []
        ims_per_class_svc = 256
        for class_dir, class_label in zip(OBJ_CLS_DIRS, OBJ_CLASSES):
            im_counter = 0
            image_paths = sorted(
                glob.glob(f'{dataset_dir}/train/{class_dir}/*'))
            while im_counter < ims_per_class_svc:
                image_path = image_paths.pop(0)
                if op.basename(image_path) not in behavioral_set:
                    classes.append(class_label)
                    images.append(image_path)
                    im_counter += 1
        SVC_images = pd.DataFrame({'class': classes, 'filepath': images})
        SVC_images.to_csv(SVC_images_path, index=False)


def make_pca_dataset(overwrite=False):
    """ This function creates a set of images with which to measure
    the principle components of layer activations. Images selected are from
    the 992 imagenet_classes not used in the behavioral experiments. """

    dataset_dir = op.expanduser(f'~/Datasets/ILSVRC2012')

    PCA_images_path = 'utils/PCA_images.csv'
    if not op.isfile(PCA_images_path) or overwrite:

        imagenet_classes = [op.basename(path) for path in sorted(glob.glob(
            f'{dataset_dir}/val/*'))]
        for d in OBJ_CLS_DIRS:
            imagenet_classes.remove(d)
        classes, images = [], []
        ims_per_class = 2
        for imagenet_class in imagenet_classes:
            images += sorted(glob.glob(
                f'{dataset_dir}/val/{imagenet_class}/*'))[:ims_per_class]
            classes += [imagenet_class] * ims_per_class
        PCA_images = pd.DataFrame({'class': classes, 'filepath': images})
        PCA_images.to_csv(PCA_images_path, index=False)


def train_svc(model_dir, m=0, total_models=0, num_procs=1, overwrite=False):
    svc_path = f'{model_dir}/behavioral/SVC.pkl'
    T.num_workers = num_procs

    if not op.isfile(svc_path) or overwrite:

        overwrite = True
        os.makedirs(op.dirname(svc_path), exist_ok=True)
        model_info = model_dir.split('models/')[-1].split('/')
        model_name, identifier = model_info[:2]
        transfer_dir = model_info[2] if len(model_info) == 3 else 'X'
        layers = [region_to_layer[model_name]['IT']]

        print(f'{now()} | Training SVC, '
              f'model {m + 1}/{total_models}, {model_name}'
              f'/{identifier}/{transfer_dir}')

        # use PCA for dimensionality reduction
        print(f'{now()} | Running PCA...')
        pca_images = pd.read_csv('utils/PCA_images.csv').filepath.values
        model = get_trained_model(model_dir, True, layers)
        activations = get_activations(
            model, model_name, pca_images, T=T, layers=layers,
            transform=get_transform(model_name), shuffle=True)
        activations = reorg_dict(activations)
        svcs = {key: {'pca': PCA().fit(value.reshape(
            [len(pca_images), -1]))} for key, value in activations.items()}

        # train a support vector machine on responses to the training set
        print(f'{now()} | Running SVC...')
        svc_dataset = pd.read_csv('utils/SVC_images.csv')
        svc_images = svc_dataset['filepath'].values
        sampler = np.random.permutation(len(svc_images))
        svc_classes = [svc_dataset['class'].values[s] for s in sampler]
        model = get_trained_model(model_dir, True, layers)  # solves mem leak
        activations = get_activations(
            model, model_name, svc_images, T=T,
            layers=layers, transform=get_transform(model_name), sampler=sampler)
        activations = reorg_dict(activations)

        for key, act in activations.items():
            pca_weights = svcs[key]['pca'].transform(
                act.reshape((len(svc_images), -1)))[:, :1000]
            clf = OneVsRestClassifier(BaggingClassifier(
                SVC(kernel='linear', probability=True),
                max_samples=1 / num_procs, n_estimators=num_procs))
            clf.fit(pca_weights, svc_classes)
            train_acc = np.mean(clf.predict(pca_weights) == svc_classes)
            svcs[key]['svc'] = clf
            print(f'{now()} | Training accuracy ({key}): {train_acc:.4f}')

        with open(svc_path, 'wb') as f:
            pkl.dump(svcs, f)

    return overwrite


def load_trials(exp, drop_human=False, model_dir=None):
    if model_dir is None:
        trials = pd.read_parquet(
            f'../data/in_vivo/behavioral/{exp}/analysis/trials.parquet')
    else:
        trials = pd.read_parquet(
            f'{model_dir}/behavioral/{exp}/trials.parquet')
        if 'accuracy' not in trials.columns:
            trials = reshape_metrics(trials, 'wide')
            trials.to_parquet(f'{model_dir}/behavioral/{exp}/trials.parquet')
        drop_human = False  # human data not present

    # drop human data
    if drop_human:  # human data
        if exp == 'exp1':
            trials.drop(columns=['prediction', 'accuracy', 'RT'], inplace=True)
        elif exp == 'exp2':
            trials = (trials
                [trials.subject.isin(['sub-01', 'sub-02'])]
                .drop(columns=[
                    'trial', 'RT', 'prediction', 'accuracy', 'visibility',
                    'subject'] + [c for c in trials.columns if c.startswith('fix')])
                .sort_values(by='stimulus_id').reset_index(drop=True))

    # enforce ordering of categorical variables
    trials.object_animacy = pd.Categorical(
        trials.object_animacy, OBJ_ANIMACIES, ordered=True)
    trials.object_class = pd.Categorical(
        trials.object_class, OBJ_CLASSES, ordered=True)
    trials.occluder_class = pd.Categorical(
        trials.occluder_class, OCC_CLASSES[exp], ordered=True)
    trials.occluder_color = pd.Categorical(
        trials.occluder_color, OCC_COLORS, ordered=True)

    return trials


def list_images(exp):
    exp_dir = f'../data/in_vivo/behavioral/{exp}'
    if exp == 'exp1':
        trials = load_trials(exp)
        image_dir = f'{exp_dir}/data'
        images = trials.occluded_object_path.values
        #images = [f'{image_dir}/{"/".join(x.split("/")[-3:])}' for x in images]
    elif exp == 'exp2':
        image_dir = f'{exp_dir}/stimuli/final'
        images = np.ravel([sorted(glob.glob(
            f'{image_dir}/set-?/frames/*/{f:03}.png')) for f in
            FRAMES]).tolist()
    else:
        raise ValueError('exp must be "exp1" or "exp2"')

    return images

def list_unocc_images(exp):
    exp_dir = f'../data/in_vivo/behavioral/{exp}'
    imagenet_dir = op.expanduser(f'~/Datasets/ILSVRC2012')
    if exp == 'exp1':
        trials = load_trials(exp)
        images = trials.object_path.values
        #images = []
        #for i, image in enumerate(trials.object_path.values):
        #    paths = glob.glob(f'{imagenet_dir}/*/*/{op.basename(image)}')
        #    assert len(paths) == 1, f'no or multiple matches for {image}'
        #    images.append(paths[0])
        #trials.object_path = images
    elif exp == 'exp2':
        image_dir = f'{exp_dir}/stimuli/final'
        images = np.ravel([[x] * len(FRAMES) for x in sorted(glob.glob(
            f'{image_dir}/set-?/frames/*/{FRAMES[-1]:03}.png'))]).tolist()
    else:
        raise ValueError('exp must be "exp1" or "exp2"')

    return images


def get_responses(model_dir, m=0, total_models=0, exp=None, num_procs=1,
                  overwrite=False):
    model_info = model_dir.split('models/')[-1].split('/')
    model_name, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'
    layers = [region_to_layer[model_name]['IT']]
    if model_name != 'pix2pix':
        layers += ['output']
    results_dir = f'{model_dir}/behavioral/{exp}'
    out_path = f'{results_dir}/trials.parquet'

    if not op.isfile(out_path) or overwrite:

        print(f'{now()} | Measuring responses for {exp} '
              f'stimuli, model {m + 1}/{total_models}, '
              f'{model_name}/{identifier}/{transfer_dir}')

        os.makedirs(op.dirname(out_path), exist_ok=True)
        T.num_workers = num_procs
        images = list_images(exp)
        sampler = np.random.permutation(len(images))
        reload_every = 4000  # overcomes memory leak with some recurrent models
        num_blocks = np.ceil(len(images) / reload_every).astype(int)
        for b in range(num_blocks):

            print(f'{now()} | Block {b + 1}/{num_blocks}')
            model = get_trained_model(model_dir, True, layers)  # reload model
            first = b * reload_every
            last = min(first + reload_every, len(images))
            inputs = [images[i] for i in sampler[first:last]]
            activations_batch = get_activations(
                model, model_name, inputs, T=T, layers=layers,
                transform=get_transform(model_name), shuffle=False)
            activations_batch = reorg_dict(activations_batch)
            if b == 0:
                activations = activations_batch
            else:
                activations = {key: np.concatenate(
                    [activations[key], activations_batch[key]], axis=0)
                    for key in activations}

        activations = {k: np.array([a[i] for i in np.argsort(
            sampler)]) for k, a in activations.items()}

        with open(f'{op.dirname(results_dir)}/SVC.pkl', 'rb') as f:
            svcs = pkl.load(f)
        all_trials = pd.DataFrame()

        for layer in layers:
            for key in [k for k in activations.keys() if k.startswith(layer)]:

                # get a copy of the trial data and add relevant info
                cycle = int(key.split('cyc')[-1]) if 'cyc' in key else -1
                trials = load_trials(exp, drop_human=True)
                trials['layer'] = layer
                trials['cycle'] = cycle
                if exp == 'exp2':
                    num_trials = len(trials)
                    trials = (pd.concat([trials] * len(FRAMES))
                              .reset_index(drop=True))
                    trials['visibility'] = [x for x in VISIBILITIES[exp] for _
                                            in range(num_trials)]

                # get pca and svc if necessary
                pca_obj, svc_obj = None, None
                if layer != 'output':
                    pca_obj = svcs[key]['pca']
                    svc_obj = svcs[key]['svc']

                # get predictions
                print(f'{now()} | Generating predictions for {layer}')
                trials = get_predictions(
                    trials=trials, activations=activations[key],
                    readout_layer=layer, pca_object=pca_obj,
                    svc_object=svc_obj)

                # print out a summary statistic
                print(f'{now()} | {key} accuracy: {trials.accuracy.mean():.4f}')
                all_trials = pd.concat([trials, all_trials])

        # save trials
        all_trials.reset_index(drop=True).to_parquet(out_path, index=False)

        return True
    return False


def get_predictions(trials, activations, readout_layer, pca_object=None,
                    svc_object=None):
    # check we have the right number of activations
    assert activations.shape[0] == len(trials), \
        'different number of images and activations'

    # predictions based on output layer or svc object
    if readout_layer == 'output':
        probs = special.softmax(activations[:, OBJ_CLS_IDCS], axis=1)
        class_map = np.arange(len(OBJ_CLASSES))
    else:
        pca_weights = pca_object.transform(
            activations.reshape((len(trials), -1)))[:, :1000]
        probs = svc_object.predict_proba(pca_weights)
        class_map = [OBJ_CLASSES.index(c) for c in svc_object.classes_]
    assert (probs.sum(1).round(4) == 1).all(), 'probabilities do not sum to 1'

    # add predictions and other measures to trials
    trials['prediction'] = [OBJ_CLASSES[class_map[c]] for c in
                            probs.argmax(axis=1)]
    trials['accuracy'] = pd.Series(
        trials.prediction == trials.object_class, dtype=int)
    trials['true_class_prob'] = [probs[i, OBJ_CLASSES.index(c)]
                                 for i, c in enumerate(trials.object_class)]
    trials['entropy'] = stats.entropy(probs, axis=1)

    return trials


def evaluate_reconstructions(model_dir, m=0, total_models=0,
                             exp=None, num_procs=1, overwrite=False):

    model_info = model_dir.split('models/')[-1].split('/')
    model_name, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'
    results_dir = f'{model_dir}/behavioral/{exp}'
    out_path = f'{results_dir}/reconstruction.parquet'

    if not op.isfile(out_path) or overwrite:

        print(f'{now()} | Evaluating reconstructions for {exp} '
              f'stimuli, model {m + 1}/{total_models}, '
              f'{model_name}/{identifier}/{transfer_dir}')

        import torch
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image
        sys.path.append(op.expanduser('~/david/repos/pix2pix'))
        from gan.generator import UnetGenerator

        os.makedirs(op.dirname(out_path), exist_ok=True)
        T.num_workers = num_procs
        images_occ = list_images(exp)
        images_unocc = list_unocc_images(exp)
        sampler = np.random.permutation(len(images_occ))
        reload_every = 4000  # overcomes memory leak with some recurrent models
        num_blocks = np.ceil(len(images_occ) / reload_every).astype(int)

        class ReconstructionImages(Dataset):
            def __init__(self,
                         mode: str = 'eval',
                         direction: str = 'B2A',
                         files_unocc: list = [],
                         files_occ: list = []):
                self.mode = mode
                self.direction = direction
                self.transform = get_transform('pix2pix')
                self.files_unocc = files_unocc
                self.files_occ = files_occ

            def __len__(self, ):
                return len(self.files_unocc)

            def __getitem__(self, idx):

                imgA = Image.open(self.files_unocc[idx]).convert('RGB')
                imgA = self.transform(imgA)
                imgB = Image.open(self.files_occ[idx]).convert('RGB')
                imgB = self.transform(imgB)
                if self.direction == 'A2B':
                    return imgA, imgB
                else:
                    return imgB, imgA

        loss = torch.nn.BCEWithLogitsLoss()
        params_path = sorted(glob.glob(f"{model_dir}/params/*.pt*"))[-1]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        losses = []
        for b in range(num_blocks):

            print(f'{now()} | Block {b + 1}/{num_blocks}')

            generator = UnetGenerator().to(device)
            if torch.cuda.device_count() > 1:
                generator = torch.nn.DataParallel(generator)
            generator.load_state_dict(torch.load(params_path))
            first = b * reload_every
            last = min(first + reload_every, len(images_occ))
            inputs_occ = [images_occ[i] for i in sampler[first:last]]
            inputs_unocc = [images_unocc[i] for i in sampler[first:last]]
            dataset = ReconstructionImages(mode='eval',
                                           files_unocc=inputs_unocc,
                                           files_occ=inputs_occ)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False,
                                    num_workers=8)

            with tqdm(dataloader, unit=f"batch(128)") as tepoch:
                for batch, (occ, unocc) in enumerate(tepoch):
                    occ = occ.to(device)
                    unocc = unocc.to(device)
                    pred = generator(occ)
                    losses.extend([loss(p, u).detach().cpu().numpy().item()
                                   for p, u in zip(pred, unocc)])

        trials = load_trials(exp, drop_human=True)
        if exp == 'exp2':
            num_trials = len(trials)
            trials = (pd.concat([trials] * len(FRAMES))
                      .reset_index(drop=True))
            trials['visibility'] = [x for x in VISIBILITIES[exp] for _
                                    in range(num_trials)]
        trials['reconstruction_loss'] = [losses[i] for i in np.argsort(sampler)]
        trials['layer'] = 'reconstruction'
        trials['cycle'] = -1
        trials.reset_index(drop=True).to_parquet(out_path, index=False)

        print(f'{now()} | reconstruction loss: '
              f'{trials.reconstruction_loss.mean():.4f}')

        return True
    return False


def measure_robustness(trials, exp):
    agg_dict = dict(accuracy='mean', true_class_prob='mean', entropy='mean')
    agg_vars = ['subject'] if exp == 'exp1' else ['stimulus_set']
    agg_vars.extend(OCC_VARIABLES + OBJ_VARIABLES)
    robustness = (
        trials.groupby(agg_vars, dropna=False).agg(agg_dict).reset_index())
    robustness['layer'] = trials.layer.values[0]
    robustness['cycle'] = trials.cycle.values[0]

    return robustness


def estimate_model_RTs(trials_long):
    def _estimate_RT(df, metric):
        sustain_period = 3
        if metric == 'accuracy':
            performance = list(df.value.to_numpy() == 1)
        elif metric == 'true_class_prob':
            performance = list(df.value.to_numpy() > .5)
        else:  # if metric == 'entropy':
            performance = list(df.value.to_numpy() < .5)
        frame_index = len(performance) - 1  # default value if no criteria met
        for i in range(len(performance) - 2):
            if performance[i:i + sustain_period] == [True] * sustain_period:
                frame_index = i
                break
        else:
            # for cases where performance hits criteria within 2 highest vis
            if max(performance[-2:]):
                frame_index = (performance[-2:].index(True) +
                               (len(performance) - 2))
        vis = frame_index / (len(performance) - 1)

        return vis

    metric = trials_long.name[-1]
    trials = (trials_long[trials_long.visibility == 1])
    #.drop(columns=['layer', 'cycle', 'visibility', 'metric'])
    #.reset_index(drop=True))
    trials['visibility'] = (
        trials_long.groupby('stimulus_id').apply(_estimate_RT, metric)).values

    return trials


def fit_visibility_curves(trials, exp):
    def _fit_curve(xvals, yvals, thr=.5):
        try:
            init_params = [max(yvals), np.median(xvals), 1, 0]
            popt, pcov = curve_fit(
                sigmoid, xvals, yvals, init_params, maxfev=int(10e5))
            curve = sigmoid(np.linspace(0, 1, 1000), *popt)
            threshold = sum(curve < thr) / 1000
        except:
            UserWarning('Curve fitting failed, returning NaNs')
            popt, threshold = [np.nan] * 4, np.nan
        return popt, threshold

    curves = pd.DataFrame()
    metric = trials.name[-1]

    if exp == 'exp1':

        vis = VISIBILITIES[exp] + [1]
        if metric != 'entropy':
            vis = [0] + vis

        # separate function for each occluder_class * occluder_color
        for occluder_class, occluder_color in itp(OCC_CLASSES[exp], OCC_COLORS):

            yvals = (trials[
                         (trials['occluder_class'] == occluder_class) &
                         (trials['occluder_color'] == occluder_color)]
                     .groupby('visibility').mean(
                numeric_only=True).value.to_list())
            yval_mean = np.mean(yvals)

            unocc = trials[trials['visibility'] == 1].value.mean()
            yvals += [unocc]
            if metric != 'entropy':
                yvals = [1 / 8] + yvals

            # fit curve function
            popt, threshold = _fit_curve(vis, yvals)
            curves = pd.concat(
                [curves, pd.DataFrame({
                    'occluder_class': [occluder_class],
                    'occluder_color': [occluder_color],
                    'L': [popt[0]],
                    'x0': [popt[1]],
                    'k': [popt[2]],
                    'b': [popt[3]],
                    'threshold_50': [threshold],
                    'mean': [yval_mean],
                })]).reset_index(drop=True)

        # single function across entire dataset
        yvals = (trials.groupby('visibility').mean(
            numeric_only=True).value.to_list())
        yval_mean = np.mean(yvals)
        if metric != 'entropy':
            yvals = [1 / 8] + yvals
        popt, threshold = _fit_curve(vis, yvals)
        curves = pd.concat(
            [curves, pd.DataFrame({
                'occluder_class': ['all'],
                'occluder_color': ['all'],
                'L': [popt[0]],
                'x0': [popt[1]],
                'k': [popt[2]],
                'b': [popt[3]],
                'threshold_50': [threshold],
                'mean': [yval_mean],
            })]).reset_index(drop=True)

    else:  # if exp == 'exp2'

        # separate curve per stimulus_set, occluder_class, occluder_color
        for stimulus_set, occluder_class, occluder_color in (
                itp(['a', 'b'], OCC_CLASSES[exp], OCC_COLORS)):
            yvals = (trials[
                         (trials['stimulus_set'] == stimulus_set) &
                         (trials['occluder_class'] == occluder_class) &
                         (trials['occluder_color'] == occluder_color)]
                     .groupby('visibility')
                     .mean(numeric_only=True).value.to_list())
            yval_mean = np.mean(yvals)

            # fit curve function
            popt, threshold = _fit_curve(VISIBILITIES[exp], yvals)
            curves = pd.concat(
                [curves, pd.DataFrame({
                    'stimulus_set': [stimulus_set],
                    'occluder_class': [occluder_class],
                    'occluder_color': [occluder_color],
                    'L': [popt[0]],
                    'x0': [popt[1]],
                    'k': [popt[2]],
                    'b': [popt[3]],
                    'threshold_50': [threshold],
                    'mean': [yval_mean],
                })]).reset_index(drop=True)

        # single function across entire dataset
        yvals = (trials.groupby('visibility').mean(
            numeric_only=True).value.to_list())
        yval_mean = np.mean(yvals)
        popt, threshold = _fit_curve(VISIBILITIES[exp], yvals)
        curves = pd.concat(
            [curves, pd.DataFrame({
                'stimulus_set': ['all'],
                'occluder_class': ['all'],
                'occluder_color': ['all'],
                'L': [popt[0]],
                'x0': [popt[1]],
                'k': [popt[2]],
                'b': [popt[3]],
                'threshold_50': [threshold],
                'mean': [yval_mean],
            })]).reset_index(drop=True)

    return curves


def measure_human_likeness_exp1(trials_model, trials_human):

    def _c_obs(trials):
        return np.mean(trials.human_performance == trials.model_performance)

    def _c_err(trials):
        hum = trials.human_performance.mean()
        mod = trials.model_performance.mean()
        ceil = 1 - np.abs(hum - mod)  # ceil = 1 only if a == b
        chance_acc = hum * mod
        chance_inacc = (1 - hum) * (1 - mod)
        chance = chance_acc + chance_inacc
        c_obs = _c_obs(trials)
        c_err = (c_obs - chance) / (ceil - chance) if ceil > chance else np.nan
        return c_err

    def _c_inacc(trials):
        both_inacc = len(trials[(trials.human_performance == 0) &
                                (trials.model_performance == 0)])
        if both_inacc:
            c_inacc = (len(trials[
                              (trials.human_performance == 0) &
                              (trials.human_prediction ==
                               trials.model_prediction)]) / both_inacc)
            return c_inacc
        else:
            return np.nan

    def _measure_consistencies(trials, metrics, groupby=()):

        all_grp_cols = ['subject'] + list(groupby)
        df = pd.DataFrame()
        if 'c_obs' in metrics:
            df['c_obs'] = trials.groupby(all_grp_cols).apply(_c_obs)
        if 'c_err' in metrics:
            df['c_err'] = trials.groupby(all_grp_cols).apply(_c_err)
        if 'c_inacc' in metrics:
            df['c_inacc'] = trials.groupby(all_grp_cols).apply(_c_inacc)
        df = df.reset_index().melt(
            id_vars=all_grp_cols, value_vars=metrics, var_name='metric_sim')
        if len(groupby):  # average for each subject
            df = (df.groupby(['subject', 'metric_sim'])
                  .agg({'value': 'mean'}).reset_index())
        df['analysis'] = 'trial-wise'
        df['level'] = '_x_'.join(all_grp_cols)

        return df

    human_likeness = pd.DataFrame()

    # combine model and human data
    #trials = trials_human.copy()
    trials_human.rename(columns={'accuracy': 'human_performance',
                                 'prediction': 'human_prediction'},
                        inplace=True)
    #trials['model_performance'] = trials_model.value
    #trials['model_prediction'] = trials_model.prediction

    trials_model.rename(columns={'value': 'model_performance',
                                 'prediction': 'model_prediction'},
                        inplace=True)
    mutual_cols = list(np.intersect1d(trials_human.columns,
                                      trials_model.columns))
    trials = trials_human.merge(trials_model, on=mutual_cols,
                                suffixes=('_h', '_m'))

    # trial-wise metrics
    if trials_model.name[-1] == 'accuracy':

        # all trials, all metrics
        human_likeness = pd.concat([human_likeness, _measure_consistencies(
            trials, metrics=['c_obs', 'c_err', 'c_inacc'])])

        # measure c_err at successively finer cond scales
        groupby = []
        for var in ['visibility', 'occluder_class', 'occluder_color']:
            groupby.append(var)
            if len(groupby) == 4:
                breakpoint()
            human_likeness = (pd.concat([human_likeness, _measure_consistencies(
                trials, groupby=groupby, metrics=['c_err'])])
                              .reset_index(drop=True))

    # condition-wise accuracy correlation (occluded only)
    groupby = ['subject']
    trials_occ = trials[trials.visibility < 1]
    for var in ['occluder_class', 'occluder_color', 'visibility']:
        groupby.append(var)
        df = (trials_occ
              .groupby(groupby)
              .agg('mean', numeric_only=True)
              .groupby('subject').apply(lambda d:
                                        np.corrcoef(d.human_performance,
                                                    d.model_performance)[0, 1])
              .reset_index()
              .rename(columns={0: 'value'}))
        df['analysis'] = 'condition-wise'
        df['level'] = '_x_'.join(groupby)
        df['metric_sim'] = 'pearson_r'
        human_likeness = pd.concat([human_likeness, df]).reset_index(drop=True)

    # condition-wise accuracy correlation (across visibility including
    # unoccluded, separate corr for each occluder * color)
    unocc_h = (trials[trials.visibility == 1]
               .groupby('subject').human_performance.mean())
    unocc_m = (trials[trials.visibility == 1]
               .groupby('subject').model_performance.mean())
    groupby = ['subject', 'occluder_class', 'occluder_color', 'visibility']
    df = (trials
        .groupby(groupby)
        .agg('mean', numeric_only=True)
        .dropna()
        .reset_index()
        .groupby(groupby[:-1])
        .apply(lambda d: np.corrcoef(
            d.human_performance.to_list() + [unocc_h[d.subject.values[0]]],
            d.model_performance.to_list() + [unocc_m[d.subject.values[0]]])[
            0, 1])#.reset_index(drop=True)
        .groupby('subject')
        .agg('mean')
        .reset_index())
    df.rename(columns={0: 'value'}, inplace=True)
    df['analysis'] = 'condition-wise'
    df['level'] = 'visibility_corr_occluder_class_x_occluder_color'
    df['metric_sim'] = 'pearson_r'
    df.rename(columns={'accuracy': 'value'})
    human_likeness = pd.concat([human_likeness, df]).reset_index(
        drop=True)

    return human_likeness


def measure_human_likeness_exp1_reconstruction(trials_model, trials_human):
    human_likeness = pd.DataFrame()

    # combine model and human data
    # trials = trials_human.copy()
    trials_human.rename(columns={'accuracy': 'human_performance'},
                        inplace=True)
    # trials['model_performance'] = trials_model.value
    # trials['model_prediction'] = trials_model.prediction

    trials_model.rename(columns={'reconstruction_loss': 'model_performance'},
                        inplace=True)
    mutual_cols = list(np.intersect1d(trials_human.columns,
                                      trials_model.columns))
    trials = trials_human.merge(trials_model, on=mutual_cols,
                                suffixes=('_h', '_m'))

    # condition-wise accuracy correlation (occluded only)
    groupby = ['subject']
    trials_occ = trials[trials.visibility < 1]
    for var in ['occluder_class', 'occluder_color', 'visibility']:
        groupby.append(var)
        df = (trials_occ
              .groupby(groupby)
              .agg('mean', numeric_only=True)
              .groupby('subject').apply(lambda d:
                                        np.corrcoef(d.human_performance,
                                                    d.model_performance)[0, 1])
              .reset_index()
              .rename(columns={0: 'value'}))
        df['analysis'] = 'condition-wise'
        df['level'] = '_x_'.join(groupby)
        df['metric'] = 'accuracy_v_reconstruction_loss'
        df['metric_sim'] = 'pearson_r'
        human_likeness = pd.concat([human_likeness, df]).reset_index(drop=True)

    # condition-wise accuracy correlation (across visibility including
    # unoccluded, separate corr for each occluder * color)
    unocc_h = (trials[trials.visibility == 1]
               .groupby('subject').human_performance.mean())
    unocc_m = (trials[trials.visibility == 1]
               .groupby('subject').model_performance.mean())
    groupby = ['subject', 'occluder_class', 'occluder_color', 'visibility']
    df = (trials
          .groupby(groupby)
          .agg('mean', numeric_only=True)
          .dropna()
          .reset_index()
          .groupby(groupby[:-1])
          .apply(lambda d: np.corrcoef(
        d.human_performance.to_list() + [unocc_h[d.subject.values[0]]],
        d.model_performance.to_list() + [unocc_m[d.subject.values[0]]])[
        0, 1])  # .reset_index(drop=True)
          .groupby('subject')
          .agg('mean')
          .reset_index())
    df.rename(columns={0: 'value'}, inplace=True)
    df['analysis'] = 'condition-wise'
    df['level'] = 'visibility_corr_occluder_class_x_occluder_color'
    df['metric'] = 'accuracy_v_reconstruction_loss'
    df['metric_sim'] = 'pearson_r'
    df.rename(columns={'accuracy': 'value'})
    human_likeness = pd.concat([human_likeness, df]).reset_index(
        drop=True)

    return human_likeness


def measure_human_likeness_exp2(trials_model_rt, trials_model, trials_human):
    def _trial_corrs(trials_h, trials_m):
        mutual_cols = list(np.intersect1d(trials_h.columns, trials_m.columns))
        [mutual_cols.remove(x) for x in ['accuracy', 'visibility']]
        trials = trials_h.merge(trials_m, on=mutual_cols, suffixes=('_h', '_m'))
        trials_acc = trials[(trials.accuracy_h == 1) & (trials.accuracy_m == 1)]
        r = np.corrcoef(trials_acc.visibility_h, trials_acc.visibility_m)[0, 1]
        return r

    def _cond_corrs(trials_h, vis_m):
        vis_h = (trials_h
                 .groupby(['occluder_class', 'occluder_color'])
                 .visibility.mean())
        r = np.corrcoef(vis_h, vis_m)[0, 1]
        return r

    stimulus_set, layer, cycle, metric = trials_model_rt.name

    # trial-wise
    trials_h = trials_human[trials_human.stimulus_set == stimulus_set]
    trial_wise = pd.DataFrame(trials_h.groupby('subject')
                              .apply(_trial_corrs, trials_model_rt))
    trial_wise.rename(columns={0: 'value'}, inplace=True)
    trial_wise['analysis'] = 'trial-wise'
    trial_wise['level'] = 'subject'
    trial_wise.reset_index(inplace=True)

    # condition-wise
    vis_m = (trials_model[
                 (trials_model.stimulus_set == stimulus_set) &
                 (trials_model.layer == layer) &
                 (trials_model.cycle == cycle) &
                 (trials_model.metric == metric)]
             .groupby(['occluder_class', 'occluder_color', 'visibility'])
             .mean(numeric_only=True).reset_index()
             .groupby(['occluder_class', 'occluder_color'])
             .apply(lambda d: d[d.value > .5].visibility.min()))
    vis_m.columns = ['visibility']
    cond_wise = pd.DataFrame(dict(value=(trials_h.groupby('subject')
                                         .apply(_cond_corrs, vis_m))))
    cond_wise['analysis'] = 'condition-wise'
    cond_wise['level'] = 'subject_x_occluder_class_x_occluder_color'
    cond_wise.reset_index(inplace=True)

    human_likeness = pd.concat([cond_wise, trial_wise]).reset_index(drop=True)

    return human_likeness


def reshape_metrics(df, shape):
    """ Reshape dataframe based on different model performance metrics to help
    with grouping, plotting, etc. """
    if shape == 'long' and 'metric' not in df.columns:
        df = df.melt(id_vars=[c for c in df.columns if c not in METRICS],
                     value_vars=METRICS, var_name='metric')
    elif shape == 'wide' and 'metric' in df.columns:
        df = df.pivot(index=[c for c in df.columns if c not in [
            'metric', 'value']], columns='metric', values='value').reset_index()
    return df


def analyse_performance(model_dir, m=0, total_models=0, exp='exp1',
                        overwrite=False, remake_plots=False):
    model_info = model_dir.split('models/')[-1].split('/')
    results_dir = f'{model_dir}/behavioral/{exp}'
    mod_str = f'model {m + 1}/{total_models}\n{"/".join(model_info)}'
    recompare_models = False
    groupby = ['layer', 'cycle', 'metric']

    # fit performance curves
    curves_path = f'{results_dir}/robustness_curves.parquet'
    if not op.isfile(curves_path) or overwrite:
        print(f'{now()} | Analysing performance curves ({exp}) | {mod_str}')
        trials_model = reshape_metrics(
            load_trials(exp, model_dir=model_dir), shape='long')
        curves = (trials_model
                  .groupby(groupby)
                  .apply(fit_visibility_curves, exp)
                  .reset_index(groupby))
        curves.to_parquet(curves_path, index=False)
        recompare_models, remake_plots = True, True

    # generate response times for each model
    trials_model_rt_path = f'{results_dir}/trials_RT.parquet'
    if exp == 'exp2' and (not op.isfile(trials_model_rt_path) or overwrite):
        print(f'{now()} | Estimating model RTs ({exp}) | {mod_str}')
        trials_model = reshape_metrics(
            load_trials(exp, model_dir=model_dir), shape='long')
        trials_model_rt = (trials_model
                           .groupby(['stimulus_set'] + groupby)
                           .apply(estimate_model_RTs).reset_index(drop=True))
        trials_model_rt = trials_model_rt.merge(
            trials_model[(trials_model.visibility == 1) &
                         (trials_model.metric == 'accuracy')][
                groupby[:-1] + ['stimulus_id', 'value']].rename(columns={
                'value': 'accuracy'}),
            on=groupby[:-1] + ['stimulus_id'])  # include model acc at vis = 1
        trials_model_rt.to_parquet(trials_model_rt_path, index=False)
        recompare_models, remake_plots = True, True

    # measure human likeness
    likeness_path = f'{results_dir}/human_likeness.parquet'
    if not op.isfile(likeness_path) or overwrite:  # or exp == 'exp2':
        print(f'{now()} | Analysing human likeness ({exp}) | {mod_str}')
        trials_human = load_trials(exp, drop_human=False)
        trials_model = reshape_metrics(load_trials(exp, model_dir=model_dir),
                                       shape='long')
        if exp == 'exp1':
            likeness = (trials_model
                        .groupby(groupby)
                        .apply(measure_human_likeness_exp1, trials_human)
                        .reset_index(level=groupby))
            if 'pix2pix' in model_dir:
                trials_model_recon = pd.read_parquet(op.join(
                    model_dir, 'behavioral/exp1/reconstruction.parquet'))
                likeness_recon = measure_human_likeness_exp1_reconstruction(
                    trials_model_recon, trials_human)
                likeness = pd.concat([likeness, likeness_recon])

        else:  # if exp == 'exp2':
            trials_model_rt = pd.read_parquet(trials_model_rt_path)
            likeness = (trials_model_rt
                        .groupby(['stimulus_set'] + groupby)
                        .apply(measure_human_likeness_exp2, trials_model,
                               trials_human)
                        .reset_index(level=['stimulus_set'] + groupby))
        likeness.to_parquet(likeness_path, index=False)
        recompare_models, remake_plots = True, True

    # make plot in each model directory
    #plot_performance(model_dir, exp, remake_plots)  #TODO: make checks before
    # performing lots of work when overwrite is False

    return recompare_models


def plot_performance(model_dir, exp, overwrite=False):
    def _make_curve_plots(robustness, curves, info):

        layer, cycle, metric = info

        plot_cfg = dict(
            accuracy=dict(
                ylabel='classification accuracy',
                ylims=(0, 1),
                yticks=(0, 1),
                chance=1 / 8),
            true_class_prob=dict(
                ylabel='true class probability',
                ylims=(0, 1),
                yticks=(0, 1),
                chance=1 / 8),
            entropy=dict(
                ylabel='uncertainty',
                ylims=(0, 2.5),
                yticks=np.arange(0, 2.5),
                chance=None))[metric]

        outpath = f'{plot_dir}/cyc{cycle:02}_{layer}_{metric}.pdf'
        if not op.isfile(outpath) or overwrite:

            if exp == 'exp1':
                robustness.rename(columns={'value': metric}, inplace=True)
                condwise_robustness_plot_array(
                    df=robustness,
                    df_curves=curves,
                    metric=metric,
                    outpath=outpath,
                    ylabel=plot_cfg['ylabel'],
                    yticks=plot_cfg['yticks'],
                    ylims=plot_cfg['ylims'],
                    chance=plot_cfg['chance'],
                    legend_path=outpath.replace(metric, 'legend'))


            elif exp == 'exp2':

                # accuracy for each occluder * visibility
                fig, axes = plt.subplots(
                    2, 3, figsize=(3.5, 2.6), sharex=True, sharey=True)

                for o, occluder_class in enumerate(OCC_CLASSES[exp]):

                    ax = axes.flatten()[o]
                    for (c, color), (stimulus_set, edge_color) in itp(enumerate(
                            OCC_COLORS), zip(['a', 'b'], ['k', 'tab:grey'])):
                        # plot curve function underneath
                        face_color = EXP2.plot_colors[o * 2 + c]
                        popt = (curves[
                                    (curves.metric == metric) &
                                    (curves.stimulus_set == stimulus_set) &
                                    (curves.occluder_class == occluder_class) &
                                    (curves.occluder_color == color)]
                                [['L', 'x0', 'k', 'b']].values)
                        assert len(popt) == 1, 'more than one curve fit'
                        curve_x = np.linspace(0, 1, 1000)
                        curve_y = sigmoid(curve_x, *popt[0])
                        #ax.plot(curve_x, curve_y, color=face_color,
                        #        clip_on=False, zorder=1)

                        # plot accuracies on top
                        yvals = robustness[
                            #(robustness.stimulus_set == stimulus_set) &
                            (robustness.occluder_class == occluder_class) &
                            (robustness.occluder_color == color) &
                            (robustness.metric == metric)].groupby(
                            'visibility').value.mean().values
                        ax.scatter(VISIBILITIES[exp], yvals, clip_on=False,
                                   facecolor=face_color, zorder=2)
                        #edgecolor=edge_color, )

                    # format plot
                    #if o == 0:
                    #    ax.set_title('biological', size=7)
                    #else:
                    #    ax.set_title(EXP2.occluder_labels[o], size=7)
                    ax.set_xticks((0, 1))
                    ax.set_xlim((0, 1))
                    ax.set_yticks(plot_cfg['yticks'])
                    ax.set_ylim(plot_cfg['ylims'])
                    ax.tick_params(axis='both', which='major',
                                   labelsize=7)
                    if plot_cfg['chance']:
                        ax.axhline(y=plot_cfg['chance'], color='k',
                                   linestyle='dotted')
                    if o == 4:
                        ax.set_xlabel('visibility', size=10)

                fig.text(0, 0.58, plot_cfg['ylabel'], va='center',
                         rotation='vertical')
                plt.tight_layout()
                plt.savefig(outpath)
                plt.close()

    def _make_scatterplots(robustness_m, robustness_h, info):

        layer, cycle, metric = info
        outpath = (
            f'{plot_dir}/cyc{cycle:02}_{layer}_{metric}_human-likeness.pdf')
        if not op.isfile(outpath) or overwrite:
            if exp == 'exp1':
                colors = EXP1.plot_colors
                metric_human = 'accuracy'
            else:  # if exp == 'exp2':
                colors = EXP2.plot_colors
                metric_human = 'visibility'
            rob_h = robustness_h.rename(columns={metric_human: f'human_value'})
            rob_m = robustness_m.rename(columns={'value': f'model_value'})
            rob_h = (rob_h[
                         rob_h.occluder_class.isin(OCC_CLASSES[exp])]
                     .groupby(['occluder_class', 'occluder_color'])
                     .agg({'human_value': 'mean'}))
            rob_m = (rob_m[
                         rob_m.occluder_class.isin(OCC_CLASSES[exp])]
                     .groupby(['occluder_class', 'occluder_color'])
                     .agg({'model_value': 'mean'}))
            plot_data = pd.merge(rob_h, rob_m,
                                 on=['occluder_class', 'occluder_color'])
            xvals, yvals = plot_data[['human_value', 'model_value']].values.T
            plt.scatter(xvals, yvals, color=colors)
            plt.xlabel(f'human {metric_human}')
            plt.ylabel(f'model {metric}')
            x = [min(xvals), max(xvals)]
            y = np.poly1d(np.polyfit(xvals, yvals, 1))(x)
            plt.plot(x, y, color='k')
            r = np.corrcoef(xvals, yvals)[0, 1]
            plt.title(f'condition-wise accuracy scatterplot\nr = {r:.2f}')
            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()

    results_dir = f'{model_dir}/behavioral/{exp}'
    plot_dir = f'{results_dir}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    trials_h = load_trials(exp)
    trials_m = load_trials(exp, model_dir=model_dir)
    curves = pd.read_parquet(f'{results_dir}/robustness_curves.parquet')

    # calculate mean performance for each condition

    # model
    groupby = ['layer', 'cycle', 'occluder_class',
               'occluder_color', 'metric', 'visibility']
    if exp == 'exp2':
        groupby += ['stimulus_set']
    robustness_m = (reshape_metrics(trials_m, 'long').groupby(groupby)
                    .agg({'value': 'mean'}).dropna().reset_index())

    # human
    groupby = ['occluder_class', 'occluder_color']
    if exp == 'exp1':
        groupby += ['visibility']
    if exp == 'exp2':
        groupby += ['stimulus_set']
    robustness_h = (trials_h.groupby(groupby)
                    .agg('mean', numeric_only=True).reset_index())

    # condition-wise scatter_plots, model vs human
    for rob_m in robustness_m.groupby(['layer', 'cycle', 'metric']):
        _make_scatterplots(rob_m[1], robustness_h, rob_m[0])

    # performance curves, model only
    if exp == 'exp1':
        trials_m.occluder_class = (
            trials_m.occluder_class.cat.add_categories('unoccluded'))
        trials_m.occluder_color = (
            trials_m.occluder_color.cat.add_categories('unoccluded'))
        trials_m.occluder_class[trials_m.visibility == 1] = 'unoccluded'
        trials_m.occluder_color[trials_m.visibility == 1] = 'unoccluded'
        trials_m.occluder_class = pd.Categorical(
            trials_m.occluder_class, OCC_CLASSES[exp] + ['unoccluded'],
            ordered=True)
    trials_m = reshape_metrics(trials_m, 'long')
    groupby = ['layer', 'cycle', 'metric']
    for rob, cur in zip(trials_m.groupby(groupby), curves.groupby(groupby)):
        _make_curve_plots(rob[1], cur[1], rob[0])

    # condition-wise performance (occluder class * occluder color)
    if exp == 'exp2':

        groupby = ['layer', 'cycle', 'metric']
        for rob in robustness_m.groupby(groupby):
            layer, cycle, metric = rob[0]
            """
            # trial-wise
            trials = trials_RT_all[
                (trials_RT_all.cycle == cycle) &
                (trials_RT_all.layer == layer)]
            outpath = (f'{plot_dir}/trial-wise_occluder_class'
                       f'-occluder_color_visibility'
                       f'_{metric}.pdf')
            if not op.isfile(outpath) or overwrite:
                condwise_robustness_plot(
                    df=trials,
                    outpath=outpath,
                    metric_column=f'RT_{metric}',
                    class_column='occluder_class',
                    color_column='occluder_color',
                    sample_column=None,
                    acc_only=True,
                    title='object visibility at RT',
                    legend_path=outpath.replace(metric, 'legend'))
            """
            # condition-wise
            outpath = (f'{plot_dir}/cyc{cycle:02}_{layer}_'
                       f'{metric}_cond-wise.pdf')
            if not op.isfile(outpath) or overwrite:
                vis_m = (rob[1]
                         .groupby(
                    ['occluder_class', 'occluder_color', 'visibility'])
                         .mean(numeric_only=True).reset_index()
                         .groupby(['occluder_class', 'occluder_color'])
                         .apply(lambda d: d[d.value > .5].visibility.min()))
                vis_m = pd.DataFrame(dict(visibility=vis_m))
                condwise_robustness_plot(
                    df=vis_m,
                    outpath=outpath,
                    metric_column='visibility',
                    class_column='occluder_class',
                    color_column='occluder_color',
                    sample_column=None,
                    title=None,
                    ylabel='visibility',
                    legend_path=outpath.replace(
                        f'{metric}_cond-wise', f'cond-wise_legend'))


def miscellaneous_plots(overwrite=False):
    """
    for metric in METRICS:
        out_path = (f'../data/in_silico/analysis/results/behavior/exp2'
                    f'/sample_trial_{metric}.pdf')
        if not op.isfile(out_path) or overwrite:
            trials = pd.read_parquet(
                f'{model_base}/alexnet/pretrained/behavioral/exp2'
                '/trials.parquet')
            values = trials[
                (trials.stimulus_id == 'a-0001') &
                (trials.layer == 'output') &
                (trials.metric == metric)].value.to_numpy()
            fig, ax = plt.subplots(figsize=(4, 2.5))
            plt.plot(VISIBILITIES['exp2'], values, marker='o', color='tab:blue')
            plt.xlabel('visibility')
            plt.ylabel(metric.replace('_', ' '))
            plt.tight_layout()
            fig.savefig(out_path)
        plt.close()


    # get_challenge_images
    public_models = model_contrasts['public_models']
    trials_public = pd.DataFrame()
    for m, (label, info) in enumerate(public_models.items()):
        path, color = info['path'], info['color']
        model_dir = f'{model_base}/{path}/behavioral/exp1'
        trials = pd.read_parquet(f'{model_dir}/trials.parquet')
        trials['model'] = label
        trials_public = pd.concat([trials_public, trials])

    sorted_trials = (trials_public.groupby(['visibility',
                                            'occluded_object_path'])
                     .agg({'accuracy': 'mean'})
                     .sort_values('accuracy')).reset_index()
    sorted_trials = sorted_trials[sorted_trials.accuracy ==
                                     0].sort_values('visibility', ascending=False)
    sorted_trials = sorted_trials[sorted_trials.visibility > .5]

    out_dir = '../data/in_silico/analysis/results/behavior/exp1/challenge_images'
    os.makedirs(out_dir, exist_ok=True)
    import shutil
    for i in range(1000):
        path_orig = sorted_trials.iloc[i].occluded_object_path
        path_end = path_orig.split("logFiles/")[1]
        path_new = (f'in_vivo/behavioral/exp1/data/'
                    f'{path_end}')
        assert op.isfile(path_new)
        shutil.copy(path_new, f'{out_dir}/{path_end.replace("/", "_")}')

    path_start=('/Users/tonglab/Desktop/Dave/p022/behavioral'
                '/v3_variousTypesLevels/logFiles')
    df = pd.DataFrame()
    for path in [
        f'{path_start}/25/stimuli/sportsCar_crossBarCardinal_0.2_black_0.png',
        f'{path_start}/33/stimuli/bison_naturalUntexturedCropped2_0.4_black_0.png',
        f'{path_start}/23/stimuli/elephant_polkadot_0.4_black_0.png']:

        df = pd.concat([df, trials_public[
            (trials_public.layer == 'output') &
            (trials_public.occluded_object_path == path)]])
    df = df[['model', 'class', 'prediction']]
    """


if __name__ == "__main__":

    from seconds_to_text import seconds_to_text

    start = time.time()

    recompare_models = False
    make_svc_dataset(overwrite=False)
    for m, model_dir in enumerate(model_dirs):
        overwrite = False
        overwrite = train_svc(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        overwrite = get_responses(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        overwrite = analyse_performance(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        plot_performance(model_dir, overwrite=overwrite)
        if overwrite:
            recompare_models = True

    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')
