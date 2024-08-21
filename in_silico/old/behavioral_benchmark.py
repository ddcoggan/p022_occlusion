'''
This scripts tests the accuracy of CNNs on classifying the exact images presented in the human behavioral experiment.
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
import itertools
import pingouin as pg
from datetime import datetime
from tqdm import tqdm
dtnow, nowstr = datetime.now, "%y/%m/%d %H:%M:%S"
np.random.seed(42)
TABCOLS = list(mcolors.TABLEAU_COLORS.keys())
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # converts to order in nvidia-smi (not in cuda)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # which device(s) to use

import torchvision.transforms as transforms

from .model_contrasts import model_contrasts, model_dirs, region_to_layer
from .helper_functions import get_trained_model, reorg_dict

sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from plot_utils import export_legend, custom_defaults
from math_functions import sigmoid
plt.rcParams.update(custom_defaults)

sys.path.append(op.expanduser('~/david/master_scripts/DNN/utils'))
from get_activations import get_activations
from predict import predict

sys.path.append(f'in_vivo/behavioral/exp1/analysis/scripts')
from classification_accuracy import CFG as BEHAV
from classification_accuracy import SDT


dataset_dir = op.expanduser(f'~/Datasets/ILSVRC2012')

transfer_methods = ['SVC', 'output']
EXP2_VIS = np.linspace(1, 360, 15, dtype=int)  # predict 15 equidistant frames
T = SimpleNamespace(classification=True, num_workers=8, nGPUs=-1, batch_size=32)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=[0.445, 0.445, 0.445],
                         std=[0.269, 0.269, 0.269]),
])


def make_SVC_dataset(overwrite=False):

    """ This function creates a set of images to train a linear support vector
    machine (SVC) classifier. The images are from the same classes as the
    behavioral experiments but are not the same images. """

    exp_dir = f'in_vivo/behavioral'
    dataset_dir = op.expanduser(f'~/Datasets/ILSVRC2012')
    SVC_images_path = ('in_silico/analysis/scripts/utils/SVC_images.csv')

    if not op.isfile(SVC_images_path) or overwrite:

        # get set of all images used in behavioral exps
        exp1_trials = pd.read_csv(
            f'{exp_dir}/exp1/data/all_trials.csv')
        exp1_images = [op.basename(x).split('.')[0] for x in 
                       exp1_trials.object_path.values]
        exp2_trials = pd.read_parquet(
            f'{exp_dir}/exp2/analysis/trials.parquet')
        exp2_images = [op.basename(x).split('.')[0] for x in
                       exp2_trials.object_path.values]
        behavioral_set = set(exp1_images + exp2_images)

        # create independent training set
        classes, images = [], []
        ims_per_class_svc = 256
        for class_dir, class_label in zip(BEHAV.class_dirs, BEHAV.classes):
            im_counter = 0
            image_paths = sorted(glob.glob(f'{dataset_dir}/train/{class_dir}/*'))
            while im_counter < ims_per_class_svc:
                image_path = image_paths.pop(0)
                if op.basename(image_path) not in behavioral_set:
                    classes.append(class_label)
                    images.append(image_path)
                    im_counter += 1
        SVC_images = pd.DataFrame({'class': classes, 'filepath': images})
        SVC_images.to_csv(SVC_images_path, index=False)


def make_PCA_dataset(overwrite=False):

    """ This function creates a set of images with which to measure
    the principle components of layer activations. Images selected are from
    the 992 imagenet_classes not used in the behavioral experiments. """

    dataset_dir = op.expanduser(f'~/Datasets/ILSVRC2012')

    PCA_images_path = ('in_silico/analysis/scripts/utils/PCA_images.csv')
    if not op.isfile(PCA_images_path) or overwrite:

        imagenet_classes = [op.basename(path) for path in sorted(glob.glob(
            f'{dataset_dir}/val/*'))]
        for d in BEHAV.class_dirs:
            imagenet_classes.remove(d)
        classes, images = [], []
        ims_per_class = 2
        for imagenet_class in imagenet_classes:
            images += sorted(glob.glob(
                f'{dataset_dir}/val/{imagenet_class}/*'))[:ims_per_class]
            classes += [imagenet_class] * ims_per_class
        PCA_images = pd.DataFrame({'class': classes, 'filepath': images})
        PCA_images.to_csv(PCA_images_path, index=False)


def train_SVC_classifier(model_dir, m=0, total_models=0, overwrite=False):

    svc_path = f'{model_dir}/behavioral/SVC.pkl'

    if not op.isfile(svc_path) or overwrite:

        overwrite = True
        os.makedirs(op.dirname(svc_path), exist_ok=True)
        model_info = model_dir.split('models/')[-1].split('/')
        model_name, identifier = model_info[:2]
        transfer_dir = model_info[2] if len(model_info) == 3 else 'X'
        layers = [region_to_layer[model_name]['IT']]

        print(f'{dtnow().strftime(nowstr)} Training SVC classifier, '
              f'model {m + 1}/{total_models}, {model_name}'
              f'/{identifier}/{transfer_dir}')

        # use PCA for dimensionality reduction
        print('Running PCA...')
        pca_images = pd.read_csv(
            'in_silico/analysis/scripts/utils/PCA_images.csv').filepath.values
        model = get_trained_model(model_dir, True, layers)
        activations = get_activations(
            model, model_name, pca_images, T=T, layers=layers,
            transform=transform, shuffle=True)
        activations = reorg_dict(activations)
        svcs = {key: {'pca': PCA().fit(value.reshape(
            [len(pca_images), -1]))} for key, value in activations.items()}

        # train a support vector machine on responses to the training set
        print('Running SVC...')
        svc_dataset = pd.read_csv(
            'in_silico/analysis/scripts/utils/SVC_images.csv')
        svc_images = svc_dataset['filepath'].values
        sampler = np.random.permutation(len(svc_images))
        svc_classes = [svc_dataset['class'].values[s] for s in sampler]
        model = get_trained_model(model_dir, True, layers)  # solves mem leak
        activations = get_activations(
            model, model_name, svc_images, T=T,
            layers=layers, transform=transform, sampler=sampler)
        activations = reorg_dict(activations)

        for key, act in activations.items():
            print(key)  # layer or layer_cycle
            pca_weights = svcs[key]['pca'].transform(
                act.reshape((len(svc_images), -1)))[:, :1000]
            clf = OneVsRestClassifier(BaggingClassifier(
                SVC(kernel='linear', probability=True),
                max_samples=1/16, n_estimators=16))
            clf.fit(pca_weights, svc_classes)
            train_acc = np.mean(clf.predict(pca_weights) == svc_classes)
            svcs[key]['svc'] = clf
            print(f'training accuracy ({key}): {train_acc:.4f}')

        pkl.dump(svcs, open(svc_path, 'wb'))

    return overwrite


def get_images(exp):

    exp_dir = f'in_vivo/behavioral/{exp}'
    if exp == 'exp1':
        trials = get_trials(exp)
        image_dir = f'{exp_dir}/data'
        images = trials.occluded_object_path.values
        images = [f'{image_dir}/{"/".join(x.split("/")[-3:])}' for x in images]
    elif exp == 'exp2':
        image_dir = f'{exp_dir}/stimuli/final'
        images = np.ravel([sorted(glob.glob(
            f'{image_dir}/set-?/frames/*/{v:03}.png')) for v in
            EXP2_VIS]).tolist()
    else:
        raise ValueError('exp must be "exp1" or "exp2"')

    return images


def get_trials(exp, drop_human=False):

    exp_dir = f'in_vivo/behavioral/{exp}'
    if exp == 'exp1':
        trials = pd.read_csv(f'{exp_dir}/data/all_trials.csv')
        if drop_human:
            trials.drop(columns=['noise_path', 'response', 'RT', 'accuracy',
                                 'animacy_task'], inplace=True)
    elif exp == 'exp2':
        trials = pd.read_parquet(f'{exp_dir}/analysis/trials.parquet')
        if drop_human:
            trials = trials[trials.subject.isin(['sub-01', 'sub-02'])]
            trials = trials.sort_values(by=['stimulus_id']).iloc[:, :13]
    else:
        raise ValueError('exp must be "exp1" or "exp2"')

    return trials



def get_responses(model_dir, m=0, total_models=0, exp=None, overwrite=False):

    model_info = model_dir.split('models/')[-1].split('/')
    model_name, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'
    layers = [region_to_layer[model_name]['IT'], 'output']
    results_dir = f'{model_dir}/behavioral/{exp}'
    out_path = f'{results_dir}/trials.parquet'

    if not op.isfile(out_path) or overwrite:

        print(f'{dtnow().strftime(nowstr)} Measuring responses for {exp} '
              f'stimuli, model {m + 1}/{total_models}, '
              f'{model_name}/{identifier}/{transfer_dir}')

        os.makedirs(op.dirname(out_path), exist_ok=True)
        images = get_images(exp)
        reload_every = 4000  # overcomes memory leak with some recurrent models
        num_blocks = np.ceil(len(images) / reload_every).astype(int)
        for b in range(num_blocks):

            print(f'Block {b + 1}/{num_blocks}')
            model = get_trained_model(model_dir, True, layers)  # reload model
            first = b * reload_every
            last = min(first + reload_every, len(images))
            image_block = images[first:last]
            activations_batch = get_activations(
                model, model_name, image_block, T=T, layers=layers,
                transform=transform, shuffle=False)
            activations_batch = reorg_dict(activations_batch)
            if b == 0:
                activations = activations_batch
            else:
                activations = {key: np.concatenate([activations[key],
                               activations_batch[key]], axis=0) for key in
                               activations}


        """
        # debugging: inspect responses
        outputs = get_activations(
        model, model_name, image_dir, T=T, output_only=True, 
        save_input_samples=True, sample_input_dir=M.outdir)['output']
        plt.imshow(model_responses)
        plt.show()
        """

        svcs = pkl.load(open(f'{op.dirname(results_dir)}/SVC.pkl', 'rb'))
        all_trials = pd.DataFrame()

        for transfer_method, layer in zip(transfer_methods, layers):
            for key in [k for k in activations.keys() if k.startswith(layer)]:

                trials = get_trials(exp, drop_human=True)

                # select activations and svc
                pca_obj, svc_obj = None, None
                if transfer_method == 'SVC':
                    pca_obj = svcs[key]['pca']
                    svc_obj = svcs[key]['svc']

                # get predictions
                print(f'Generating predictions for {transfer_method} ...')
                trials = get_predictions(
                    exp=exp, trials=trials, activations=activations[key],
                    method=transfer_method, pca_object=pca_obj,
                    svc_object=svc_obj)

                # print out a summary statistic
                if exp == 'exp1':
                    accuracy = trials.accuracy.mean()
                elif exp == 'exp2':
                    accuracy = np.mean([a[7] for a in trials.accuracy.values])
                print(f'{key} accuracy: {accuracy:.4f}')

                # get animacy task performamce
                #trials = get_animacy_performance(trials)

                cycle = key.split('_')[-1] if 'cyc' in key else 'cyc-1'
                trials['layer'] = layer
                trials['cycle'] = cycle
                trials['transfer_method'] = transfer_method

                all_trials = pd.concat(
                    [trials, all_trials]).reset_index(drop=True)

        all_trials.to_parquet(out_path, index=False)

        return True
    return False


def get_predictions(exp, trials, activations, method,
                    pca_object=None, svc_object=None):

    # check we have the right number of activations
    n_responses = len(trials) if exp == 'exp1' else len(trials) * len(EXP2_VIS)
    assert activations.shape[0] == n_responses, \
        'different number of trials and activations'

    # predictions based on output layer
    if method == 'output':
        resp_orig = predict(activations, 'ILSVRC2012', afc=BEHAV.class_idxs,
                            label_type='directory')
        responses = [
            BEHAV.classes[BEHAV.class_dirs.index(x)] for x in resp_orig]

    # predictions based on SVC using activations in layer corresponding to IT
    else:  # if method == 'SVC':
        pca_weights = pca_object.transform(
            activations.reshape((n_responses, -1)))[:, :1000]
        responses = []
        for t in tqdm(range(n_responses), unit='trial'):
            responses.append(svc_object.predict(
                pca_weights[t, :].reshape(1, -1))[0])

    # separate responses at different time points for exp2
    if exp == 'exp1':
        ground_truth = trials['class']
        accuracies = pd.Series(responses == ground_truth, dtype=int)
    elif exp == 'exp2':
        ground_truth = pd.Series(
            trials.object_ordinate.tolist() * len(EXP2_VIS))
        accuracies = pd.Series(responses == ground_truth, dtype=int)
        responses = np.array(responses).reshape(len(trials), len(EXP2_VIS))
        responses = pd.Series(
            [responses[t].tolist() for t in range(len(trials))])
        accuracies = np.array(accuracies).reshape(len(trials), len(EXP2_VIS))
        accuracies = pd.Series(
            [accuracies[t].tolist() for t in range(len(trials))])
    else:
        ValueError('exp must be "exp1" or "exp2"')

    trials['response'] = responses
    trials['accuracy'] = accuracies

    # get confidence in correct response (only for exp2, output method)
    if exp == 'exp2' and method == 'output':
        confidences = special.softmax(activations[:, BEHAV.class_idxs], axis=1)
        class_idxs = [BEHAV.classes.index(i) for i in ground_truth]
        conf_acc = [confidences[i,c] for i, c in enumerate(class_idxs)]
        conf_acc = np.array(conf_acc).reshape(len(trials), len(EXP2_VIS))
        conf_acc = pd.Series(
            [conf_acc[t].tolist() for t in range(len(trials))])
        trials['confidence'] = conf_acc

    return trials

"""
def get_animacy_performance(trials):

    animacy_task = []
    animate_classes = BEHAV.animate_classes
    inanimate_classes = BEHAV.inanimate_classes
    for t in range(len(trials)):
        image_class = trials['class'][t]
        response = trials['response'][t]
        if image_class in animate_classes and response in animate_classes:
            animacy_task.append('H')
        elif image_class in animate_classes and response in inanimate_classes:
            animacy_task.append('M')
        elif image_class in inanimate_classes and response in inanimate_classes:
            animacy_task.append('CR')
        elif image_class in inanimate_classes and response in animate_classes:
            animacy_task.append('FA')
        else:
            raise Exception('something went wrong')
    trials['animacy_task'] = animacy_task

    return trials
"""


def get_robustness(trials, exp):

    if exp == 'exp1':
        df = pd.DataFrame()
        subjects = trials.subject.unique()
        for transfer_method, subject, (occluder, visibility) in itertools.product(
                transfer_methods, subjects, BEHAV.occ_vis_combos):
            trials_m = trials[
                (trials['subject'] == subject) &
                (trials['occluder'] == occluder) &
                (trials['visibility'] == visibility) &
                (trials['transfer_method'] == transfer_method)]

            # classification accuracy
            accuracy = trials_m['accuracy'].mean()
            """
            # d' for animacy task
            hits = trials_m['animacy_task'].isin(['H']).sum()
            misses = trials_m['animacy_task'].isin(['M']).sum()
            CRs = trials_m['animacy_task'].isin(['CR']).sum()
            FAs = trials_m['animacy_task'].isin(['FA']).sum()
            SDT_results = SDT(hits, misses, FAs, CRs)
            d_prime = SDT_results["d'"]
            """
            df = pd.concat([df, pd.DataFrame({
                'transfer_method': [transfer_method],
                'subject': [subject],
                'occluder': [occluder],
                'visibility': [visibility],
                'accuracy': [accuracy],
                # "d'": [d_prime],
            })]).reset_index(drop=True)

    return df


def get_sigmoids(trials, exp):

    # mean for each occluder with sigmoid functions
    chance = 1 / 8
    sigmoids = pd.DataFrame()

    if exp == 'exp1':
        occluders = BEHAV.occluders
        xvals = [0] + BEHAV.visibilities + [1]
    elif exp == 'exp2':
        occluders = trials.occluder_ordinate.unique()
        xvals = np.linspace(0, 1, len(EXP2_VIS))


    # accuracy for each occluder * visibility
    for o, occluder in enumerate(occluders):

        if exp == 'exp1':
            acc_unocc = trials[trials['occluder'] == 'unoccluded'][
                'accuracy'].mean()
            acc_occ = trials[trials['occluder'] == occluder]
            accs_mean = acc_occ.groupby('visibility')['accuracy'].mean().values
            yvals = np.array([chance] + list(accs_mean) + [acc_unocc])
            acc_mean = np.mean(accs_mean)
        elif exp == 'exp2':
            yvals = np.mean(np.array(trials[trials['occluder_ordinate'] ==
                                     occluder]['accuracy'].values), axis=0)
            acc_mean = np.mean(yvals)

        # fit sigmoid function
        init_params = [max(yvals), np.median(xvals), 1, 0]
        try:
            popt, pcov = curve_fit(
                sigmoid, xvals, yvals, init_params, maxfev=10e5)
            curve_x = np.linspace(0, 1, 1000)
            curve_y = sigmoid(curve_x, *popt)
            threshold = sum(curve_y < .5) / 1000
        except:
            popt = [np.nan] * 4
            threshold = np.nan
        sigmoids = pd.concat(
            [sigmoids, pd.DataFrame({
                'occluder': [occluder],
                'metric': ['accuracy'],
                'L': [popt[0]],
                'x0': [popt[1]],
                'k': [popt[2]],
                'b': [popt[3]],
                'threshold': [threshold],
                'mean': [acc_mean]
            })]).reset_index(drop=True)

    return sigmoids


def get_human_likeness(trials_model, trials_human, exp):

    if exp == 'exp1':
        hum_lik = pd.DataFrame()
        for subject in trials_human.subject.unique():
            c_obss, c_errs, c_inaccs = [], [], []
            for occluder, visibility in BEHAV.occ_vis_combos:
                trials_m = trials_model[
                    (trials_model['subject'] == subject) &
                    (trials_model['occluder'] == occluder) &
                    (trials_model['visibility'] == visibility)]

                trials_h = trials_human[
                    (trials_human['subject'] == subject) &
                    (trials_human['occluder'] == occluder) &
                    (trials_human['visibility'] == visibility)]

                c_obs, c_err, c_inacc = measure_consistencies(trials_m, trials_h)
                c_obss.append(c_obs)
                c_errs.append(c_err)
                c_inaccs.append(c_inacc)

            hum_lik = pd.concat([hum_lik, pd.DataFrame({
                'c_obs': [np.nanmean(c_obss)],
                'c_err': [np.nanmean(c_errs)],
                'c_inacc': [np.nanmean(c_inaccs)]})])

    return hum_lik

    """
    # estimate separately for each visibility
    c_obss, c_errs, c_inaccs = [], [], []
    for visibility in BEHAV.visibilities + [1.]:
        trials_m = trials_model[
            (trials_model['subject'] == subject) &
            (trials_model['visibility'] == visibility) &
            (trials_model['layer'] == layer) &
            (trials_model['cycle'] == cycle)]

        trials_h = trials_human[
            (trials_human['subject'] == subject) &
            (trials_human['visibility'] == visibility)]

        c_obs, c_err, c_inacc = get_human_likeness(
            trials_m, trials_h)
        c_obss.append(c_obs)
        c_errs.append(c_err)
        c_inaccs.append(c_inacc)

    hum_lik = pd.concat([hum_lik, pd.DataFrame({
        'level': ['visibility'],
        'c_obs': [np.nanmean(c_obss)],
        'c_err': [np.nanmean(c_errs)],
        'c_inacc': [np.nanmean(c_inaccs)]})])

    # estimate for all stimuli
    trials_m = trials_model[
        (trials_model['subject'] == subject) &
        (trials_model['layer'] == layer) &
        (trials_model['cycle'] == cycle)]

    trials_h = trials_human[
        (trials_human['subject'] == subject)]

    c_obs, c_err, c_inacc = get_human_likeness(
        trials_m, trials_h)

    hum_lik = pd.concat([hum_lik, pd.DataFrame({
        'level': ['all_stimuli'],
        'c_obs': [c_obs],
        'c_err': [c_err],
        'c_inacc': [c_inacc]})])

    # unoccluded only
    trials_m = trials_model[
        (trials_model['subject'] == subject) &
        (trials_model['visibility'] == 1) &
        (trials_model['layer'] == layer) &
        (trials_model['cycle'] == cycle)]

    trials_h = trials_human[
        (trials_human['subject'] == subject) &
        (trials_human['visibility'] == 1)]

    c_obs, c_err, c_inacc = get_human_likeness(
        trials_m, trials_h)

    hum_lik = pd.concat([hum_lik, pd.DataFrame({
        'level': ['unoccluded'],
        'c_obs': [c_obs],
        'c_err': [c_err],
        'c_inacc': [c_inacc]})])

    hum_lik['layer'] = layer
    hum_lik['cycle'] = cycle
    hum_lik['subject'] = subject
    """

def layer_cycle_iterator(trials):
    for layer in trials.layer.unique():
        cycles = sorted(set(trials[trials.layer == layer].cycle.values))
        for cycle in cycles:
            yield layer, cycle

def analyse_performance(model_dir, m=0, total_models=0,
                        exp='exp1', overwrite=False):

    model_info = model_dir.split('models/')[-1].split('/')
    model_name, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'
    results_dir = f'{model_dir}/behavioral/{exp}'
    trials_model = pd.read_parquet(f'{results_dir}/trials.parquet')
    trials_human = get_trials(exp, drop_human=False)
    subjects = trials_human.subject.unique()
    mod_str = (f'model {m + 1}/{total_models}, '
               f'{model_name}/{identifier}/{transfer_dir}')

    # calculate occlusion robustness in each condition
    robustness_path = f'{results_dir}/occlusion_robustness.csv'
    if not op.isfile(robustness_path) or overwrite:
        now = dtnow().strftime(nowstr)
        print(f'{now} | Analysing robustness ({exp}) | {mod_str}')
        robustness = pd.DataFrame()
        for layer, cycle in layer_cycle_iterator(trials_model):
                trials_m = trials_model[
                        (trials_model['layer'] == layer) &
                        (trials_model['cycle'] == cycle)]
                df = get_robustness(trials_m, exp)
                df['layer'] = layer
                df['cycle'] = int(cycle[-2:])
                robustness = pd.concat([robustness, df]).reset_index(drop=True)
        robustness.to_csv(robustness_path, index=False)

    # calculate sigmoid parameters for each occluder and visibility (exp1 only)
    if exp == 'exp1':
        sigmoid_params_path = f'{results_dir}/robustness_sigmoids.csv'
        if not op.isfile(sigmoid_params_path) or overwrite:
            now = dtnow().strftime(nowstr)
            print(f'{now} | Calculating sigmoids ({exp}) | {mod_str}')
            sigmoids = pd.DataFrame()
            for layer, cycle in layer_cycle_iterator(trials_model):
                trials_m = trials_model[
                    (trials_model['layer'] == layer) &
                    (trials_model['cycle'] == cycle)]
                df = get_sigmoids(trials_m)
                df['layer'] = layer
                df['cycle'] = int(cycle[-2:])
                sigmoids = pd.concat([sigmoids, df]).reset_index(drop=True)
            sigmoids.to_csv(sigmoid_params_path, index=False)

    # calculate human likeness in each condition
    human_likeness_path = f'{results_dir}/human_likeness.csv'
    if not op.isfile(human_likeness_path) or overwrite:
        now = dtnow().strftime(nowstr)
        print(f'{now} | Analysing robustness {exp} | {mod_str}')
        likeness = pd.DataFrame()
        for layer, cycle in layer_cycle_iterator(trials_model):
            trials_m = trials_model[
                (trials_model['layer'] == layer) &
                (trials_model['cycle'] == cycle)]
            df = get_human_likeness(trials_m, trials_human, exp)
            df['layer'] = layer
            df['cycle'] = int(cycle[-2:])
            likeness = pd.concat([likeness, df]).reset_index(drop=True)
        likeness.to_csv(human_likeness_path, index=False)


def measure_consistencies(trials_m, trials_h):

    trials_m.reset_index(drop=True, inplace=True)
    trials_h.reset_index(drop=True, inplace=True)

    acc_m = trials_m['accuracy'].mean()
    acc_h = trials_h['accuracy'].mean()

    both_acc = (trials_m['accuracy'] * trials_h['accuracy']).mean()
    both_inacc = ((1-trials_m['accuracy']) * (1-trials_h['accuracy'])).mean()

    c_obs = both_acc + both_inacc
    c_err = error_consistency(acc_m, acc_h, c_obs)
    if both_inacc == 0:
        c_inacc = np.nan
    else:
        c_inacc = pd.Series(
            (trials_m['accuracy'] == 0) &
            (trials_m['response'] == trials_h['response']),
            dtype=int).mean() / both_inacc

    return c_obs, c_err, c_inacc


def error_consistency(hum, mod, c_obs):

    ceil = 1 - np.abs(hum - mod)  # ceil = 1 only if hum == mod
    chance_acc = hum * mod
    chance_inacc = (1-hum) * (1-mod)
    chance = chance_acc + chance_inacc
    coef = (c_obs - chance) / (ceil - chance) if ceil > chance else np.nan
    return coef


def plot_performance(model_dir, overwrite=False):

    colors = list(mcolors.TABLEAU_COLORS.keys())

    for transfer_method in transfer_methods:

        # output directory for within-model analyses
        out_dir = f'{model_dir}/behavioral/{transfer_method}'

        performance = pd.read_csv(f'{out_dir}/occlusion_robustness.csv')
        sigmoid_params = pd.read_csv(f'{out_dir}/robustness_sigmoids.csv')
        if 'cycle' not in performance.columns:
            performance['cycle'] = 0
            sigmoid_params['cycle'] = 0
            performance['layer'] = 'decoder' if transfer_method == 'output' \
                else 'IT'
            sigmoid_params['layer'] = 'decoder' if transfer_method == 'output' \
                else 'IT'

        layers = performance.layer.unique()
        for layer in layers:
            cycles = sorted(performance[performance.layer ==
                                        layer].cycle.unique())
            for cycle, layer, metric in itertools.product(
                    cycles, layers, ['accuracy', "d'"]):

                if metric == 'accuracy':
                    metric_label = 'accuracy'
                    ylabel = 'proportion accurate'
                    ylims = (0, 1.05)
                    yticks = (0, 1)
                    chance = 1 / 8
                else:  # if metric == "d'":
                    metric_label = 'd_prime'
                    ylabel = "d'"
                    ylims = (0, 4)
                    yticks = np.arange(6)
                    chance = 0

                these_perfs = performance[
                    (performance['cycle'] == cycle) &
                    (performance['layer'] == layer)]
                these_sigs = sigmoid_params[
                    (sigmoid_params['cycle'] == cycle) &
                    (sigmoid_params['layer'] == layer)]

                outpath = f'{out_dir}/cyc{cycle:02}_{layer}_{metric}.pdf'
                if not op.isfile(outpath) or overwrite:

                    xvals = BEHAV.visibilities + [1]
                    perf_unocc = these_perfs[
                        these_perfs['occluder'] == 'unoccluded'][metric].mean()

                    # accuracy for each occluder * visibility
                    fig, axes = plt.subplots(
                        3, 3, figsize=(4, 4), sharex=True, sharey=True)

                    for o, occluder in enumerate(BEHAV.occluders):

                        # select axis
                        ax_row = math.floor(o / 3)
                        ax_col = o % 3
                        ax = axes[ax_row, ax_col]

                        # get data for this occluder
                        perf_occ = these_perfs[these_perfs['occluder'] == occluder]
                        perf_mean = perf_occ.groupby('visibility')[
                            metric].mean().values

                        # plot accuracies
                        yvals = np.array(list(perf_mean) + [perf_unocc])
                        ax.scatter(xvals, yvals, color=colors[o])

                        # fit sigmoid function
                        if metric != 'RT':
                            popt = these_sigs[
                                (these_sigs['metric'] == metric_label) &
                                (these_sigs['occluder'] == occluder)][
                                ['L', 'x0', 'k', 'b']].values[0]
                            curve_x = np.linspace(0, 1, 1000)
                            curve_y = sigmoid(curve_x, *popt)
                            ax.plot(curve_x, curve_y, color=(0.2, 0.2, 0.2))

                        # format plot
                        ax.set_title(BEHAV.occluder_labels[o], size=7)
                        ax.set_xticks((0, 1))
                        ax.set_xlim((0, 1.05))
                        ax.set_yticks(yticks)
                        ax.set_ylim(ylims)
                        ax.tick_params(axis='both', which='major', labelsize=7)
                        # ax.axhline(y=acc1unalt, color=colors[0], linestyle='dashed')
                        if metric == 'accuracy':
                            ax.axhline(y=chance, color='k', linestyle='dotted')
                        if o == 7:
                            ax.set_xlabel('visibility', size=10)
                        if o == 3:
                            ax.set_ylabel(ylabel, size=10)

                    plt.tight_layout()
                    plt.savefig(outpath)
                    plt.close()


def compare_models(overwrite=False):

    exp_dir = f'in_vivo/behavioral/exp1'
    results_dir = f'in_silico/analysis/results/behavior/exp1'
    figsize_scatter = (2.5,2.5)
    human_thresh = pd.read_csv(f'{exp_dir}/analysis/sigmoid_params.csv')
    human_perf = pd.read_csv(f'{exp_dir}/analysis/performance.csv')
    transfer_methods = ['SVC', 'output']

    for (model_contrast, config), transfer_method in \
            itertools.product(model_contrasts.items(), transfer_methods):

        num_models = len(config)

        # occlusion robustness measures
        figsize = (2 + ((num_models + 1) / 4), 3)
        plot_cfgs = {
            'accuracy': {
                'title': 'classification accuracy',
                'ylabel': 'proportion accurate',
                'yticks': np.arange(0, 2, .2),
                'ylims': (0, 1),
                'chance': 1 / 8},
            "d'": {
                'title': 'animacy task (2-AFC)',
                'ylabel': "d'",
                'yticks': np.arange(0, 5, 1),
                'ylims': (0, 4),
                'chance': np.nan}}

        # collate data from each model
        model_perf = pd.DataFrame()
        #model_sigm = pd.DataFrame()
        for m, (label, info) in enumerate(config.items()):
            path, color = info['path'], info['color']

            model_dir = (f'in_silico/models/{path}/behavioral/{transfer_method}')

            these_perfs = pd.read_csv(f'{model_dir}/occlusion_robustness.csv')
            these_perfs['model'] = label
            model_perf = pd.concat([model_perf, these_perfs])

            #these_sigs = pd.read_csv(f'{model_dir}/robustness_sigmoids.csv')
            #these_sigs['model'] = label
            #model_sigm = pd.concat([model_sigm, these_sigs])


        #model_perf['cycle'].replace({np.nan: 0}, inplace=True)
        #model_perf['layer'].replace({np.nan: model_perf['layer'].values[-1]},
        #                           inplace=True)
        # make subset of final cycle in each layer
        model_perf_final = pd.DataFrame()
        for model in model_perf.model.unique():
            temp = model_perf[model_perf.model == model].copy(deep=True)
            for layer in temp.layer.unique():
                temp2 = temp[temp.layer == layer].copy(deep=True)
                last_cycle = sorted(temp2.cycle.unique())[-1]
                model_perf_final = pd.concat([
                    model_perf_final,
                    temp2[temp2.cycle == last_cycle]])


        for metric, plot_cfg in plot_cfgs.items():

            out_dir = (f'{results_dir}/{model_contrast}/{transfer_method}/'
                f'occlusion_robustness/{metric}')
            os.makedirs(out_dir, exist_ok=True)

            # analyse cycles
            if 'cycle' in model_perf.columns and model_perf.cycle.max() > 0:

                # make plot
                outpath = f'{out_dir}/{metric}_cyclewise.png'
                if not op.isfile(outpath) or overwrite:

                    fig, ax = plt.subplots(figsize=figsize)
                    unocc_width = 1 / ((num_models + 1) * 7)

                    # human robustness
                    data_hum = human_perf.drop(
                        columns='subject').groupby('occluder').agg('mean')
                    hum_unocc = data_hum[data_hum.index == 'unoccluded'][
                        metric].item()
                    hum_occ = data_hum[data_hum.index != 'unoccluded'][
                        metric].values
                    # sns.swarmplot(x=0, y=hum_occ, color=TABCOLS[:9],
                    #              size=4, edgecolor='white', linewidth=1)
                    plt.bar(0, np.mean(hum_unocc), color='#C0C0C0',
                            width=.5)
                    plt.bar(0, np.mean(hum_occ), color='tab:gray', width=.5)
                    #ax.axhline(y=hum_unocc, xmin=-.5, xmax=unocc_width*5,
                    #           color='#606060')

                    # robustness for each model
                    for m, (label, info) in enumerate(config.items()):
                        color = info['color']
                        data_mod = model_perf[
                            model_perf.model == label].groupby(
                            ['occluder', 'cycle']).agg(
                            'mean', numeric_only=True).reset_index()
                        for c in data_mod.cycle.unique():
                            xpos = m + 1 + (c * .15 - .3)
                            mod_unocc = data_mod[
                                (data_mod.occluder == 'unoccluded') &
                                (data_mod.cycle == c)][metric].item()
                            mod_occ = data_mod[
                                (data_mod.occluder != 'unoccluded') &
                                (data_mod.cycle == c)][metric].values
                            # sns.swarmplot(x=m + 1, y=mod_occ, color=TABCOLS[:9],
                            #              size=4, edgecolor='white', linewidth=1)
                            plt.bar(xpos, mod_unocc, color='#C0C0C0',
                                    width=.15)
                            plt.bar(xpos, np.mean(mod_occ), color=color,
                                    width=.15)

                            #xmin = (np.linspace(0,1,
                            #                    num_models+2)[m+1] +
                            #        .01 + (c*unocc_width))
                            #xmax = xmin + unocc_width
                            #print(xmin, xmax)
                            #ax.axhline(
                            #    y=mod_unocc, xmin=xmin, xmax=xmax,
                            #    color='#606060')

                    plt.yticks(plot_cfg['yticks'])
                    plt.ylim(plot_cfg['ylims'])
                    ax.axhline(y=plot_cfg['chance'], color='k',
                               linestyle='dotted')
                    plt.xticks([-2, -1])
                    plt.xlim(-.5, num_models + .5)
                    plt.ylabel(plot_cfg['ylabel'])
                    plt.title(plot_cfg['title'])
                    plt.tight_layout()
                    fig.savefig(outpath)
                    plt.close()


            # plots of just the final/only cycle
            outpath = f'{out_dir}/{metric}.png'
            if not op.isfile(outpath) or overwrite:

                fig, ax = plt.subplots(figsize=figsize)
                unocc_width = 1 / (num_models + 1)

                # human performance
                data_hum = human_perf.drop(
                    columns='subject').groupby('occluder').agg('mean').reset_index()
                hum_unocc = data_hum[data_hum.occluder == 'unoccluded'][
                    metric].item()
                hum_occ = data_hum[data_hum.occluder != 'unoccluded'][
                    metric].values
                sns.swarmplot(x=0, y=hum_occ, color=TABCOLS[:9],
                              size=2, edgecolor='white', linewidth=1)
                plt.bar(0, np.mean(hum_occ), color='tab:gray',
                        width=.5)
                ax.axhline(y=hum_unocc, xmin=-.5, xmax=unocc_width,
                           color='#606060')

                # performance for each model
                for m, (label, info) in enumerate(config.items()):
                    color = info['color']
                    data_mod = model_perf_final[
                        model_perf_final.model == label].groupby(
                        'occluder').agg('mean', numeric_only=True).reset_index()
                    mod_unocc = data_mod[data_mod.occluder == 'unoccluded'][
                        metric].item()
                    mod_occ = data_mod[data_mod.occluder != 'unoccluded'][
                        metric].values
                    sns.swarmplot(x=m + 1, y=mod_occ, color=TABCOLS[:9],
                                  size=2, edgecolor='white', linewidth=1)
                    plt.bar(m + 1, np.mean(mod_occ), color=color, width=.5)
                    xmin = (m + 1) * unocc_width
                    xmax = xmin + unocc_width
                    ax.axhline(y=mod_unocc, xmin=xmin, xmax=xmax, color='#606060')
                plt.yticks(plot_cfg['yticks'])
                plt.ylim(plot_cfg['ylims'])
                ax.axhline(y=plot_cfg['chance'], color='k', linestyle='dotted')
                plt.xticks([-2, -1])
                plt.xlim(-.5, num_models + .5)
                plt.ylabel(plot_cfg['ylabel'])
                plt.title(plot_cfg['title'])
                plt.tight_layout()
                fig.savefig(outpath)
                plt.close()

            # plots of just the final/only cycle
            outpath = f'{out_dir}/{metric}_errorbar.png'
            if not op.isfile(outpath) or overwrite:

                fig, ax = plt.subplots(figsize=figsize)
                unocc_width = 1 / (num_models + 1)

                # human performance
                data_hum = human_perf.groupby(['subject','occluder']).agg(
                    'mean', numeric_only=True).reset_index()
                hum_unocc = data_hum[data_hum.occluder == 'unoccluded'][
                    metric].mean()
                hum_occ = data_hum[
                    data_hum.occluder != 'unoccluded'].groupby('subject').agg(
                    'mean', numeric_only=True)[metric].values
                plt.bar(0, np.mean(hum_occ), color='tab:gray',
                        width=.5)
                plt.errorbar(0, np.mean(hum_occ), yerr=stats.sem(hum_occ),
                    color='k', capsize=2)
                ax.axhline(y=hum_unocc, xmin=-.5, xmax=unocc_width,
                           color='#606060')

                # performance for each model
                for m, (label, info) in enumerate(config.items()):
                    color = info['color']
                    data_mod = model_perf_final[
                        model_perf_final.model == label].groupby(
                        ['subject','occluder']).agg(
                        'mean', numeric_only=True).reset_index()

                    mod_unocc = data_mod[data_mod.occluder == 'unoccluded'][
                        metric].mean()
                    mod_occ = data_mod[
                        data_mod.occluder != 'unoccluded'].groupby(
                        'subject').agg(
                        'mean', numeric_only=True)[metric].values
                    plt.bar(m+1, np.mean(mod_occ), color=color, width=.5)
                    plt.errorbar(m+1, np.mean(mod_occ), yerr=stats.sem(mod_occ),
                                 color='k', capsize=2)
                    xmin = (m + 1) * unocc_width
                    xmax = xmin + unocc_width
                    ax.axhline(y=mod_unocc, xmin=xmin, xmax=xmax,
                               color='#606060')
                plt.yticks(plot_cfg['yticks'])
                plt.ylim(plot_cfg['ylims'])
                ax.axhline(y=plot_cfg['chance'], color='k',
                           linestyle='dotted')
                plt.xticks([-2, -1])
                plt.xlim(-.5, num_models + .5)
                plt.ylabel(plot_cfg['ylabel'])
                plt.title(plot_cfg['title'])
                plt.tight_layout()
                fig.savefig(outpath)
                plt.close()


            # legend
            outpath = f'{out_dir}/legend.pdf'
            leg_colors = ['tab:gray'] + [m['color'] for m in config.values()]
            leg_labels = ['humans'] + list(config.keys())
            if not op.isfile(outpath) or overwrite:
                f = lambda m, c: \
                plt.plot([], [], marker=m, color=c, linestyle="None")[0]
                handles = [f('s', color) for color in leg_colors]
                legend = plt.legend(handles, leg_labels, loc=3)
                export_legend(legend, filename=outpath)


            # ANOVA
            anova_path = f'{out_dir}/{metric}_anova.csv'
            ph_path = anova_path.replace('anova', 'posthocs')
            if not op.isfile(anova_path) or not op.isfile(ph_path) or overwrite:
                model_perf_sum = model_perf_final[
                    model_perf_final.visibility < 1].groupby(
                    ['model', 'subject']).agg(
                    'mean', numeric_only=True).reset_index()
                anova = pg.rm_anova(
                    dv=metric, within='model', subject='subject',
                    data=model_perf_sum, detailed=True)
                post_hocs = pg.pairwise_tests(
                    dv=metric, within='model', subject='subject',
                    data=model_perf_sum, padjust='bonf')
                anova.to_csv(f'{out_dir}/{metric}_anova.csv')
                post_hocs.to_csv(f'{out_dir}/{metric}_posthocs.csv')


            # plot sigmoid curves
            if metric == 'accuracy':
                ylims = (0, 1.02)
                yticks = (0, 1)
                chance = 1 / 8
            else:  # if metric == "d'":
                ylims = (0, 4)
                yticks = np.arange(6)
                chance = 0

            outpath = f'{out_dir}/{metric}_curves.pdf'
            if not op.isfile(outpath) or overwrite:

                fig, ax = plt.subplots(figsize=(4, 4))

                # get human data
                human_acc = human_perf.drop(columns='subject').groupby(
                    'visibility').agg('mean', numeric_only=True)[
                    metric].values
                xvals = [0] + BEHAV.visibilities + [1]
                yvals = np.array([chance] + list(human_acc))
                init_params = [max(yvals), np.median(xvals), 1, 0]
                p = curve_fit(sigmoid, xvals, yvals, init_params, maxfev=100000)
                curve_x = np.linspace(0, 1, 1000)
                curve_y = sigmoid(curve_x, *p[0])
                threshold = sum(curve_y < .5) / 1000
                ax.scatter(xvals[1:], yvals[1:], color='tab:gray', s=2)
                ax.plot(curve_x, curve_y, color='tab:gray')

                # models
                for m, (label, info) in enumerate(config.items()):
                    color = info['color']
                    model_acc = model_perf[
                        model_perf.model == label].groupby(
                        'visibility').agg('mean', numeric_only=True)[
                        metric].values
                    yvals = np.array([chance] + list(model_acc))
                    init_params = [max(yvals), np.median(xvals), 1, 0]
                    curve_x = np.linspace(0, 1, 1000)
                    try:
                        p = curve_fit(sigmoid, xvals, yvals, init_params,
                                      maxfev=100000)
                        curve_y = sigmoid(curve_x, *p[0])
                    except:
                        curve_y = np.ones_like(curve_x) / 8
                    # threshold = sum(curve_y < .5) / 1000
                    ax.scatter(xvals[1:], yvals[1:], color=color, s=2)
                    ax.plot(curve_x, curve_y, color=color)

                # format plot
                # ax.set_title(BEHAV.occluder_labels[o], size=7)
                ax.set_xticks(xvals[1:])
                ax.set_xlim((0, 1.02))
                ax.set_yticks(yticks)
                ax.set_ylim(ylims)
                ax.tick_params(axis='both', which='major',
                               labelsize=7)
                if metric == 'accuracy':
                    ax.axhline(y=chance, color='k',
                               linestyle='dotted')
                ax.set_xlabel('visibility')
                ax.set_ylabel(metric)
                plt.tight_layout()
                fig.savefig(outpath)
                plt.close()


        # human likeness measures
        figsize = (2 + (num_models / 4), 3)

        # condition-level comparisons (uses robustness data)
        plot_cfgs = {
            'occluder_accuracy': {
                'title': 'accuracy similarity to humans\n(condition-wise)',
                'ylabel': "Correlation across 9 occluders ($\it{r}$)",
                'yticks': np.arange(-1, 2, .2),
                'ylims': (-.5, 1),
                'chance': 0},
            "occluder_d'": {
                'title': 'd\' similarity to humans\n(condition-wise)',
                'ylabel': "Correlation across 9 occluders ($\it{r}$)",
                'yticks': np.arange(0, 2, .1),
                'ylims': (0, .5),
                'chance': np.nan},
            'occluder-visibility_accuracy': {
                'title': 'accuracy similarity to humans\n(condition-wise)',
                'ylabel': "Correlation across 9 occ * 5 vis ($\it{r}$)",
                'yticks': np.arange(-1, 2, .2),
                'ylims': (0, .8),
                'chance': 0},
            "occluder-visibility_d'": {
                'title': 'd\' similarity to humans\n(condition-wise)',
                'ylabel': "Correlation across 9 occ * 5 vis ($\it{r}$)",
                'yticks': np.arange(0, 2, .1),
                'ylims': (0, .4),
                'chance': np.nan}}

        for metric, plot_cfg in plot_cfgs.items():

            out_dir = (f'{results_dir}/{model_contrast}/'
                       f'{transfer_method}/human_likeness/{metric}')
            os.makedirs(out_dir, exist_ok=True)

            # analyse cycles
            if 'cycle' in model_perf.columns and model_perf.cycle.max() > 0:

                # make plot
                outpath = f'{out_dir}/{metric}_cyclewise.png'

                if not op.isfile(outpath) or overwrite:

                    fig, ax = plt.subplots(figsize=figsize)

                    # correlation between performance agg. across conditions
                    metric_info = metric.split('_')
                    base_metric = metric_info[-1]
                    grouping_vars = metric_info[0].split('-')

                    # human_values
                    hum_vals = []
                    for subject in human_perf.subject.unique():
                        hum_data = human_perf[
                            (human_perf.subject == subject) &
                            (human_perf.occluder != 'unoccluded')].groupby(
                            grouping_vars).agg('mean').reset_index()
                        if 'visibility' in grouping_vars:
                            # subtract mean for each visibility
                            hum_data[base_metric] -= hum_data.groupby(
                                'visibility').agg(
                                'mean',
                                numeric_only=True).reset_index()[
                                                         base_metric].to_list() * 9
                        hum_vals.append(hum_data[base_metric].values)
                    hum_vals = np.array(hum_vals)

                    # noise_ceiling
                    grp = np.mean(hum_vals, axis=0)
                    subs = np.arange(hum_vals.shape[0])
                    lwr, upr = [], []
                    for s in subs:
                        ind = hum_vals[s]
                        rem_grp = np.mean(hum_vals[subs != s], axis=0)
                        lwr.append(np.corrcoef(ind, rem_grp)[0, 1])
                        upr.append(np.corrcoef(ind, grp)[0, 1])
                    lwr, upr = np.mean(lwr), np.mean(upr)
                    ax.fill_between(np.arange(-1, 50), lwr, upr,
                                    color='lightgray', lw=0)

                    # model values
                    for m, (label, info) in enumerate(config.items()):
                        color = info['color']
                        data_mod = model_perf[
                            (model_perf.model == label) &
                            (model_perf.occluder != 'unoccluded')].groupby(
                            grouping_vars + ['cycle']).agg(
                            'mean', numeric_only=True).reset_index()
                        num_cycles = len(data_mod.cycle.unique())
                        for c in data_mod.cycle.unique():
                            cycle_data = data_mod[data_mod.cycle == c].copy(
                                deep=True)
                            xpos = m + ((c - (num_cycles // 2)) * .15)
                            if 'visibility' in grouping_vars:
                                # subtract mean for each visibility
                                cycle_data[base_metric] -= \
                                    cycle_data.groupby(
                                        'visibility').agg(
                                        'mean',
                                        numeric_only=True).reset_index(
                                    )[base_metric].to_list() * 9
                            mod_vals = cycle_data[base_metric].values
                            sims = [
                                np.corrcoef(hum_vals[s], mod_vals)[0,
                                1] for s in subs]
                            plt.bar(xpos, np.mean(sims), color=color, width=.15)
                            plt.errorbar(
                                xpos, np.mean(sims), yerr=stats.sem(sims),
                                color='k', capsize=2)

                    plt.yticks(plot_cfg['yticks'])
                    plt.ylim(plot_cfg['ylims'])
                    ax.axhline(y=plot_cfg['chance'], color='k',
                               linestyle='dotted')
                    plt.xticks([-2, -1])
                    plt.xlim(-.5, num_models - .5)
                    plt.ylabel(plot_cfg['ylabel'])
                    plt.title(plot_cfg['title'])
                    plt.tight_layout()
                    fig.savefig(outpath)
                    plt.close()

            # analyse final/only cycle
            outpath = f'{out_dir}/{metric}.png'
            if not op.isfile(outpath) or overwrite:

                # errorbar plots
                fig, ax = plt.subplots(figsize=figsize)

                # correlation between performance agg. across conditions
                metric_info = metric.split('_')
                base_metric = metric_info[-1]
                grouping_vars = metric_info[0].split('-')

                # human_values
                hum_vals = []
                for subject in human_perf.subject.unique():
                    hum_data = human_perf[
                        (human_perf.subject == subject) &
                        (human_perf.occluder != 'unoccluded')].groupby(
                        grouping_vars).agg('mean').reset_index()
                    if 'visibility' in grouping_vars:
                        # subtract mean for each visibility
                        hum_data[base_metric] -= hum_data.groupby(
                            'visibility').agg(
                            'mean', numeric_only=True).reset_index()[
                                                     base_metric].to_list() * 9
                    hum_vals.append(hum_data[base_metric].values)
                hum_vals = np.array(hum_vals)


                # noise_ceiling
                grp = np.mean(hum_vals, axis=0)
                subs = np.arange(hum_vals.shape[0])
                lwr, upr = [], []
                for s in subs:
                    ind = hum_vals[s]
                    rem_grp = np.mean(hum_vals[subs != s], axis=0)
                    lwr.append(np.corrcoef(ind, rem_grp)[0, 1])
                    upr.append(np.corrcoef(ind, grp)[0, 1])
                lwr, upr = np.mean(lwr), np.mean(upr)
                ax.fill_between(np.arange(-1, 50), lwr, upr,
                                color='lightgray', lw=0)

                # model values
                similarities = pd.DataFrame()
                for m, (label, info) in enumerate(config.items()):
                    color = info['color']

                    data_mod = model_perf_final[
                        (model_perf_final.model == label) &
                        (model_perf_final.occluder != 'unoccluded')].groupby(
                        grouping_vars).agg(
                        'mean', numeric_only=True).reset_index()
                    if 'visibility' in grouping_vars:
                        # subtract mean for each visibility
                        data_mod[base_metric] -= data_mod.groupby(
                            'visibility').agg(
                            'mean', numeric_only=True).reset_index(
                        )[base_metric].to_list() * 9
                    mod_vals = data_mod[base_metric].values
                    sims = [np.corrcoef(hum_vals[s], mod_vals)[0,
                        1] for s in subs]
                    plt.bar(m, np.mean(sims), color=color, width=.5)
                    plt.errorbar(m, np.mean(sims), yerr=stats.sem(sims),
                                 color='k', capsize=2)

                    similarities = pd.concat([
                        similarities, pd.DataFrame({
                            'model': [label]*len(subs),
                            'subject': subs,
                            'r': sims})])

                plt.yticks(plot_cfg['yticks'])
                plt.ylim(plot_cfg['ylims'])
                ax.axhline(y=plot_cfg['chance'], color='k',
                           linestyle='dotted')
                plt.xticks([-2, -1])
                plt.xlim(-.5, num_models - .5)
                plt.ylabel(plot_cfg['ylabel'])
                plt.title(plot_cfg['title'])
                plt.tight_layout()
                fig.savefig(outpath)
                plt.close()

                # ANOVA
                anova = pg.rm_anova(
                    dv='r', within='model', subject='subject',
                    data=similarities, detailed=True)
                post_hocs = pg.pairwise_tests(
                    dv='r', within='model', subject='subject',
                    data=similarities, padjust='bonf')
                anova.to_csv(f'{out_dir}/{metric}_anova.csv')
                post_hocs.to_csv(f'{out_dir}/{metric}_posthocs.csv')

                # save performance profile
                similarities.to_csv(
                    f'{out_dir}/{metric}_similarities.csv')
                sim_summary = similarities.drop(columns='subject').groupby(
                    'model').agg(['mean', 'sem'], numeric_only=True).reset_index()
                sim_summary.columns = ['model', 'mean', 'sem']
                sim_summary = pd.concat([
                    sim_summary, pd.DataFrame({
                        'model': ['nc_lwr', 'nc_upr'],
                        'mean': [lwr, upr],
                        'sem': [np.nan, np.nan]})])
                sim_summary.to_csv(
                    f'{out_dir}/{metric}_similarities_summary.csv')


        # image-level comparisons
        plot_cfgs = {
            'c_inacc': {
                'title': 'inaccurate consistency with humans\n(image-wise)',
                'ylabel': 'inaccurate consistency',
                'yticks': np.arange(0, 1, .1),
                'ylims': (0, .3),
                'chance': 1 / 7},
            'c_obs': {
                'title': 'observed consistency with humans\n(image-wise)',
                'ylabel': 'observed consistency',
                'yticks': np.arange(0, 2, .1),
                'ylims': (.5, .8),
                'chance': np.nan},
            'c_err': {
                'title': 'error consistency with humans\n(image-wise)',
                'ylabel': "Cohen's kappa",
                'yticks': np.arange(0, .5, .1),
                'ylims': (0, .3),
                'chance': 0}}

        # collate data from each model
        model_perf = pd.DataFrame()
        for m, (label, info) in enumerate(config.items()):
            path, color = info['path'], info['color']
            these_perfs = pd.read_csv(f'in_silico/models/{path}/behavioral/'
                         f'{transfer_method}/human_likeness.csv')
            these_perfs['model'] = label
            model_perf = pd.concat([model_perf, these_perfs])

        # make subset of final cycle in each layer
        model_perf_final = pd.DataFrame()
        for model in model_perf.model.unique():
            temp = model_perf[model_perf.model == model].copy(deep=True)
            for layer in temp.layer.unique():
                temp2 = temp[temp.layer == layer].copy(deep=True)
                last_cycle = sorted(temp2.cycle.unique())[-1]
                model_perf_final = pd.concat([
                    model_perf_final,
                    temp2[temp2.cycle == last_cycle]])

        for level, (metric, plot_cfg) in itertools.product(
                ['all_stimuli','visibility','occluder_visibility',
                 'unoccluded'],
                plot_cfgs.items()):

            out_dir = (f'{results_dir}/{model_contrast}/'
                       f'{transfer_method}/human_likeness/{metric}')
            os.makedirs(out_dir, exist_ok=True)

            # analyse cycles
            outpath = f'{out_dir}/{metric}_{level}_cyclewise.png'
            if not op.isfile(outpath) or overwrite:
                fig, ax = plt.subplots(figsize=figsize)
                for m, (label, info) in enumerate(config.items()):
                    color = info['color']
                    data_mod = model_perf[
                        (model_perf.model == label) &
                        (model_perf.level == level)].groupby([
                            'subject', 'cycle']).agg(
                            'mean', numeric_only=True).reset_index()
                    num_cycles = len(data_mod.cycle.unique())
                    for c in data_mod.cycle.unique():
                        cycle_data = data_mod[data_mod.cycle == c]
                        xpos = m + ((max(0, int(c[3:])) - (num_cycles / 2)) *
                                     .15)
                        mod_mean = cycle_data[metric].mean()
                        mod_sem = cycle_data[metric].sem()
                        plt.bar(xpos, mod_mean, color=color,
                                width=.15)
                        plt.errorbar(xpos, mod_mean, yerr=mod_sem,
                                     color='k', capsize=2)

                plt.yticks(plot_cfg['yticks'])
                ax.axhline(y=plot_cfg['chance'], color='k',
                           linestyle='dotted')
                plt.xticks([-2, -1])
                plt.xlim(-.5, num_models - .5)
                plt.ylabel(plot_cfg['ylabel'])
                plt.ylim(plot_cfg['ylims'])
                plt.title(plot_cfg['title'])
                plt.tight_layout()
                fig.savefig(outpath)
                plt.close()

            # analyse final/only cycle
            outpath = f'{out_dir}/{metric}_{level}.png'
            if not op.isfile(outpath) or overwrite:
                fig, ax = plt.subplots(figsize=figsize)
                for m, (label, info) in enumerate(config.items()):
                    color = info['color']
                    data_mod = model_perf_final[
                        (model_perf_final.model == label) &
                        (model_perf_final.level == level)].groupby(
                        'subject').agg('mean', numeric_only=True)
                    mod_mean = data_mod[metric].mean()
                    mod_sem = data_mod[metric].sem()
                    plt.bar(m, mod_mean, color=color, width=.5)
                    plt.errorbar(m, mod_mean, yerr=mod_sem, color='k',
                                 capsize=2)
                plt.yticks(plot_cfg['yticks'])
                ax.axhline(y=plot_cfg['chance'], color='k', linestyle='dotted')
                plt.xticks([-2, -1])
                plt.xlim(-.5, num_models - .5)
                plt.ylabel(plot_cfg['ylabel'])
                plt.ylim(plot_cfg['ylims'])
                plt.title(plot_cfg['title'])
                plt.tight_layout()
                fig.savefig(outpath)
                plt.close()

                # ANOVA
                model_perf_sum = model_perf_final.groupby(
                    ['model', 'subject']).agg(
                    'mean', numeric_only=True).reset_index()
                anova = pg.rm_anova(
                    dv=metric, within='model', subject='subject',
                    data=model_perf_sum, detailed=True)
                post_hocs = pg.pairwise_tests(
                    dv=metric, within='model', subject='subject',
                    data=model_perf_sum, padjust='bonf')
                anova.to_csv(f'{out_dir}/{metric}_{level}_anova.csv')
                post_hocs.to_csv(f'{out_dir}/{metric}_{level}_posthocs.csv')

        # legend
        outpath = f'{out_dir}/legend.pdf'
        leg_colors = [m['color'] for m in config.values()]
        if not op.isfile(outpath) or overwrite:
            f = lambda m, c: \
            plt.plot([], [], marker=m, color=c, linestyle="None")[0]
            handles = [f('s', color) for color in leg_colors]
            legend = plt.legend(handles, list(config.keys()), loc=3)
            export_legend(legend, filename=outpath)

    # save occluder legend separately
    outpath = f'{results_dir}/occluder_types_legend.pdf'
    if not op.isfile(outpath) or overwrite:
        f = lambda m, c: plt.plot(
            [], [], marker=m, markerfacecolor=c, color='white')[0]
        handles = [f('o', TABCOLS[c]) for c in range(len(
            BEHAV.occluders) - 1)]
        labels = BEHAV.occluders
        legend = plt.legend(handles, labels, loc=3)
        export_legend(legend, filename=outpath)

    # save occluder legend separately
    outpath = f'{results_dir}/occluder_types_legend_w-unocc.pdf'
    if not op.isfile(outpath) or overwrite:
        f = lambda m, c, mo, ls: plt.plot(
            [], [], marker=m, markerfacecolor=c, color=mo,
            linestyle=ls)[0]
        handles = [f(None, '#606060', '#606060', None)]
        handles += [f('o', TABCOLS[c], 'white', None) for c in
                     range(len(BEHAV.occluders))]
        labels = ['unoccluded'] + BEHAV.occluders
        legend = plt.legend(handles, labels, loc=3)
        export_legend(legend, filename=outpath)


if __name__ == "__main__":

    from seconds_to_text import seconds_to_text
    start = time.time()

    recompare_models = False
    make_SVC_training_set(overwrite=False)
    for m, model_dir in enumerate(model_dirs):
        overwrite = False
        overwrite = train_SVC_classifier(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        overwrite = get_responses(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        overwrite = analyse_performance(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        plot_performance(model_dir, overwrite=overwrite)
        if overwrite:
            recompare_models = True
    compare_models(overwrite=recompare_models)

    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')


