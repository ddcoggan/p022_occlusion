'''
This scripts tests the accuracy of CNNs on classifying the exact images presented in the human behavioral experiment.
'''

import os
import os.path as op
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle as pkl
import pandas as pd
from scipy.optimize import curve_fit
import math
import time
from types import SimpleNamespace
from sklearn.decomposition import PCA
from sklearn import svm
import itertools
from datetime import datetime
dtnow, nowstr = datetime.now, "%d/%m/%y %H:%M:%S"

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # converts to order in nvidia-smi (not in cuda)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # which device(s) to use

import torchvision.transforms as transforms


sys.path.append('in_silico/analysis/scripts')
sys.path.append(op.expanduser('~/david/master_scripts'))
from misc.seconds_to_text import seconds_to_text
from misc.plot_utils import export_legend, custom_defaults
from misc.math_functions import sigmoid
plt.rcParams.update(custom_defaults)

from in_silico.analysis.scripts.test_utils.config import PROJ_DIR, TABCOLS, model_dirs
sys.path.append(f'{PROJ_DIR}/in_vivo/behavioral/exp1/analysis/scripts')
from classification_accuracy import CFG as BEHAV


def make_SVC_training_set():

    behavioural_dir = f'in_vivo/behavioral/exp1'
    subjects = [op.basename(x) for x in glob.glob(f'{behavioural_dir}/logFiles/*') if op.isdir(x)]
    dataset_dir = op.expanduser(f'~/Datasets/ILSVRC2012')

    # make list of training images for linear SVC. These must be independent from those used in behavioral study
    SVM_train_path = 'in_silico/analysis/scripts/linear_SVC_training_set.csv'
    if not op.isfile(SVM_train_path) or overwrite:

        # get set of all images used in behavioral
        behavioural_images = []
        for s, subject in enumerate(subjects):
            subject_dir = f'{behavioural_dir}/logFiles/{subject}'
            trials = pkl.load(open(f'{subject_dir}/trials.pkl', 'rb'))
            filenames = [op.basename(path) for path in trials.objectPath.values]
            behavioural_images += filenames
        behavioural_set = set(behavioural_images)

        # create non-overlapping set
        categories = []
        train_paths = []
        ims_per_class_svm = 256
        for class_dir, class_label in zip(BEHAV.class_dirs, 
                                          BEHAV.class_labels_alt):
            im_counter = 0
            image_paths = sorted(glob.glob(f'{dataset_dir}/train/{class_dir}/*'))
            while im_counter < ims_per_class_svm:
                image_path = image_paths.pop(0)
                if op.basename(image_path) not in behavioural_set:
                    categories.append(class_label)
                    train_paths.append(image_path)
                    im_counter += 1
        SVM_train_set = pd.DataFrame({'category': categories,
                                      'filepath': train_paths})
        SVM_train_set.to_csv(SVM_train_path)



def train_SVM_classifier(model_dir, m=0, total_models=0, overwrite=False):

    SVM_train_path = \
        'in_silico/analysis/scripts/test_utils/linear_SVC_training_set.csv'
    SVM_train_set = pd.read_csv(open(SVM_train_path, 'r+'))
    dataset_dir = op.expanduser(f'~/Datasets/ILSVRC2012')
    LAYERS = ['IT']

    # output directory for within-model analyses
    outdir = f'{model_dir}/behavioral'
    os.makedirs(outdir, exist_ok=True)

    pca_path = f'{model_dir}/behavioral/PCA.pkl'
    svm_path = f'{model_dir}/behavioral/SVM.pkl'
    load_model = not op.isfile(pca_path) or not op.isfile(svm_path) or overwrite

    if load_model:

        print(f'model {m}/{total_models} | {model_dir}')

        # get model
        model_name = model_dir.split('/')[2]
        if model_name != 'cornet_s_custom':
            M = SimpleNamespace(model_dir=model_dir)
            T = SimpleNamespace()

        else:
            CFG = pkl.load(open(f'{model_dir}/config.pkl', 'rb'))
            M, T = CFG.M, CFG.T
        M.model_name = model_name
        if not hasattr(M, 'model'):
            from DNN.utils import get_model
            print('loading model...')
            M.model = get_model(M)

        # load parameters
        from DNN.utils import load_params
        print('loading parameters...')
        params_path = sorted(glob.glob(f"{model_dir}/params/*.pt"))[-1]
        M.model = load_params(params_path, M.model, 'model')
        M.params_loaded = True  # stops params being reloaded

        # set model to classify
        T.classification = True
        T.num_workers = 4
        T.nGPUs = -1
        T.batch_size = 32


    # create transfer learning model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize(mean=[0.445, 0.445, 0.445],
                             std=[0.269, 0.269, 0.269]),
    ])

    # perform PCA based on responses to 992 imagenet val images of
    # unused categories
    if not op.isfile(pca_path) or overwrite:

        overwrite = True

        print('Running PCA...')
        imagenet_classes = [op.basename(path) for path in sorted(glob.glob(
            f'{dataset_dir}/val/*'))]
        image_paths_pca = []
        ims_per_class_pca = 2
        for imagenet_class in imagenet_classes:
            if imagenet_class not in BEHAV.class_dirs:
                image_paths_pca += sorted(glob.glob(
                    f'{dataset_dir}/val/{imagenet_class}/*'))[:ims_per_class_pca]
        total_images = len(image_paths_pca)

        from DNN.utils import get_activations
        activations = get_activations(M, image_paths_pca, T=T, layers=LAYERS, 
                                      transform=transform, shuffle=True)

        layerwise_pca = {}
        for layer in LAYERS:
            activations_layer = activations[layer]
            if isinstance(activations_layer, list):
                activations_layer = activations_layer[0]
            #activations_layer = activations_layer.detach().cpu()
            layerwise_pca[layer] = PCA().fit(
                activations_layer.reshape([total_images, -1]))

        pkl.dump(layerwise_pca, open(pca_path, 'wb'))

    else:
        layerwise_pca = pkl.load(open(pca_path, 'rb'))

    # train a support vector machine on responses to the training set
    if not op.isfile(svm_path) or overwrite:

        overwrite = True

        print('Running SVM...')
        layerwise_svm = {layer: {} for layer in LAYERS}

        from DNN.utils import get_activations
        sampler = np.random.permutation(len(SVM_train_set))
        activations = get_activations(M, SVM_train_set.filepath.values, T=T,
                                      layers=LAYERS, transform=transform,
                                      sampler=sampler)
        labels = [SVM_train_set.category.values[s] for s in sampler]

        for layer in LAYERS:
            activations_layer = activations[layer]
            if isinstance(activations_layer, list):
                activations_layer = activations_layer[0]
            pca_weights = layerwise_pca[layer].transform(
                activations_layer.reshape((len(SVM_train_set), -1)))[:, :1000]
            classifier = svm.LinearSVC(max_iter=2000)
            classifier.fit(pca_weights, labels)
            layerwise_svm[layer] = classifier
        pkl.dump(layerwise_svm, open(svm_path, 'wb'))

    return overwrite


def get_accuracy(model_dir, m=0, total_models=0, overwrite=False):

    behavioural_dir = f'in_vivo/behavioral/exp1'
    subjects = [op.basename(x) for x in sorted(glob.glob(
        f'{behavioural_dir}/data/*')) if op.isdir(x)]
    accuracy_path = f'{model_dir}/behavioral/accuracies.csv'
    if os.path.isfile(accuracy_path) and not overwrite:
        accuracies = pd.read_csv(open(accuracy_path, 'r+'), index_col=0)
    else:
        accuracies = pd.DataFrame()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize(mean=[0.445, 0.445, 0.445],
                             std=[0.269, 0.269, 0.269]),
    ])

    # put in subject batch loop to prevent memory overload
    subj_batch_size = 5
    for subject_batch in range(30//subj_batch_size):
        subject_ids = [int(x) for x in subjects[subject_batch * subj_batch_size:
                                                (subject_batch + 1) * subj_batch_size]]

        if ('transfer' in op.basename(model_dir) or 'finetune' in
                op.basename(model_dir)):
            model_name, identifier, transfer_dir = model_dir.split('/')[-3:]
        else:
            model_name, identifier, transfer_dir = model_dir.split('/')[-2:] + ['X']

        # output directory for within-model analyses
        outdir = f'{model_dir}/behavioral'
        os.makedirs(outdir, exist_ok=True)

        # skip models if already tested
        params_path = sorted(glob.glob(f"{model_dir}/params/???.pt"))[-1]
        epochs_trained = int(params_path[-6:-3])


        current_results = accuracies[(accuracies['subject'].isin(subject_ids))]

        # 10 subjects * ( 9 occluder types * 5 visibilities ) + unoccluded ) * batchnorms * transfer methods
        num_results = subj_batch_size * ((len(BEHAV.occluders_test)-1) * len(
            BEHAV.visibilities_test) + 1) * 2 * 2

        if overwrite or len(current_results) < num_results or current_results['epochs_trained'].values[0] < epochs_trained:

            overwrite = True

            # remove results that are incomplete or from a previous training epoch
            accuracies = accuracies.drop(index=current_results.index)

            # get model
            if model_name != 'cornet_s_custom':
                M = SimpleNamespace(model_dir=model_dir)
                T = SimpleNamespace()

            else:
                CFG = pkl.load(open(f'{model_dir}/config.pkl', 'rb'))
                M, T = CFG.M, CFG.T
            M.model_name = model_name
            if not hasattr(M, 'model'):
                from DNN.utils import get_model
                print('loading model...')
                M.model = get_model(M)

            # load parameters
            from DNN.utils import load_params
            print('loading parameters...')
            params_path = sorted(glob.glob(f"{model_dir}/params/*.pt"))[-1]
            M.model = load_params(params_path, M.model, 'model')
            M.params_loaded = True  # stops params being reloaded

            # set model to classify
            T.classification = True
            T.num_workers = 4
            T.nGPUs = -1
            T.batch_size = 32

            D = SimpleNamespace(dataset='ILSVRC2012')

            for transfer_method in ['output', 'SVM']:

                if transfer_method == 'SVM':
                    #print('Loading PCA...')
                    pca_path = f'{model_dir}/behavioral/PCA.pkl'
                    layerwise_pca = pkl.load(open(pca_path, 'rb'))
                    #print('Loading SVM...')
                    svm_path = f'{model_dir}/behavioral/SVM.pkl'
                    layerwise_svm = pkl.load(open(svm_path, 'rb'))
                    LAYERS = ['IT']
                else:
                    LAYERS = ['output']

                for s, subject in enumerate(subject_ids):

                    # get human behavioral data
                    subject_dir = f'{behavioural_dir}/logFiles/{subject}'
                    trials = pkl.load(open(f'{subject_dir}/trials.pkl', 'rb'))
                    trials['occlusionType'] = trials['occlusionType'].astype('category')
                    trials['visibility'] = np.round((1 - trials['occlusionLevel']) * 100).astype('int').astype('category')

                    # get images shown to humans
                    image_dir = f'{subject_dir}/stimuli'
                    image_list_model = [op.basename(x) for x in sorted(glob.glob(f'{image_dir}/*.png'))]
                    image_list_human = [op.basename(x) for x in trials.occludedObjectPath]
                    sampler = [image_list_model.index(x) for x in image_list_human]

                    for batchnorm in ['test-minibatch', 'train-running']:

                        norm_minibatch = batchnorm == 'test-minibatch'

                        # get model responses to stimuli
                        from DNN.utils import get_activations
                        activations = get_activations(M, image_dir, T=T,
                                                      layers=LAYERS,
                                                      norm_minibatch=norm_minibatch, sampler=sampler, transform=transform)


                        if transfer_method == 'output':
                            outputs = activations['output']
                            if hasattr(M, 'out_channels') and M.out_channels == 2: # select classification head from multiple heads
                                outputs = outputs[:,:,0].squeeze()
                            assert outputs.shape[0] == 752
                            """
                            # debugging: inspect responses
                            outputs = get_activations(M, image_dir, T=T, output_only=True, save_input_samples=True,
                                                         sample_input_dir=M.outdir)['output']
                            if M.out_channels == 2: # select classification head from multiple heads
                                outputs = outputs[:,:,0].squeeze()
                            plt.imshow(model_responses)
                            plt.show()
                            """
                            from DNN.utils import predict
                            model_responses_directory = predict(
                                outputs, D.dataset, afc=BEHAV.class_idxs,
                                label_type='directory')
                            trials['model_response'] = [BEHAV.class_labels_alt[
                                BEHAV.class_dirs.index(x)] for x in \
                                model_responses_directory]
                            trials['model_correct'] = pd.Series(trials['category'] == trials['model_response'],
                                                                dtype=int)
                            mean_acc = np.mean(trials.model_correct.values)

                        else:

                            best_layer, best_acc = '', 0
                            for layer in LAYERS:
                                activations_layer = activations[layer]
                                if isinstance(activations_layer, list):
                                    activations_layer = activations_layer[0]
                                assert activations_layer.shape[0] == 752
                                pca_weights = layerwise_pca[layer].transform(
                                    activations_layer.reshape((752, -1)))[:,
                                              :1000]
                                model_responses = [layerwise_svm[layer].predict(pca_weights[idx, :].reshape(1, -1))[0]
                                                   for idx in range(len(sampler))]
                                model_correct = pd.Series(trials['category'] == model_responses, dtype=int)
                                mean_acc = np.mean(model_correct)
                                if mean_acc > best_acc:
                                    best_layer, best_acc = [layer, mean_acc]
                                    trials['model_response'] = model_responses
                                    trials['model_correct'] = model_correct

                        print(f'{dtnow().strftime(nowstr)} Measuring accuracy '
                              f'batch({subject_batch+1}/6) '
                              f'model({m + 1}/{total_models} {identifier}'
                              f'/{transfer_dir}) '
                              f'subj({s+1}/5) BN({batchnorm}) '
                              f'({transfer_method}) acc({mean_acc:.3f})')


                        trials['same_correct'] = trials['correct'] * trials['model_correct']
                        trials['both_incorrect'] = (1-trials['correct']) * (1-trials['model_correct'])
                        trials['same_incorrect'] = pd.Series((trials['both_incorrect'] == 1) &
                                                             (trials['response'] == trials['model_response']), dtype=int)

                        # measure performance for each occluder * visibility level
                        for o, occluder_test in enumerate(BEHAV.occluders_test):

                            if occluder_test == 'unaltered':
                                these_visibilities = [1]
                            else:
                                these_visibilities = BEHAV.visibilities_test

                            for visibility_test in these_visibilities:

                                these_trials = trials[(trials['occlusionType'] == BEHAV.occluders_test_human[o]) &
                                                             (trials['visibility'] == visibility_test*100)]

                                # store in table and save
                                accuracies = pd.concat([accuracies,
                                                         pd.DataFrame(
                                                                 {'epochs_trained': [epochs_trained],
                                                                  'batchnorm': [batchnorm],
                                                                  'transfer_method': [transfer_method],
                                                                  'subject': [subject],
                                                                  'occluder_test': [occluder_test],
                                                                  'visibility_test': [visibility_test],
                                                                  'accuracy': [these_trials.model_correct.mean()],
                                                                  'same_correct': [these_trials.same_correct.mean()],
                                                                  'both_incorrect': [these_trials.both_incorrect.mean()],
                                                                  'same_incorrect': [these_trials.same_incorrect.mean()]})]
                                                             ).reset_index(drop=True)

    # save out accuracy table
    accuracies.to_csv(accuracy_path)

    return overwrite


def fit_sigmoids(model_dir, overwrite=False):

    # load accuracies
    accuracy_path = f'{model_dir}/behavioral/accuracies.csv'
    accuracies = pd.read_csv(open(accuracy_path, 'r+'), index_col=0)
    epochs_trained = accuracies['epochs_trained'].max()
    accuracies_subject_mean = accuracies.groupby(['epochs_trained',
                                                  'batchnorm',
                                                  'transfer_method',
                                                  'occluder_test',
                                                  'visibility_test']).agg('mean').dropna().reset_index()

    # load sigmoid parameters
    sigmoid_params_path = f'{model_dir}/behavioral/sigmoid_params.csv'
    if os.path.isfile(sigmoid_params_path):
        sigmoid_params = pd.read_csv(open(sigmoid_params_path, 'r+'), index_col=0)
    else:
        sigmoid_params = pd.DataFrame(
            {'epochs_trained': [],
             'batchnorm': [],
             'transfer_method': [],
             'occluder_test': [],
             'L': [],
             'x0': [],
             'k': [],
             'b': [],
             'threshold_.5': []},
        )

    for batchnorm, transfer_method in itertools.product(
            ['test-minibatch', 'train-running'], ['output', 'SVM']):

        current_results = sigmoid_params[
            (sigmoid_params['epochs_trained'] == epochs_trained) &
            (sigmoid_params['batchnorm'] == batchnorm) &
            (sigmoid_params['transfer_method'] == transfer_method)]

        if (overwrite or len(current_results) < len(BEHAV.occluders_test) - 1
                or current_results['epochs_trained'].values[0] <
                epochs_trained):

            overwrite = True

            # remove existing data for this model
            sigmoid_params = sigmoid_params.drop(index=current_results.index)

            # get unaltered accuracy
            unoccluded_accuracy = accuracies_subject_mean['accuracy'][
                (accuracies_subject_mean['epochs_trained'] == epochs_trained) &
                (accuracies_subject_mean['batchnorm'] == batchnorm) &
                (accuracies_subject_mean['transfer_method'] ==
                 transfer_method) &
                (accuracies_subject_mean['occluder_test'] == 'unaltered')].item()


            # accuracy for each occluder * visibility
            for o, occluder_test in enumerate(BEHAV.occluders_test[1:]):

                occluded_accuracies = accuracies_subject_mean['accuracy'][
                    (accuracies_subject_mean['epochs_trained'] == epochs_trained) &
                    (accuracies_subject_mean['batchnorm'] == batchnorm) &
                    (accuracies_subject_mean['transfer_method'] ==
                     transfer_method) &
                    (accuracies_subject_mean['occluder_test'] ==
                     occluder_test)].tolist()
                assert(len(occluded_accuracies) == len(BEHAV.visibilities_test))

                # get visibility-wise accuracies
                xvals = [0] + BEHAV.visibilities_test + [1]
                yvals = [1/8] + occluded_accuracies + [unoccluded_accuracy]

                # fit sigmoid function
                curve_x = np.linspace(0, 1, 1000)
                params = [max(yvals), np.median(xvals), 1, 0]  # initial parameters

                try:
                    popt, pcov = curve_fit(sigmoid, xvals, yvals, params, maxfev=100000)
                    curve_y = sigmoid(curve_x, *popt)
                except:
                    curve_y = np.ones_like(curve_x)*-1
                    popt = [0]*4

                threshold = sum(curve_y < .5) / 1000
                sigmoid_params = pd.concat([
                    sigmoid_params, pd.DataFrame({
                        'epochs_trained': [epochs_trained],
                        'batchnorm': [batchnorm],
                        'transfer_method': [transfer_method],
                        'occluder_test': [occluder_test],
                        'mean_accuracy': [np.mean(occluded_accuracies)],
                        'L': [popt[0]],
                        'x0': [popt[1]],
                        'k': [popt[2]],
                        'b': [popt[3]],
                        'threshold_.5': [threshold]},
                    )]).reset_index(drop=True)
    sigmoid_params.to_csv(sigmoid_params_path)

    return overwrite


def plot_accuracies(model_dir, overwrite=False):

    for batchnorm, transfer_method in itertools.product(
            ['test-minibatch', 'train-running'], ['output', 'SVM']):

        # output directory for within-model analyses
        outdir = f'{model_dir}/behavioral/{batchnorm}/{transfer_method}'
        import shutil
        if overwrite:
            shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir, exist_ok=True)

        # make plot
        outpath = f'{outdir}/accuracy_stimuli.pdf'
        if not op.isfile(outpath) or overwrite:

            accuracy_path = f'{model_dir}/behavioral/accuracies.csv'
            accuracies = pd.read_csv(open(accuracy_path, 'r+'), index_col=0)
            epochs_trained = accuracies['epochs_trained'].max()
            accuracies_subject_mean = accuracies.groupby(
                ['epochs_trained', 'batchnorm', 'transfer_method',
                 'occluder_test', 'visibility_test']).agg(
                'mean').dropna().reset_index()
            sigmoid_params_path = f'{model_dir}/behavioral/sigmoid_params.csv'
            sigmoid_params = pd.read_csv(open(sigmoid_params_path, 'r+'),
                                         index_col=0)

            fig_size_array = (8, 8)

            # get unaltered accuracy
            unoccluded_accuracy = accuracies_subject_mean['accuracy'][
                (accuracies_subject_mean['epochs_trained'] == epochs_trained) &
                (accuracies_subject_mean['batchnorm'] == batchnorm) &
                (accuracies_subject_mean['transfer_method'] ==
                 transfer_method) &
                (accuracies_subject_mean['occluder_test'] == 'unaltered')].item()

            fig, axes = plt.subplots(3, 3, figsize=fig_size_array, sharex=True,
                                     sharey=True)

            for o, occluder_test in enumerate(BEHAV.occluders_test[1:]):

                # select axis
                axR = math.floor(o / 3)
                axC = o % 3
                ax = axes[axR, axC]

                # plot accuracies
                occluded_accuracies = accuracies_subject_mean['accuracy'][
                    (accuracies_subject_mean[
                         'epochs_trained'] == epochs_trained) &
                    (accuracies_subject_mean['batchnorm'] == batchnorm) &
                    (accuracies_subject_mean['transfer_method'] ==
                     transfer_method) &
                    (accuracies_subject_mean['occluder_test'] ==
                     occluder_test)].tolist()
                assert (len(occluded_accuracies) == len(
                    BEHAV.visibilities_test))
                xvals = [0] + BEHAV.visibilities_test + [1]
                yvals = [1 / 8] + occluded_accuracies + [unoccluded_accuracy]
                ax.scatter(xvals[1:], yvals[1:])

                # plot sigmoid curve
                model_occluder_sigmoid = sigmoid_params[
                    (sigmoid_params['epochs_trained'] == epochs_trained) &
                    (sigmoid_params['batchnorm'] == batchnorm) &
                    (sigmoid_params['transfer_method'] == transfer_method) &
                    (sigmoid_params['occluder_test'] == occluder_test)
                    ].reset_index(drop=True)
                popt = [model_occluder_sigmoid[col][0].item() for col in ['L', 'x0', 'k', 'b']]
                curve_x = np.linspace(0, 1, 1000)
                curve_y = sigmoid(curve_x, *popt)
                ax.plot(curve_x, curve_y, color='tab:orange')

                # axis-specific formatting
                ax.set_title(BEHAV.occluders_test_labels[o+1], size=14)
                ax.set_xlabel('visibility', size=14)
                ax.set_ylabel('accuracy', size=14)
                ax.set_xlim((-.0, 1.05))
                ax.set_ylim((-.0, 1.05))
                ax.set_xticks((0, 1))
                ax.set_yticks((0, 1))
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.axhline(y=1 / 8, color='k', linestyle='dotted')
                if axR == 2:
                    ax.set_xlabel('visibility', size=14)
                if axC == 0:
                    ax.set_ylabel('accuracy', size=14)

            # figure-wide formatting
            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()

    return overwrite


def compare_models(overwrite=False):

    print('Comparing models (behaviour)...')

    behavioural_dir = f'in_vivo/behavioral/exp1'
    results_dir = f'in_silico/analysis/results/behaviour/behavioural_stimuli'
    figsize_scatter = (2.5,2.5)

    human_thresh = pkl.load(open(f'{behavioural_dir}/analysis/accuracy_sigmoid_params.pkl', 'rb'))
    human_acc = pkl.load(open(f'{behavioural_dir}/analysis/accuracies.pkl', 'rb'))
    human_data = {'threshold': human_thresh['individual']['thresholds_.5'],
                  'accuracy': np.mean(human_acc['occ'], axis=2),
                  'accuracy-occ-vis': human_acc['occ']}
    noise_ceiling = {}
    for measure, values in human_data.items():
        corrs_lower = []
        corrs_upper = []
        grp = np.mean(values, axis=0).flatten()  # entire group mean
        for s in range(values.shape[0]):
            ind = values[s, :].flatten()  # individual values

            # lower noise ceiling is each subject v remaining group
            not_s = [su for su in range(values.shape[0]) if su != s]  # remaining subject idxs
            rem_grp = np.mean(values[not_s, :], axis=0).flatten()  # remaining
            # group values
            corrs_lower.append(np.corrcoef(ind, rem_grp)[0][1])

            # upper noise ceiling is each subject v entire group
            corrs_upper.append(np.corrcoef(ind, grp)[0][1])

        noise_ceiling[measure] = [np.mean(corrs_lower), np.mean(corrs_upper)]

    from .config import model_contrasts
    for model_contrast, config in model_contrasts.items():

        n_models = len(config['paths'])
        figsize = (2 + (n_models / 4), 3)

        # collate data from each model
        accuracy_df = pd.DataFrame()
        sigmoid_df = pd.DataFrame()
        for m, (label, path, colour) in enumerate(zip(
                config['labels'], config['paths'], config['colours'])):

            model_dir = f'in_silico/models/{path}/behavioral'
            accuracy_path = f'{model_dir}/accuracies.csv'
            these_accs = pd.read_csv(accuracy_path, index_col=0)
            these_accs = these_accs[(these_accs.epochs_trained == 
                                      these_accs.epochs_trained.max())] 
            these_accs['model'] = label
            accuracy_df = pd.concat([accuracy_df, these_accs])
            sigmoid_path = f'{model_dir}/sigmoid_params.csv'
            these_sigs = pd.read_csv(sigmoid_path, index_col=0)
            these_sigs = these_sigs[(these_sigs.epochs_trained ==
                                        these_sigs.epochs_trained.max())]
            these_sigs['model'] = label
            sigmoid_df = pd.concat([sigmoid_df, these_sigs])


        for batchnorm, transfer_method in itertools.product(
                ['test-minibatch', 'train-running'], ['output', 'SVM']):

            this_results_dir = (f'{results_dir}/{model_contrast}/'
                                f'{batchnorm}_{transfer_method}')
            
            accs = accuracy_df[
                (accuracy_df['batchnorm'] == batchnorm) &
                (accuracy_df['transfer_method'] == transfer_method)]
            sigs = sigmoid_df[
                (sigmoid_df['batchnorm'] == batchnorm) &
                (sigmoid_df['transfer_method'] == transfer_method)]


            # accuracy/threshold bar plots
            outdir = f'{this_results_dir}/occlusion_robustness'
            os.makedirs(outdir, exist_ok=True)
            for measure_human, y_label, title, model_key in zip(
                    ['accuracy','threshold'],
                    ['proportion correct', 'proportion visible'],
                    ['classification accuracy',
                     '50% accuracy threshold\n(visibility)'],
                    ['mean_accuracy', 'threshold_.5']):
                outpath = f'{outdir}/{measure_human}.pdf'
                if not op.isfile(outpath) or overwrite:
                    
                    plt.figure(figsize=figsize)

                    # humans
                    human_values = np.mean(human_data[measure_human], axis=0)
                    sns.swarmplot(x=0, y=human_values, color=TABCOLS[:9],
                                  size=4, edgecolor='white', linewidth=1)
                    plt.bar(0, np.mean(human_values), color='tab:gray',
                            width=.5)
                    
                    # models
                    model_xpos = 0
                    for m, (label, path, colour) in enumerate(zip(
                        config['labels'], config['paths'], config['colours'])):

                        model_values = list(sigs[sigs.model == label][model_key])
                        model_xpos += 1
                        sns.swarmplot(x=model_xpos, y=model_values,
                                      color=TABCOLS[:9], size=4,
                                      edgecolor='white', linewidth=1)
                        plt.bar(model_xpos, np.mean(model_values), color=colour,
                                width=.5)

                    fig = plt.gcf()
                    ax = plt.gca()
                    plt.yticks(np.arange(0, 2, .2))
                    plt.ylim((0, 0.8))
                    if measure_human == 'accuracy':
                        ax.axhline(y=1 / 8, color='k', linestyle='dotted')
                    plt.xticks([-2, -1])
                    plt.xlim(-.5, model_xpos + .5)
                    plt.ylabel(y_label)
                    #plt.title(title)
                    plt.tight_layout()
                    fig.savefig(outpath)
                    plt.close()

            # legend
            outpath = f'{outdir}/legend.pdf'
            if not op.isfile(outpath) or overwrite:
                f = lambda m, c: plt.plot([], [], marker=m, color=c, linestyle="None")[0]
                handles = [f('s', colour) for colour in ['tab:gray'] + config['colours']]
                legend = plt.legend(handles, ['humans'] + config['labels'], loc=3)
                export_legend(legend, filename=outpath)



            # model versus human accuracy/threshold scatterplots (occluder-wise)
            outdir = f'{this_results_dir}/human_likeness'
            os.makedirs(outdir, exist_ok=True)
            for measure_human, y_label, title, model_key in zip(
                    ['accuracy', 'threshold'],
                    ['proportion correct', 'proportion visible'],
                    ['classification accuracy',
                     '50% accuracy threshold\n(visibility)'],
                    ['mean_accuracy', 'threshold_.5']):
                outpath = (f'{outdir}/'
                           f'{measure_human}-occ_scatter.pdf')
                if not op.isfile(outpath) or overwrite:
                    human_values = np.mean(human_data[measure_human], axis=0)
                    plt.figure(figsize=figsize_scatter)
                    labels = []

                    for m, (label, path, colour) in enumerate(zip(
                        config['labels'], config['paths'], config['colours'])):

                        model_values = list(sigs[sigs.model == label][model_key])
                        plt.scatter(human_values, model_values, color=colour,
                                    marker='o')
                        b, a = np.polyfit(human_values, model_values, deg=1)
                        xseq = np.linspace(.2, .65, num=1000)
                        plt.plot(xseq, a + b * xseq, color=colour, lw=2.5)

                        # make legend labels
                        try:
                            p = np.corrcoef(human_values, model_values)[0, 1]
                            labels.append(f'r = {p:.2f}')
                        except:
                            labels.append('r = nan')

                    fig = plt.gcf()
                    plt.xticks(np.arange(0, 2, .1))
                    plt.yticks(np.arange(0, 2, .5))
                    plt.xlim((.35, .65))
                    plt.ylim((0, 1))
                    plt.xlabel(f'human {y_label}')
                    plt.ylabel(f'model {y_label}')
                    plt.title(f'human v model\n{y_label}')
                    plt.tight_layout()
                    fig.savefig(outpath)
                    plt.close()

                # save legend separately
                outpath = f'{outdir}/{measure_human}-occ_legend.pdf'
                if not op.isfile(outpath) or overwrite:
                    f = lambda m, c: plt.plot([], [], marker=m, color=c)[0]
                    handles = [f('o', colour) for colour in config['colours']]
                    legend = plt.legend(handles, labels, loc=3)
                    export_legend(legend, filename=outpath)



            """
            # model versus human accuracy/threshold scatterplots sequentially add each model
            for measure_human, title, model_key in zip(['accuracy', 'threshold'],
                                                           ['classification accuracy',
                                                            'visibility threshold\n(50% accuracy)'],
                                                           ['mean_accuracy', 'threshold_.5']):
                final_outpath = (f'{outdir}/'
                                 f'{measure_human}_sequential_'
                                 f'{n_models + 1}.pdf')
                if not op.isfile(final_outpath) or overwrite:
                    human_values = np.mean(human_data[measure_human], axis=0)
                    plt.figure(figsize=figsize_scatter)
                    for m, (label, path, colour) in enumerate(zip(
                        config['labels'], config['paths'], config['colours'])):

                        model_values = list(sigs[sigs.model == label][model_key])
                        if m == 0:
                            plt.scatter(human_values,
                                        model_values,
                                        color=colour,
                                        marker='o')
                            # plot regression line
                            b, a = np.polyfit(human_values, model_values, deg=1)
                            xseq = np.linspace(.2, .65, num=1000)
                            plt.plot(xseq, a + b * xseq, color=colour, lw=2.5)
                            fig = plt.gcf()
                            ax = plt.gca()
                            plt.xticks(np.arange(0, 2, .1))
                            plt.yticks(np.arange(0, 2, .5))
                            if measure_human == 'accuracy':
                                plt.xlim((.35, .65))
                            else:
                                plt.xlim((.2, .65))
                            plt.ylim((0, 1))
                            plt.xlabel(f'human {measure_human}')
                            plt.ylabel(f'model {measure_human}')
                            plt.title(f'human v model {measure_human}')
                            plt.tight_layout()
                            fig.savefig(f'{outdir}/{measure_human}_sequential_{m + 1}.pdf')
                        else:
                            plt.scatter(human_values,
                                        model_values,
                                        color=colour,
                                        marker='o')
                            # plot regression line
                            b, a = np.polyfit(human_values, model_values, deg=1)
                            xseq = np.linspace(.2, .65, num=1000)
                            plt.plot(xseq, a + b * xseq, color=colour, lw=2.5)
                            fig.savefig(f'{outdir}/{measure_human}_sequential_{m + 1}.pdf')
                    plt.close()
            """

            # bar plots of accuracy/threshold correlation with humans
            # occluder
            for measure_human, title, model_key in zip(
                    ['accuracy', 'threshold'],
                    ['occluder-wise\naccuracy similarity',
                     'occluder-wise\nthreshold similarity'],
                    ['mean_accuracy', 'threshold_.5']):

                outpath = f'{outdir}/{measure_human}-occ.pdf'
                if not op.isfile(outpath) or overwrite:

                    sims_df = pd.DataFrame()
                    plt.figure(figsize=figsize)

                    for m, (label, path, colour) in enumerate(zip(
                        config['labels'], config['paths'], config['colours'])):

                        model_values = list(sigs[sigs.model == label][model_key])
                        human_values = human_data[measure_human]
                        sims = []
                        for s in range(human_values.shape[0]):
                            sims.append(np.corrcoef(
                                human_values[s, :], model_values)[0, 1])

                        sims_df = pd.concat([sims_df, pd.DataFrame(
                            {'model': [label]*len(sims), 'similarity': sims})])

                        plt.bar(m, np.mean(sims), color=colour, width=.5)
                        plt.errorbar(m, np.mean(sims), yerr=stats.sem(sims),
                                     color='k', capsize=2)

                    fig = plt.gcf()
                    ax = plt.gca()
                    l_nc, u_nc = noise_ceiling[measure_human]
                    ax.fill_between(np.arange(-1, 50), l_nc, u_nc,
                                    color='lightgray', lw=0)
                    plt.yticks(np.arange(0,1.1,.5))
                    plt.ylim((-0,1))
                    plt.xticks([-2, -1])
                    plt.xlim(-.5, n_models - .5)
                    plt.ylabel("Correlation ($\it{r}$)")
                    plt.title(title)
                    plt.tight_layout()
                    fig.savefig(outpath)
                    plt.close()

                    # save out sims
                    sims_df.to_csv(f'{outdir}/{measure_human}-occ.csv')

                # save legend separately
                outpath = f'{outdir}/{measure_human}_legend.pdf'
                if not op.isfile(outpath) or overwrite:
                    f = lambda m, c: plt.plot([], [], marker=m, color=c, linestyle="None")[0]
                    handles = [f('s', colour) for colour in config['colours']]
                    legend = plt.legend(handles, config['labels'], loc=3)
                    export_legend(legend, filename=outpath)


            # bar plots of accuracy/threshold correlation with humans
            # occluder * visibility
            outpath = f'{outdir}/accuracy-occ-vis.pdf'
            if not op.isfile(outpath) or overwrite:

                sims_df = pd.DataFrame()
                plt.figure(figsize=figsize)

                for m, (label, path, colour) in enumerate(zip(
                        config['labels'], config['paths'],
                        config['colours'])):

                    # get mean human and model accuacies for each occ * vis
                    model_accs = accs[accs.model == label][[
                        'occluder_test', 'visibility_test', 'accuracy'
                    ]].groupby(['occluder_test', 'visibility_test']).agg(
                        'mean').reset_index()
                    human_values = human_acc['occ']
                    model_values = np.empty_like(human_values[0])
                    for o, occ_test in enumerate(BEHAV.occluders_test[1:]):
                        for v, vis_test in enumerate(BEHAV.visibilities_test):
                            model_values[o, v] = model_accs.accuracy[
                                (model_accs.occluder_test == occ_test) &
                                (model_accs.visibility_test == vis_test)].item()

                    # normalize by subtracting mean for each vis level
                    human_values_norm = human_values - np.tile(np.mean(
                        human_values, axis=1, keepdims=True), (1, 9, 1))
                    model_values_norm = model_values - np.tile(np.mean(
                        model_values, axis=0, keepdims=True), (9, 1))

                    # measure similarity
                    sims = []
                    for s in range(human_values.shape[0]):
                        sims.append(np.corrcoef(
                            human_values_norm[s].flatten(),
                            model_values_norm.flatten())[0,1])

                    sims_df = pd.concat([sims_df, pd.DataFrame(
                        {'model': [label] * len(sims),
                         'similarity': sims})])

                    plt.bar(m, np.mean(sims), color=colour, width=.5)
                    plt.errorbar(m, np.mean(sims), yerr=stats.sem(sims),
                                 color='k', capsize=2)

                fig = plt.gcf()
                ax = plt.gca()
                l_nc, u_nc = noise_ceiling['accuracy-occ-vis']
                ax.fill_between(np.arange(-1, 50), l_nc, u_nc,
                                color='lightgray', lw=0)
                plt.yticks(np.arange(0, 1.1, .5))
                plt.ylim((0, 1))
                plt.xticks([-2, -1])
                plt.xlim(-.5, n_models - .5)
                plt.ylabel("Correlation ($\it{r}$)")
                plt.title('occluder*visibility-wise\naccuracy '
                          'similarity')
                plt.tight_layout()
                fig.savefig(outpath)
                plt.close()

                # save out sims
                sims_df.to_csv(f'{outdir}/accuracy-occ-vis.csv')

            # save legend separately
            outpath = f'{outdir}/{measure_human}_legend.pdf'
            if not op.isfile(outpath) or overwrite:
                f = lambda m, c: plt.plot([], [], marker=m, color=c,
                                          linestyle="None")[0]
                handles = [f('s', colour) for colour in config['colours']]
                legend = plt.legend(handles, config['labels'], loc=3)
                export_legend(legend, filename=outpath)



            # bar plots of image-wise similarity with humans, same correct, same incorrect response
            def same_correct(hum, mod, both):
                floor = max(0, (hum + mod) - 1)
                ceil = min(hum, mod)
                coef = (both - floor) / (ceil - floor)
                return coef

            def error_consistency(hum, mod, same):
                floor = np.abs(1 - (hum + mod))
                ceil = 1 - (floor + min(hum, mod))
                chance = (ceil - floor) / 2
                coef = (same - chance) / (1 - chance)
                return coef

            for measure, title, ylabel in zip(
                    ['same_correct', 'same_incorrect',
                     'observed_consistency', 'error_consistency'],
                    ['response similarity\naccurate trials',
                     'response similarity\ninaccurate trials',
                     'observed consistency', 'error consistency'],
                    ['proportion of trials\nhuman and model correct',
                     'proportion of trials\nsame incorrect response',
                     'observed consistency', "Cohen's Kappa"]):

                outpath =f'{outdir}/{measure}.pdf'
                if not op.isfile(outpath) or overwrite:
                    plt.figure(figsize=figsize)
                    labels = []

                    for m, (label, path, colour) in enumerate(zip(
                        config['labels'], config['paths'], config['colours'])):

                        sims = np.empty((30, 9, 5))
                        for subject, occluder_test, visibility in (
                                itertools.product(
                                accs.subject.unique(),
                                BEHAV.occluders_test[1:],
                                BEHAV.visibilities_test)):
                            s = list(accs.subject.unique()).index(subject)
                            o = BEHAV.occluders_test[1:].index(occluder_test)
                            v = BEHAV.visibilities_test.index(visibility)
                            df_subj = accs[
                                (accs.model == label) &
                                (accs.subject == subject) &
                                (accs.occluder_test == occluder_test) &
                                (accs.visibility_test == visibility)]
                            hum_acc = df_subj['correct'].mean()
                            mod_acc = df_subj['accuracy'].mean()
                            both_acc = df_subj['same_correct'].mean()
                            both_inacc = 1 - (mod_acc + (hum_acc-both_acc))
                            same_inacc = df_subj['same_incorrect'].mean()
                            both = both_acc + both_inacc

                            if measure == 'same_correct':
                                coef = same_correct(
                                    hum_acc, mod_acc, both_acc)
                                floor, chance, ceiling = 0, .5, 1

                            elif measure == 'same_incorrect':
                                coef = same_inacc / both_inacc
                                floor, chance, ceiling = 0, 1/7, 1

                            elif measure == 'observed_consistency':
                                coef = both_acc + both_inacc
                                floor, chance, ceiling = 0, -1, 1

                            else:  # if measure == 'error_consistency':
                                coef = error_consistency(hum_acc, mod_acc, both)
                                floor, chance, ceiling = -1, 0, 1

                            sims[s, o, v] = coef

                        # average across occluders and visibilities
                        sims = sims.reshape((30, -1))
                        sims = np.mean(sims, axis=1)

                        plt.bar(m, np.mean(sims),
                                color=colour,
                                width=.5)
                        plt.errorbar(m, np.mean(sims),
                                     yerr=stats.sem(sims),
                                     color='k')
                    fig = plt.gcf()
                    ax = plt.gca()
                    #plt.yticks((floor, chance, ceiling))
                    #plt.ylim((floor, ceiling))
                    plt.xticks([-2, -1])
                    plt.xlim(-.5, n_models - .5)
                    plt.ylabel(ylabel)
                    #ax.axhline(y=chance, color='k', ls='dotted')
                    plt.title(title)
                    plt.tight_layout()
                    fig.savefig(outpath)
                    plt.close()


        # save occluder legend separately
        outpath = f'{results_dir}/occluder_types_legend.pdf'
        if not op.isfile(outpath) or overwrite:
            f = lambda m, c: plt.plot(
                [], [], marker=m, markerfacecolor=c, color='white')[0]
            handles = [f('o', TABCOLS[c]) for c in range(len(
                BEHAV.occluders_test_labels) - 1)]
            labels = BEHAV.occluders_test_labels[1:]
            legend = plt.legend(handles, labels, loc=3)
            export_legend(legend, filename=outpath)


if __name__ == "__main__":

    start = time.time()

    total_models = len(model_dirs)
    for m, model_dir in enumerate(model_dirs):
        overwrite = False
        overwrite = train_SVM_classifier(model_dir, m, total_models,
                                         overwrite=overwrite)
        overwrite = get_accuracy(model_dir, m, total_models,
                                 overwrite=overwrite)
        overwrite = fit_sigmoids(model_dir, overwrite=overwrite)
        plot_accuracies(model_dir, overwrite=overwrite)
    compare_models(overwrite)

    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')


