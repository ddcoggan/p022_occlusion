'''
This scripts tests the accuracy of CNNs on the Imagenet validation set for the 8 exemplars used in fMRI and behavioral
experiments, with the same occluder types and visibilities applied. It is an alternative to accuracy_behav_stim.py,
which uses the exact images presented in the behavioral experiment. The approach in the present script can be used to
measure robustness to the fMRI occluder type. It can also be adapted to assess performance over many more classes and
occluder types than those used in the human experiments. However, it doesn't allow for trial-wise comparison with human
behavioral responses.
'''

import os.path as op
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from argparse import Namespace
import time

sys.path.append(op.expanduser('~/david/master_scripts/DNN'))
sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from plot_utils import export_legend, custom_defaults
from seconds_to_text import seconds_to_text
plt.rcParams.update(custom_defaults)
from math_functions import sigmoid


from in_vivo.fMRI.utils import *

def accuracy_behav_fMRI_occs(model_dirs, overwrite=False):

    results_dir = f'/in_silico/analysis/results/classification_accuracy/behavioural_and_fMRI_occluders'
    os.makedirs(results_dir, exist_ok=True)
    accuracy_path = op.join(results_dir, 'accuracies.csv')
    sigmoid_params_path = op.join(results_dir, 'sigmoid_params.csv')

    if os.path.isfile(accuracy_path):
        accuracies = pd.read_csv(open(accuracy_path, 'r+'), index_col=0)
    else:
        accuracies = pd.DataFrame(
            {'model_name': [],
             'identifier': [],
             'occluder_test': [],
             'visibility_test': [],
             'acc1': [],
             'acc5': []},
        )

    if os.path.isfile(sigmoid_params_path):
        sigmoid_params = pd.read_csv(open(sigmoid_params_path, 'r+'), index_col=0)
    else:
        sigmoid_params = pd.DataFrame(
            {'model_name': [],
             'identifier': [],
             'epochs_trained': [],
             'occluder_test': [],
             'L': [],
             'x0': [],
             'k': [],
             'b': [],
             'threshold_.25': []},
        )

    # start of model loop
    for m, model_dir in enumerate(model_dirs):

        # output directory for within-model analyses
        outdir_model = f'{model_dir}/behavioral'
        os.makedirs(outdir_model, exist_ok=True)

        model_name, identifier = model_dir.split('/')[-2:]

        # skip models if already tested
        params_path = sorted(glob.glob(f"{model_dir}/params/*.pt"))[-1]
        epochs_trained = int(params_path[-6:-3])
        current_results = accuracies[
            (accuracies['model_name'] == model_name) &
            (accuracies['identifier'] == identifier)]

        num_results = ((len(occluders_test) - 1) * len(visibilities_test) + 1)

        if overwrite or len(current_results) < num_results or current_results['epochs_trained'].values[-1] < epochs_trained:

            print(f'model {m + 1}/{len(model_dirs)} {params_path}')

            # remove results that are incomplete or from a previous training epoch
            accuracies = accuracies.drop(index=current_results.index)
            CFG = pkl.load(open(f'{model_dir}/config.pkl', 'rb'))

            # get model
            if not hasattr(CFG.M, 'model'):
                from utils import get_model
                print('loading model...')
                CFG.M.model = get_model(CFG.M)

            # load parameters
            from utils import load_params
            print('loading parameters...')
            CFG.M.model = load_params(params_path, model=CFG.M.model)
            CFG.M.params_loaded = True  # stops params being reloaded

            for occluder_test in occluders_test + fMRI.occluders:

                if occluder_test == 'unaltered':
                    these_visibilities = [1]
                else:
                    these_visibilities = visibilities_test

                for visibility_test in these_visibilities:

                    if len(accuracies[
                               (accuracies['model_name'] == model_name) &
                               (accuracies['identifier'] == identifier) &
                               (accuracies['occluder_test'] == occluder_test) &
                               (accuracies['visibility_test'] == visibility_test)]) < 1:

                        Occlusion = Namespace(
                            type = occluder_test,  # occluder type or list thereof
                            prop_occluded = 1,  # proportion of images to be occluded
                            visibility = visibility_test,  # image visibility portion or list thereof
                            colour=[(0, 0, 0), (255, 255, 255)],
                        )
                        CFG.D.Occlusion = Occlusion
                        CFG.D.greyscale = True
                        CFG.D.save_input_samples=True
                        #CFG.D.class_subset = class_idxs

                        # set model to classify
                        CFG.T.learning = 'supervised_classification'
                        #CFG.T.batch_size = 256

                        # measure accuracy
                        print(f'\n{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Testing {occluder_test} at {visibility_test} on model at {model_dir}')
                        from utils import test_model
                        metrics = test_model(CFG)
                        acc1, acc5 = metrics['acc1'], metrics['acc5']

                        # store in table and save
                        accuracies = pd.concat([
                            accuracies,
                            pd.DataFrame({
                                'model_name': [model_name],
                                'identifier': [identifier],
                                'epochs_trained': [epochs_trained],
                                'occluder_test': [occluder_test],
                                'visibility_test': [visibility_test],
                                'acc1': [acc1],
                                'acc5': [acc5]}
                            )]).reset_index(drop=True)
            accuracies.to_csv(accuracy_path)

            # sigmoid function
            current_results = sigmoid_params[  # have not already been analysed
                (sigmoid_params['model_name'] == model_name) &
                (sigmoid_params['identifier'] == identifier)]

            # remove results that are incomplete or from a previous training epoch
            sigmoid_params = sigmoid_params.drop(index=current_results.index)

            # get unaltered accuracy
            acc1unalt = accuracies['acc1'][
                               (accuracies['model_name'] == model_name) &
                               (accuracies['identifier'] == identifier) &
                               (accuracies['occluder_test'] == 'unaltered')].item()
    
            # accuracy for each occluder * visibility (behavioral occluders only)
            for o, occluder_test in enumerate(occluders_test[1:]):

                # occluded accuracies
                model_occluder_subset = accuracies[(accuracies['model_name'] == model_name) &
                                                   (accuracies['identifier'] == identifier) &
                                                   (accuracies['occluder_test'] == occluder_test)]
                xvals = [0] + visibilities_test + [1]
                yvals = [1/1000] + list(model_occluder_subset['acc1']) + [acc1unalt]


                # fit sigmoid function
                params = [max(yvals), np.median(xvals), 1, 0]  # initial parameters
                popt, pcov = curve_fit(sigmoid, xvals, yvals, params, maxfev=100000)
                curve_x = np.linspace(0, 1, 1000)
                curve_y = sigmoid(curve_x, *popt)
                threshold = sum(curve_y < .5) / 1000
                sigmoid_params = pd.concat([
                    sigmoid_params,
                    pd.DataFrame({
                        'model_name': [model_name],
                        'identifier': [identifier],
                        'occluder_test': [occluder_test],
                        'epochs_trained': [epochs_trained],
                        'mean_accuracy': [np.mean(model_occluder_subset['acc1'].values)],
                        'L': [popt[0]],
                        'x0': [popt[1]],
                        'k': [popt[2]],
                        'b': [popt[3]],
                        'threshold_.5': [threshold]},
                    )]).reset_index(drop=True)
            sigmoid_params.to_csv(sigmoid_params_path)


def accuracy_behav_fMRI_occs_plots_modelwise():

    results_dir = f'in_silico/analysis/results/classification_accuracy/behavioural_and_fMRI_occluders'
    accuracy_path = op.join(results_dir, 'accuracies.csv')
    accuracies = pd.read_csv(open(accuracy_path, 'r+'), index_col=0)
    sigmoid_params_path = op.join(results_dir, 'sigmoid_params.csv')
    sigmoid_params = pd.read_csv(open(sigmoid_params_path, 'r+'), index_col=0)

    fig_size_array = (8, 8)
    colours = list(mcolors.TABLEAU_COLORS.keys())

    # start of model loop
    for m, model_dir in enumerate(model_dirs):
        
        # output directory for within-model analyses
        outdir_model = f'{model_dir}/behavioral'
        os.makedirs(outdir_model, exist_ok=True)

        model_name, identifier = model_dir.split('/')[-2:]

        # get unaltered accuracy
        acc1unalt = accuracies['acc1'][
            (accuracies['model_name'] == model_name) &
            (accuracies['identifier'] == identifier) &
            (accuracies['occluder_test'] == 'unaltered')].item()

        # accuracy for each occluder * visibility (behavioral occluders only)
        fig, axes = plt.subplots(3, 3, figsize=fig_size_array, sharex=True, sharey=True)
        
        for o, occluder_test in enumerate(occluders_test[1:]):

            # occluded accuracies
            model_occluder_accuracies = accuracies[(accuracies['model_name'] == model_name) &
                                                    (accuracies['identifier'] == identifier) &
                                                    (accuracies['occluder_test'] == occluder_test)]
            xvals = [0] + visibilities_test + [1]
            yvals = [1 / 8] + list(model_occluder_accuracies['accuracy']) + [acc1unalt]
            ax.scatter(xvals[1:], yvals[1:], color=colours[0])

            # plot sigmoid curve
            model_occluder_sigmoid = sigmoid_params[(sigmoid_params['model_name'] == model_name) &
                                                    (sigmoid_params['identifier'] == identifier) &
                                                    (sigmoid_params['occluder_test'] == occluder_test)]
            popt = [model_occluder_sigmoid[col][0].item() for col in ['L', 'x0', 'k', 'b']]
            curve_x = np.linspace(0, 1, 1000)
            curve_y = sigmoid(curve_x, *popt)
            ax.plot(curve_x, curve_y, color=colours[1])
            
            # plot for behavioral occluders
            if occluder_test in occluders_test[1:]:

                # select axis
                axR = math.floor(o / 3)
                axC = o % 3
                ax = axes[axR, axC]
                ax.scatter(xvals[1:], yvals[1:], color=colours[0])
                ax.plot(curve_x, curve_y, color=colours[1])

                # format axis
                ax.set_title(occluders_test_labels[o + 1], size=14)
                ax.set_xlim((0, 1.05))
                ax.set_ylim((0, 1.05))
                ax.set_xticks((0, 1))
                ax.set_yticks((0, 1))
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.axhline(y=1 / 8, color='k', linestyle='dotted')

                if axR == 2:
                    ax.set_xlabel('visibility', size=14)
                if axC == 0:
                    ax.set_ylabel('accuracy', size=14)


        # format and save figure
        plt.tight_layout()
        plt.savefig(f'{outdir_model}/accuracy_occluder.pdf')
        plt.show()
        
        
def accuracy_behav_fMRI_occs_plots_model_contrasts():

    results_dir = f'in_silico/analysis/results/classification_accuracy/behavioural_and_fMRI_occluders'
    accuracy_path = op.join(results_dir, 'accuracies.csv')
    accuracies = pd.read_csv(open(accuracy_path, 'r+'), index_col=0)
    sigmoid_params_path = op.join(results_dir, 'sigmoid_params.csv')
    sigmoid_params = pd.read_csv(open(sigmoid_params_path, 'r+'), index_col=0)
    
    figsize_scatter = (5, 5)
        
    human_thresh = pkl.load(
        open(f'in_vivo/behavioral/exp1/analysis/accuracy_sigmoid_params.pkl', 'rb'))
    human_acc = pkl.load(open(f'in_vivo/behavioral/exp1/analysis/accuracies.pkl', 'rb'))
    human_data = {'threshold': human_thresh['individual']['thresholds_.5'],
                  'accuracy': np.mean(human_acc['occ'], axis=2)}
    noise_ceiling = {}
    for measure, values in human_data.items():
        corrs_lower = []
        corrs_upper = []
        grp = np.mean(values, axis=0)  # entire group mean
        for s in range(values.shape[0]):
            ind = values[s, :]  # individual values

            # lower noise ceiling is each subject v remaining group
            not_s = [su for su in range(values.shape[0]) if su != s]  # remaining subject idxs
            rem_grp = np.mean(values[not_s, :], axis=0)  # remaining group values
            corrs_lower.append(np.corrcoef(ind, rem_grp)[0][1])

            # upper noise ceiling is each subject v entire group
            corrs_upper.append(np.corrcoef(ind, grp)[0][1])

        noise_ceiling[measure] = [np.mean(corrs_lower), np.mean(corrs_upper)]


    from in_silico.analysis.scripts.model_contrasts import model_contrasts

    for model_contrast, config in model_contrasts.items():

        if model_contrast not in ['model_name', 'norm']:

            this_results_dir = f'{results_dir}/{model_contrast}'
            os.makedirs(this_results_dir, exist_ok=True)

            # legend for all models
            f = lambda m, c: plt.plot([], [], marker=m, color=c, linestyle="None")[0]
            handles = [f('s', colour) for colour in ['tab:gray'] + config['colours']]
            legend = plt.legend(handles, ['humans'] + config['model_labels'], loc=3)
            export_legend(legend, filename=f'{this_results_dir}/models_legend.pdf')

            num_models = len(config['model_configs'])

            # accuracy/threshold bar plots
            this_results_subdir = f'{this_results_dir}/classification_accuracy'
            os.makedirs(this_results_subdir, exist_ok=True)
            figsize = (4 + num_models / 2, 5)
            for measure_human, title, measure_model in zip(['accuracy', 'threshold'],
                                                           ['classification accuracy',
                                                            'visibility threshold\n(50% accuracy)'],
                                                           ['mean_accuracy', 'threshold_.5']):

                human_values = np.mean(human_data[measure_human], axis=0)
                plt.figure(figsize=figsize)

                # add human data
                sns.swarmplot(x=0,
                              y=human_values,
                              color=list(mcolors.TABLEAU_COLORS.keys())[:9],
                              size=8,
                              edgecolor='white',
                              linewidth=1)
                plt.bar(0,
                        np.mean(human_values),
                        color='tab:gray',
                        width=.5
                        )

                for m in range(len(config['model_configs'])):
                    model_config = config['model_configs'][m]
                    model_name = model_config['model_name']
                    identifier = model_config['identifier']
                    colour = config['colours'][m]

                    # get data for this model configuration
                    df = sigmoid_params[(sigmoid_params['model_name'] == model_name) &
                                        (sigmoid_params['identifier'] == identifier)]


                    model_values = list(df[measure_model])[:-1]
                    sns.swarmplot(x=m + 1,
                                  y=model_values,
                                  color=list(mcolors.TABLEAU_COLORS.keys())[:9],
                                  size=8,
                                  edgecolor='white',
                                  linewidth=1)
                    plt.bar(m + 1,
                            np.mean(model_values),
                            color=colour,
                            width=.5
                            )

                fig = plt.gcf()
                ax = plt.gca()
                plt.yticks(np.arange(0, 2, .2))
                plt.ylim((0, .8))
                plt.ylabel(measure_human)
                if measure_human == 'accuracy':
                    ax.axhline(y=1 / 8, color='k', linestyle='dotted')
                    plt.ylabel('proportion correct')
                plt.xticks([-2, -1])
                plt.xlim(-.5, num_models + .5)
                plt.title(title)
                plt.tight_layout()
                fig.savefig(f'{this_results_subdir}/{measure_human}.pdf')
                plt.show()


            # model versus human accuracy/threshold scatterplots
            this_results_subdir = f'{this_results_dir}/human_v_model_accuracy_scatterplots'
            os.makedirs(this_results_subdir, exist_ok=True)
            for measure_human, title, measure_model in zip(['accuracy', 'threshold'],
                                                           ['classification accuracy',
                                                            'visibility threshold\n(50% accuracy)'],
                                                           ['mean_accuracy', 'threshold_.5']):

                human_values = np.mean(human_data[measure_human], axis=0)
                plt.figure(figsize=figsize_scatter)
                labels = []
                colours = []
                for m in range(len(config['model_configs'])):
                    model_config = config['model_configs'][m]
                    model_name = model_config['model_name']
                    identifier = model_config['identifier']
                    colour = config['colours'][m]

                    # get data for this model configuration
                    df = sigmoid_params[(sigmoid_params['model_name'] == model_name) &
                                        (sigmoid_params['identifier'] == identifier)]

                    if len(df):
                        model_values = list(df[measure_model])[:-1]

                        p = np.corrcoef(human_values, model_values)[0, 1]
                        labels.append(f'r = {p:.2f}')
                        colours.append(colour)

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
                plt.xlim((.35, .65))
                plt.ylim((0, 1))
                plt.xlabel(f'human {measure_human}')
                plt.ylabel(f'model {measure_human}')
                plt.title(f'human v model {measure_human}')
                plt.tight_layout()
                fig.savefig(f'{this_results_subdir}/{measure_human}.pdf')
                plt.show()

                # save legend separately
                f = lambda m, c: plt.plot([], [], marker=m, color=c, linestyle=None)[0]
                handles = [f('o', colour) for colour in colours]
                legend = plt.legend(handles, labels, loc=3)
                export_legend(legend, filename=f'{this_results_subdir}/{measure_human}_legend.pdf')


            # model versus human accuracy/threshold scatterplots sequentially add each model
            for measure_human, title, measure_model in zip(['accuracy', 'threshold'],
                                                           ['classification accuracy',
                                                            'visibility threshold\n(50% accuracy)'],
                                                           ['mean_accuracy', 'threshold_.5']):

                human_values = np.mean(human_data[measure_human], axis=0)
                plt.figure(figsize=figsize_scatter)
                labels = []
                colours = []

                for m in range(len(config['model_configs'])):
                    model_config = config['model_configs'][m]
                    model_name = model_config['model_name']
                    identifier = model_config['identifier']
                    colour = config['colours'][m]

                    # get data for this model configuration
                    df = sigmoid_params[(sigmoid_params['model_name'] == model_name) &
                                        (sigmoid_params['identifier'] == identifier)]

                    if len(df):
                        model_values = list(df[measure_model])[:-1]

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
                            fig.savefig(f'{this_results_subdir}/{measure_human}_sequential_{m + 1}.pdf')

                        else:

                            plt.scatter(human_values,
                                        model_values,
                                        color=colour,
                                        marker='o')

                            # plot regression line
                            b, a = np.polyfit(human_values, model_values, deg=1)
                            xseq = np.linspace(.2, .65, num=1000)
                            plt.plot(xseq, a + b * xseq, color=colour, lw=2.5)
                            fig.savefig(f'{this_results_subdir}/{measure_human}_sequential_{m + 1}.pdf')
                plt.show()


            # bar plots of accuracy/threshold correlation with humans
            this_results_subdir = f'{this_results_dir}/human_v_model_accuracy_barplots'
            os.makedirs(this_results_subdir, exist_ok=True)
            figsize = (1.5 + num_models / 2, 5)
            for measure_human, title, measure_model in zip(['accuracy', 'threshold'],
                                                           ['model-human occluderwise\naccuracy similarity',
                                                            'model-human occluderwise\nthreshold similarity'],
                                                           ['mean_accuracy', 'threshold_.5']):
                plt.figure(figsize=figsize)
                labels = []
                for m in range(len(config['model_configs'])):
                    model_config = config['model_configs'][m]
                    model_name = model_config['model_name']
                    identifier = model_config['identifier']
                    colour = config['colours'][m]

                    # get data for this model configuration
                    df = sigmoid_params[(sigmoid_params['model_name'] == model_name) &
                                        (sigmoid_params['identifier'] == identifier)]

                    if len(df):
                        model_values = list(df[measure_model])[:-1]
                        human_values = human_data[measure_human]
                        r_scores = []
                        for s in range(human_values.shape[0]):
                            # z_scores.append(np.arctanh(np.corrcoef(human_values[s,:], model_values)[0,1]))
                            r_scores.append(np.corrcoef(human_values[s, :], model_values)[0, 1])

                        plt.bar(m,
                                np.mean(r_scores),
                                yerr=stats.sem(r_scores),
                                color=colour,
                                )
                        plt.errorbar(m,
                                     np.mean(r_scores),
                                     yerr=stats.sem(r_scores),
                                     color='k',
                                     )

                fig = plt.gcf()
                ax = plt.gca()
                ax.fill_between(np.arange(-1, 10), noise_ceiling[measure_human][0], noise_ceiling[measure_human][1],
                                color='lightgray', lw=0)
                plt.yticks(np.arange(-1, 1.5, .5))
                plt.ylim((-.3, 1))
                plt.xticks([-2, -1])
                plt.xlim(-.5, num_models - .5)
                plt.ylabel('correlation (r)')
                # ax.axhline(y=chance_scaled, color='k', linestyle='dotted')
                plt.title(title, size=18)
                plt.tight_layout()
                fig.savefig(f'{this_results_subdir}/{measure_human}.pdf')
                plt.show()

                # save legend separately
                f = lambda m, c: plt.plot([], [], marker=m, color=c, linestyle=None)[0]
                handles = [f('o', colour) for colour in colours]
                legend = plt.legend(handles, labels, loc=3)
                export_legend(legend, filename=f'{this_results_subdir}/{measure_human}_legend.pdf')


    # save occluder legend separately
    cols = list(mcolors.TABLEAU_COLORS.keys())[:9]
    f = lambda m, c: plt.plot([], [], marker=m, markerfacecolor=c, color='white')[0]
    handles = [f('o', cols[c]) for c in range(len(occluders_test_labels) - 1)]
    labels = occluders_test_labels[1:]
    legend = plt.legend(handles, labels, loc=3)
    export_legend(legend, filename=f'{results_dir}/occluder_types_legend.pdf')


if __name__ == "__main__":

    model_search = "cornet_s?*"
    model_dirs = sorted(glob.glob(f'in_silico/models/{model_search}/*'))
    model_dirs = [model_dir for model_dir in model_dirs if ('cont' not in model_dir) ^ ('transfer' in model_dir)]
    start = time.time()
    accuracy_behav_fMRI_occs(model_dirs, overwrite=True)
    accuracy_behav_fMRI_occs_plots_modelwise()
    accuracy_behav_fMRI_occs_plots_model_contrasts()
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')