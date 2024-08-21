
import os
import os.path as op
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pickle
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from argparse import Namespace
import math

sys.path.append('in_silico/analysis/scripts')
sys.path.append(op.expanduser('~/david/masterScripts/DNN'))
sys.path.append(op.expanduser('~/david/masterScripts/misc'))
from plot_utils import export_legend, custom_defaults
from math_functions import sigmoid
plt.rcParams.update(custom_defaults)

occluders_test_labels = ['unoccluded', 'bars (H)', 'bars (V)', 'bars (O)','crossbars (C)','crossbars (O)', 'mud splash', 'polkadot','polkasquare', 'natural']
colours = list(mcolors.TABLEAU_COLORS.keys())

# results
results_dir = op.expanduser(f'~/david/projects/p022_occlusion/in_silico/analysis/results/classification_accuracy/behavioural_and_fMRI_occluders')

accuracy_path = op.join(results_dir, 'accuracies.csv')
accuracies = pd.read_csv(open(accuracy_path, 'r+'), index_col=0)

sigmoid_params_path = op.join(results_dir, 'sigmoid_params.csv')
sigmoid_params = pd.read_csv(open(sigmoid_params_path, 'r+'), index_col=0)

human_thresh = pickle.load(open(f'in_vivo/behavioural/v3_variousTypesLevels/analysis/accuracy_sigmoid_params.pkl', 'rb'))
human_acc = pickle.load(open(f'in_vivo/behavioural/v3_variousTypesLevels/analysis/accuracies.pkl', 'rb'))
human_data = {'threshold': human_thresh['individual']['thresholds_.5'],
              'accuracy': np.mean(human_acc['occ'], axis=2)}
noise_ceiling = {}
for measure, values in human_data.items():
    corrs_lower = []
    corrs_upper = []
    grp = np.mean(values, axis=0) # entire group mean
    for s in range(values.shape[0]):

        ind = values[s,:] # individual values

        # lower noise ceiling is each subject v remaining group
        not_s = [su for su in range(values.shape[0]) if su != s]  # remaining subject idxs
        rem_grp = np.mean(values[not_s, :], axis=0)  # remaining group values
        corrs_lower.append(np.corrcoef(ind, rem_grp)[0][1])

        # upper noise ceiling is each subject v entire group
        corrs_upper.append(np.corrcoef(ind, grp)[0][1])

    noise_ceiling[measure] = [np.mean(corrs_lower), np.mean(corrs_upper)]

figsize_scatter = (5,5)

# plots across different models
from in_silico.analysis.scripts.model_contrasts_old import model_contrasts

for model_contrast, config in model_contrasts.items():

    if model_contrast not in ['model_name','norm']:
    #if model_contrast == 'CCN':

        this_results_dir = f'{results_dir}/{model_contrast}'
        os.makedirs(this_results_dir, exist_ok=True)

        if model_contrast == 'recurrent_cycles':
            print()

        # legend for all models
        f = lambda m, c, a: plt.plot([], [], marker=m, color=c, alpha=a, linestyle="None")[0]
        handles = [f('s', colour, alpha) for colour, alpha in
                   zip(['tab:gray'] + config['colours'], [1] + config['alphas'])]
        legend = plt.legend(handles, ['humans'] + config['model_labels'], loc=3)
        export_legend(legend, filename=f'{this_results_dir}/models_legend.pdf')

        num_models = len(config['model_configs'])

        # accuracy/threshold bar plots
        this_results_subdir = f'{this_results_dir}/classification_accuracy'
        os.makedirs(this_results_subdir, exist_ok=True)
        figsize = (4 + num_models / 2, 5)
        for measure_human, title, measure_model in zip(['accuracy', 'threshold'],
                                                        ['classification accuracy', 'visibility threshold\n(50% accuracy)'],
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
                model_label = config['model_labels'][m]
                colour = config['colours'][m]
                alpha = config['alphas'][m]


                # get data for this model configuration
                df = sigmoid_params.copy()
                for variable in model_config:
                    if variable in df:
                        df = df[df[variable] == model_config[variable]]


                model_values = list(df[measure_model])[:-1]
                sns.swarmplot(x=m+1,
                              y=model_values,
                              color=list(mcolors.TABLEAU_COLORS.keys())[:9],
                              size=8,
                              edgecolor='white',
                              linewidth=1)
                plt.bar(m+1,
                        np.mean(model_values),
                        color=colour,
                        alpha=alpha,
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
            alphas = []
            for m in range(len(config['model_configs'])):
                model_config = config['model_configs'][m]
                model_label = config['model_labels'][m]
                colour = config['colours'][m]
                alpha = config['alphas'][m]


                # get data for this model configuration
                df = sigmoid_params.copy()
                for variable in model_config:
                    if variable in df:
                        df = df[df[variable] == model_config[variable]]

                if len(df):
                    model_values = list(df[measure_model])[:-1]

                    p = np.corrcoef(human_values, model_values)[0,1]
                    labels.append(f'r = {p:.2f}')
                    colours.append(colour)
                    alphas.append(alpha)

                    plt.scatter(human_values,
                                model_values,
                                color=colour,
                                alpha=alpha,
                                marker='o')

                    # plot regression line
                    b, a = np.polyfit(human_values, model_values, deg=1)
                    xseq = np.linspace(.2, .65, num=1000)
                    plt.plot(xseq, a + b * xseq, alpha = alpha, color=colour, lw=2.5)

            fig = plt.gcf()
            ax = plt.gca()
            plt.xticks(np.arange(0,2,.1))
            plt.yticks(np.arange(0,2,.5))
            plt.xlim((.35, .65))
            plt.ylim((0, 1))
            plt.xlabel(f'human {measure_human}')
            plt.ylabel(f'model {measure_human}')
            plt.title(f'human v model {measure_human}')
            plt.tight_layout()
            fig.savefig(f'{this_results_subdir}/{measure_human}.pdf')
            plt.show()

            # save legend separately
            f = lambda m, c, a: plt.plot([], [], marker=m, color=c, alpha=a, linestyle=None)[0]
            handles = [f('o', colour, alpha) for colour, alpha in zip(colours, alphas)]
            legend = plt.legend(handles, labels, loc=3)
            export_legend(legend, filename=f'{this_results_subdir}/{measure_human}_legend.pdf')


        # model versus human accuracy/threshold scatterplots sequentially add each model
        if model_contrast == 'CCN':
            for measure_human, title, measure_model in zip(['accuracy', 'threshold'],
                                                           ['classification accuracy',
                                                            'visibility threshold\n(50% accuracy)'],
                                                           ['mean_accuracy', 'threshold_.5']):

                human_values = np.mean(human_data[measure_human], axis=0)
                plt.figure(figsize=figsize_scatter)
                labels = []
                colours = []
                alphas = []

                for m in range(len(config['model_configs'])):
                    model_config = config['model_configs'][m]
                    model_label = config['model_labels'][m]
                    colour = config['colours'][m]
                    alpha = config['alphas'][m]

                    # get data for this model configuration
                    df = sigmoid_params.copy()
                    for variable in model_config:
                        if variable in df:
                            df = df[df[variable] == model_config[variable]]

                    if len(df):

                        model_values = list(df[measure_model])[:-1]

                        if m == 0:

                            plt.scatter(human_values,
                                        model_values,
                                        color=colour,
                                        alpha=alpha,
                                        marker='o')

                            # plot regression line
                            b, a = np.polyfit(human_values, model_values, deg=1)
                            xseq = np.linspace(.2, .65, num=1000)
                            plt.plot(xseq, a + b * xseq, alpha=alpha, color=colour, lw=2.5)

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
                            fig.savefig(f'{this_results_subdir}/{measure_human}_sequential_{m+1}.pdf')

                        else:

                            plt.scatter(human_values,
                                        model_values,
                                        color=colour,
                                        alpha=alpha,
                                        marker='o')

                            # plot regression line
                            b, a = np.polyfit(human_values, model_values, deg=1)
                            xseq = np.linspace(.2, .65, num=1000)
                            plt.plot(xseq, a + b * xseq, alpha=alpha, color=colour, lw=2.5)
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
                model_label = config['model_labels'][m]
                colour = config['colours'][m]
                alpha = config['alphas'][m]

                # get data for this model configuration
                df = sigmoid_params.copy()
                for variable in model_config:
                    if variable in df:
                        df = df[df[variable] == model_config[variable]]
                        print(f'{m}, {variable}, {len(df)}')

                if len(df):

                    model_values = list(df[measure_model])[:-1]
                    human_values = human_data[measure_human]
                    r_scores = []
                    for s in range(human_values.shape[0]):
                        #z_scores.append(np.arctanh(np.corrcoef(human_values[s,:], model_values)[0,1]))
                        r_scores.append(np.corrcoef(human_values[s,:], model_values)[0,1])

                    plt.bar(m,
                             np.mean(r_scores),
                             yerr=scipy.stats.sem(r_scores),
                             color=colour,
                             alpha=alpha
                            )
                    plt.errorbar(m,
                                 np.mean(r_scores),
                                 yerr=scipy.stats.sem(r_scores),
                                 color='k',
                                 alpha=alpha
                                 )
                else:
                    print(f'skipping model {model_label}')

            fig = plt.gcf()
            ax = plt.gca()
            ax.fill_between(np.arange(-1, 10), noise_ceiling[measure_human][0], noise_ceiling[measure_human][1], color='lightgray', lw=0)
            plt.yticks(np.arange(-1,1.5,.5))
            plt.ylim((-.3, 1))
            plt.xticks([-2, -1])
            plt.xlim(-.5, num_models-.5)
            plt.ylabel('correlation (r)')
            # ax.axhline(y=chance_scaled, color='k', linestyle='dotted')
            plt.title(title, size=18)
            plt.tight_layout()
            fig.savefig(f'{this_results_subdir}/{measure_human}.pdf')
            plt.show()

            # save legend separately
            f = lambda m, c, a: plt.plot([], [], marker=m, color=c, alpha=a, linestyle=None)[0]
            handles = [f('s', colour, alpha) for colour, alpha in zip(colours, alphas)]
            legend = plt.legend(handles, labels, loc=3)
            export_legend(legend, filename=f'{this_results_subdir}/{measure_human}_legend.pdf')

# save occluder legend separately
cols = list(mcolors.TABLEAU_COLORS.keys())[:9]
f = lambda m, c: plt.plot([], [], marker=m, markerfacecolor=c, color='white')[0]
handles = [f('o', cols[c]) for c in range(len(occluders_test_labels)-1)]
labels = occluders_test_labels[1:]
legend = plt.legend(handles, labels, loc=3)
export_legend(legend, filename=f'{results_dir}/occluder_types_legend.pdf')

