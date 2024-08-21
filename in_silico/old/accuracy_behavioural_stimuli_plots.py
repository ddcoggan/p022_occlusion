
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
results_dir = op.expanduser(f'~/david/projects/p022_occlusion/in_silico/analysis/results/classification_accuracy/behavioural_stimuli')

accuracy_path = op.join(results_dir, 'accuracies.csv')
accuracies = pd.read_csv(open(accuracy_path, 'r+'), index_col=0)

sigmoid_params_path = op.join(results_dir, 'sigmoid_params.csv')
sigmoid_params = pd.read_csv(open(sigmoid_params_path, 'r+'), index_col=0)

human_thresh = pickle.load(open(f'in_vivo/behavioural/v3_variousTypesLevels/analysis/sigmoid_params.pkl', 'rb'))
human_acc = pickle.load(open(f'in_vivo/behavioural/v3_variousTypesLevels/analysis/accuracies.pkl', 'rb'))
human_data = {'threshold': human_thresh['individual']['thresholds_.5'],
              'accuracy': np.mean(human_acc['occ'], axis=2)/100}

figsize_scatter = (5,5)

# plots across different models
from in_silico.analysis.scripts.model_contrasts_old import model_contrasts

for model_contrast, config in model_contrasts.items():


    if model_contrast not in ['model_name','norm']:

        this_results_dir = f'{results_dir}/{model_contrast}'
        os.makedirs(this_results_dir, exist_ok=True)

        figsize = (4 + len(model_idxs)/2, 5)

        num_models = len(config['model_configs'])

        # accuracy/threshold bar plots
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

            model_xpos = 0
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

                if len(df) and m in model_idxs:
                    model_xpos += 1
                    model_values = list(df[measure_model])
                    sns.swarmplot(x=model_xpos,
                                  y=model_values,
                                  color=list(mcolors.TABLEAU_COLORS.keys())[:9],
                                  size=8,
                                  edgecolor='white',
                                  linewidth=1)
                    plt.bar(model_xpos,
                            np.mean(model_values),
                            color=colour,
                            alpha=alpha,
                            width=.5
                            )

            fig = plt.gcf()
            ax = plt.gca()
            plt.yticks(np.arange(0, 2, .2))
            plt.ylim((0, 1.02))
            if measure_human == 'accuracy':
                ax.axhline(y=1 / 8, color='k', linestyle='dotted')
            plt.xticks([-2, -1])
            plt.xlim(-.5, model_xpos + .5)
            plt.ylabel(measure_human)
            plt.title(title)
            plt.tight_layout()
            fig.savefig(f'{this_results_dir}/{measure_human}.pdf')
            plt.show()

        # legend
        f = lambda m, c, a: plt.plot([], [], marker=m, color=c, alpha=a)[0]
        handles = [f('o', colour, alpha) for colour, alpha in zip(['tab:gray'] + config['colours'], [1] + config['alphas'])]
        legend = plt.legend(handles, ['humans'] + config['model_labels'], loc=3)
        export_legend(legend, filename=f'{this_results_dir}/accuracies_and_thresholds_legend.pdf')


        # model versus human accuracy/threshold scatterplots
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

                if len(df) and m in model_idxs:
                    model_values = list(df[measure_model])

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
            fig.savefig(f'{this_results_dir}/{measure_human}_versus_human.pdf')
            plt.show()

            # save legend separately
            f = lambda m, c, a: plt.plot([], [], marker=m, color=c, alpha=a, linestyle=None)[0]
            handles = [f('o', colour, alpha) for colour, alpha in zip(colours, alphas)]
            legend = plt.legend(handles, labels, loc=3)
            export_legend(legend, filename=f'{this_results_dir}/{measure_human}_versus_human_legend.pdf')


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

                    if len(df) and m in model_idxs:
                        model_values = list(df[measure_model])

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
                            fig.savefig(f'{this_results_dir}/{measure_human}_versus_human_sequential_{m+1}.pdf')

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
                            fig.savefig(f'{this_results_dir}/{measure_human}_versus_human_sequential_{m + 1}.pdf')
                plt.show()

        # bar plots of accuracy/threshold correlation with humans
        for measure_human, title, measure_model in zip(['accuracy', 'threshold'],
                                                       ['model-human occluderwise\naccuracy similarity',
                                                        'model-human occluderwise\nthreshold similarity'],
                                                       ['mean_accuracy', 'threshold_.5']):
            plt.figure(figsize=figsize)
            labels = []
            model_xpos = 0
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

                if len(df) and m in model_idxs:

                    model_values = list(df[measure_model])
                    human_values = human_data[measure_human]
                    z_scores = []
                    for s in range(human_values.shape[0]):
                        z_scores.append(np.arctanh(np.corrcoef(human_values[s,:], model_values)[0,1]))

                    plt.bar(model_xpos,
                             np.mean(z_scores),
                             yerr=scipy.stats.sem(z_scores),
                             color=colour,
                             alpha=alpha,
                             width=.5
                            )
                    plt.errorbar(model_xpos,
                                 np.mean(z_scores),
                                 yerr=scipy.stats.sem(z_scores),
                                 color='k',
                                 alpha=alpha
                                 )
                    model_xpos += 1

            fig = plt.gcf()
            ax = plt.gca()
            plt.yticks((-1, 0, 1))
            plt.ylim((-1, 1))
            plt.xticks([-2, -1])
            plt.xlim(-.5, model_xpos - .5)
            plt.ylabel('correlation (Z)')
            # ax.axhline(y=chance_scaled, color='k', linestyle='dotted')
            plt.title(title, size=18)
            plt.tight_layout()
            fig.savefig(f'{this_results_dir}/{measure_human}_versus_human_bar.pdf')
            plt.show()

            # save legend separately
            f = lambda m, c, a: plt.plot([], [], marker=m, color=c, alpha=a, linestyle=None)[0]
            handles = [f('s', colour, alpha) for colour, alpha in zip(colours, alphas)]
            legend = plt.legend(handles, labels, loc=3)
            export_legend(legend, filename=f'{this_results_dir}/{measure_human}_legend.pdf')


        # bar plots of image-wise similarity with humans, same correct, same incorrect response
        for measure, title, ylabel in zip(['same_correct', 'same_incorrect'],['response similarity\naccurate trials', 'response similarity\ninaccurate trials'], ['proportion of trials\nhuman and model correct', 'proportion of trials\nsame incorrect response']):
            plt.figure(figsize=figsize)
            labels=[]
            model_xpos = 0
            
            for m in range(len(config['model_configs'])):
                model_config = config['model_configs'][m]
                model_label = config['model_labels'][m]
                colour = config['colours'][m]
                alpha = config['alphas'][m]

                # get data for this model configuration
                df = accuracies.copy()
                for variable in model_config:
                    if variable in df and variable != 'identifier':
                        df = df[df[variable] == model_config[variable]]

                if len(df) and m in model_idxs:

                    sims = []
                    for s, subject in enumerate(df['subject'].unique()):
                        df_subj = df[df['subject'] == subject]
                        response_similarity = df_subj[measure].mean()
                        human_accuracy = np.mean(human_data['accuracy'],axis=1)[s]

                        # calculate chance, floor, ceiling performance
                        model_accuracy = df_subj['accuracy'].mean()
                        both_correct = df_subj['same_correct'].mean()
                        both_incorrect = 1 - model_accuracy - (human_accuracy - both_correct)
                        if measure == 'same_correct':
                            floor = max(0, (human_accuracy + model_accuracy) - 1)
                            ceiling = min(human_accuracy, model_accuracy)
                            chance = floor + ( (1-floor) * (human_accuracy-floor) * (model_accuracy-floor) )
                        elif measure == 'same_incorrect':
                            chance = both_incorrect / 8
                            floor = 0
                            ceiling = both_incorrect

                        # scale the performance and chance between 0 (floor) and 1 (ceiling)
                        response_similarity_scaled = (response_similarity - floor) / (ceiling - floor)
                        #chance_scaled = (chance - floor) / (ceiling - floor)
                        #labels += [f'chance = {chance_scaled:.2f}']
                        sims.append(response_similarity_scaled)

                    plt.bar(model_xpos,
                            np.mean(sims),
                            color=colour,
                            alpha=alpha,
                            width=.5
                            )
                    plt.errorbar(model_xpos,
                                 np.mean(sims),
                                 yerr=scipy.stats.sem(sims),
                                 color='k',
                                 alpha=alpha
                                 )
                    #plt.axhline(y=chance_scaled, xmin=model_xpos-.5, xmax=model_xpos+.5, color='k', linestyle='dotted')
                    model_xpos += 1

            fig = plt.gcf()
            ax = plt.gca()
            plt.yticks((0,1), labels=['floor','ceiling'])
            plt.ylim((0, 1))
            plt.xticks([-2,-1])
            plt.xlim(-.5,model_xpos -.5)
            plt.ylabel(ylabel)
            #ax.axhline(y=chance_scaled, color='k', linestyle='dotted')
            plt.title(title, size=18)
            plt.tight_layout()
            fig.savefig(f'{this_results_dir}/{measure}_versus_human.pdf')
            plt.show()

            # save legend separately
            f = lambda m, c, a: plt.plot([], [], marker=m, color=c, alpha=a, linestyle=None)[0]
            handles = [f('s', colour, alpha) for colour, alpha in zip(colours, alphas)]
            legend = plt.legend(handles, labels, loc=3)
            export_legend(legend, filename=f'{this_results_dir}/{measure}_legend.pdf')


# save occluder legend separately
cols = list(mcolors.TABLEAU_COLORS.keys())[:9]
f = lambda m, c: plt.plot([], [], marker=m, markerfacecolor=c, color='white')[0]
handles = [f('o', cols[c]) for c in range(len(occluders_test_labels)-1)]
labels = occluders_test_labels[1:]
legend = plt.legend(handles, labels, loc=3)
export_legend(legend, filename=f'{results_dir}/visibility_thresholds_legend.pdf')

