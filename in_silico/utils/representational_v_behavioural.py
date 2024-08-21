'''
This scripts tests the accuracy of CNNs on classifying the exact images presented in the human behavioral experiment.
'''
import matplotlib
tab20b = matplotlib.cm.tab20b.colors
import os
import os.path as op
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import time

sys.path.append('code/in_silico')
sys.path.append(op.expanduser('~/david/master_scripts/DNN'))
sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from seconds_to_text import seconds_to_text
from plot_utils import custom_defaults

plt.rcParams.update(custom_defaults)

from in_silico.analysis.scripts.model_contrasts import model_contrasts

def representation_v_behavioural(overwrite=False):

    print('Plotting model comparisons...')

    behavioural_dir = f'data/in_vivo/behavioral/exp1'
    results_dir = f'data/in_silico/analysis/results/rep-v-beh'
    os.makedirs(results_dir, exist_ok=True)

    accuracy_path = (f'data/in_silico/analysis/results/behaviour'
                     f'/behavioural_stimuli/accuracies.csv')
    accuracies = pd.read_csv(open(accuracy_path, 'r+'), index_col=0)
    sigmoid_params_path =  \
        (f'data/in_silico/analysis/results/behaviour/behavioural_stimuli'
         f'/sigmoid_params.csv')
    sigmoid_params = pd.read_csv(open(sigmoid_params_path, 'r+'), index_col=0)

    figsize_scatter = (5,5)

    human_thresh = pkl.load(open(f'{behavioural_dir}/analysis/accuracy_sigmoid_params.pkl', 'rb'))
    human_acc = pkl.load(open(f'{behavioural_dir}/analysis/accuracies.pkl', 'rb'))
    human_data = {'threshold': human_thresh['individual']['thresholds_.5'],
                  'accuracy': np.mean(human_acc['occ'], axis=2)}

    rep_path = (f'data/in_silico/analysis/results/representation/fMRI_stim/RSA'
                f'.csv')
    results_rep = pd.read_csv(open(rep_path, 'r+'), index_col=0)
    results_rep['value'] = results_rep['value'].astype(float)
    results_rep['error'] = results_rep['error'].astype(float)

    for batchnorm in ['test-minibatch', 'train-running']:
        for transfer_method in ['output', 'SVM']:

            plot_values = {
                'human-likeness': {'behavioral': [], 'representational': []},
                'occlusion-robustness': {'behavioral': [], 'representational': []},
            }
            colours = []

            human_values = np.mean(human_data['accuracy'], axis=0)

            for model_name in accuracies.model_name.unique():
                accuracies_model = accuracies[accuracies['model_name'] == model_name]
                for identifier in accuracies_model.identifier.unique():
                    accuracies_id = accuracies[accuracies['identifier'] == identifier]
                    for transfer_dir in accuracies_id.transfer_dir.unique():

                        if model_name == 'cornet_s_custom' and identifier == 'base-model':
                            colour = model_contrasts['VSS']['colours'][0]
                        elif model_name == 'cornet_s_custom' and identifier == 'occ-fmri':
                            colour = model_contrasts['VSS']['colours'][1]
                        elif model_name == 'cornet_s_custom' and identifier == 'occ-beh':
                            colour = model_contrasts['VSS']['colours'][2]
                        elif model_name == 'cornet_s_custom' and identifier == 'occ-beh_task-cont_cont-one':
                            colour = model_contrasts['VSS']['colours'][3]
                        else:
                            colour = 'k'

                        # get behavioral data
                        df_beh = sigmoid_params[(sigmoid_params['model_name'] == model_name) &
                                            (sigmoid_params['identifier'] == identifier) &
                                            (sigmoid_params['transfer_dir'] == transfer_dir) &
                                            (sigmoid_params['batchnorm'] == batchnorm) &
                                            (sigmoid_params['transfer_method'] == transfer_method)
                                            ].reset_index(drop=True)
                        df_beh = df_beh[df_beh['epochs_trained'] == df_beh['epochs_trained'].max()]
                        assert len(df_beh) <= 9
                        #assert df_beh['mean_accuracy'][0].isnan() == False

                        # get representational data
                        df_rep = results_rep[
                            (results_rep['model_name'] == model_name) &
                            (results_rep['identifier'] == identifier) &
                            (results_rep['transfer_dir'] == transfer_dir) &
                            (results_rep['batchnorm'] == batchnorm) &
                            (results_rep['norm'] == 'all_conds') &
                            (results_rep['sim_measure'] == 'Kendall') &
                            (results_rep['analysis'] == 'RSA_exp2') &
                            (results_rep['subtype'] == 'attention') &
                            (results_rep['level'] == 'attn-on_ind')].reset_index(drop=True)
                        df_rep = df_rep[df_rep['epochs_trained'] == df_rep['epochs_trained'].max()]
                        assert len(df_rep) <= 4
                        #assert df_rep['value'][0].isna() == False

                        if len(df_beh) and len(df_rep):

                            # behavioral correlation
                            model_values = list(df_beh['mean_accuracy'])
                            plot_values['human-likeness']['behavioral'].append(np.corrcoef(human_values, model_values)[0, 1])

                            # behavioral robustness
                            plot_values['occlusion-robustness']['behavioral'].append(df_beh['mean_accuracy'].mean())

                            # representational correlation
                            plot_values['human-likeness']['representational'].append(df_rep['value'].mean())

                            # representational robustness
                            df_rep = results_rep[
                                (results_rep['model_name'] == model_name) &
                                (results_rep['identifier'] == identifier) &
                                (results_rep['transfer_dir'] == transfer_dir) &
                                (results_rep['batchnorm'] == batchnorm) &
                                (results_rep['norm'] == 'all_conds') &
                                (results_rep['sim_measure'] == 'Kendall') &
                                (results_rep['analysis'] == 'occlusion_robustness') &
                                (results_rep['subtype'] == 'index') &
                                (results_rep['level'] == 'grp')].reset_index(drop=True)
                            df_rep = df_rep[df_rep['epochs_trained'] == df_rep['epochs_trained'].max()]

                            plot_values['occlusion-robustness']['representational'].append(df_rep['value'].mean())

                            # colour 
                            colours.append(colour)
                
                for measure in ['behavioral']:#, 'representational']:

                #for benchmark, values in plot_values.items():
                    """
                    if benchmark == 'human-likeness':
                        size_values = plot_values['occlusion-robustness']['behavioral']*20
                    else:
                        size_values = plot_values['human-likeness']['representational']*20
                    """
                    outdir = f'{results_dir}/{measure}'
                    os.makedirs(outdir, exist_ok=True)
                    
                    outpath = f'{outdir}/{batchnorm}_{transfer_method}.pdf'
                    if not op.isfile(outpath) or overwrite:
                        plt.figure(figsize=figsize_scatter)
                        for m in range(len(plot_values['human-likeness'][measure])):
                            plt.plot(plot_values['human-likeness'][measure][m],
                                    plot_values['occlusion-robustness'][measure][m],
                                    color=tab20b[m%20],
                                    marker='o',
                                   # markersize=size_values[m],
                                     linestyle='')
        
                        # plot regression line
                        b, a = np.polyfit(plot_values['human-likeness'][measure], plot_values['occlusion-robustness'][measure], deg=1)
                        xseq = np.linspace(-2, 2, num=1000)
                        plt.plot(xseq, a + b * xseq, color=colour, lw=2.5)
                        r = np.corrcoef(plot_values['human-likeness'][measure], plot_values['occlusion-robustness'][measure])[0, 1]
        
                        fig = plt.gcf()
                        ax = plt.gca()
                        xlims = [np.min(plot_values['human-likeness'][measure])-.05, np.max(plot_values['human-likeness'][measure])+.05]
                        ylims = [np.min(plot_values['occlusion-robustness'][measure])-.05, np.max(plot_values['occlusion-robustness'][measure])+.05]
                        #plt.xticks(np.arange(-1, 1, .5))
                        #plt.yticks(np.arange(-1, 1, .02))
                        plt.xlim(xlims)
                        plt.ylim(ylims)
                        plt.xlabel(f'{measure} human-likeness')
                        plt.ylabel(f'{measure} occlusion-robustness')
                        plt.title(f'r={r:.2f}')
                        plt.tight_layout()
                        fig.savefig(outpath)
                        plt.show()


if __name__ == "__main__":

    start = time.time()

    overwrite = True
    representation_v_behavioural(overwrite=overwrite)
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')
