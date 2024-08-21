import os
import os.path as op
import sys
import glob
import datetime
import matplotlib.pyplot as plt
import time
from scipy.stats import kendalltau
import pandas as pd
import numpy as np
import pickle as pkl
from scipy import stats
from frrsa import frrsa

sys.path.append(f'{op.expanduser("~")}/david/master_scripts/misc')
from plot_utils import export_legend, custom_defaults
plt.rcParams.update(custom_defaults)
from seconds_to_text import seconds_to_text

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # converts to order in nvidia-smi (not in cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # which device(s) to use

from in_vivo.fMRI.utils import layers_of_interest, fMRI, figsizes, sim_measures, regression_models


def fMRI_stim_RSA(model_dirs, overwrite=False):

    analysis_dir = f'in_silico/analysis/results/representation/fMRI_stim'
    os.makedirs(analysis_dir, exist_ok=True)
    results_path = f'{analysis_dir}/RSA.csv'
    total_params_paths = len(model_dirs)#sum([len(glob.glob(f'{model_dir}/params/*.pt')) for model_dir in model_dirs])
    n_conds = fMRI.n_img


    if op.isfile(results_path):
        results = pd.read_csv(open(results_path, 'r+'), index_col=0)
    else:
        results = pd.DataFrame()

    counter = 1
    for m, model_dir in enumerate(model_dirs):

        if 'transfer' in op.basename(model_dir):
            model_name, identifier, transfer_dir = model_dir.split('/')[-3:]
        else:
            model_name, identifier, transfer_dir = model_dir.split('/')[-2:] + ['X']
        outdir_model = f'{model_dir}/fMRI'
        epochs_trained = int(sorted(glob.glob(f"{model_dir}/params/*.pt"))[-1][-6:-3])
        RDMs_path = f'{outdir_model}/RDMs/RDMs.pkl'

        if not op.isfile(RDMs_path):
            make_RDMs = True
        else:
            RDMs = pkl.load(open(RDMs_path, 'rb'))
            make_RDMs = epochs_trained not in RDMs
        if make_RDMs:
            from in_silico.analysis.scripts.RSA_fMRI import fMRI_stim_RDMs
            fMRI_stim_RDMs([model_dir], overwrite=True)
            RDMs = pkl.load(open(RDMs_path, 'rb'))

        if len(results):
            current_results = results[(results['model_name'] == model_name) &
                                  (results['identifier'] == identifier) &
                                  (results['transfer_dir'] == transfer_dir)]
        else:
            current_results = pd.DataFrame()

        num_results = 57 # 5 hardcoded

        if len(current_results) < num_results or make_RDMs or current_results['epochs_trained'].max() < epochs_trained or overwrite:

            results = results.drop(index=current_results.index)

            for layer in layers_of_interest:

                print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {counter}/{total_params_paths} | {model_name} | {identifier} | {transfer_dir} | epoch {epochs_trained} | {layer} ')

                results_layer = pd.DataFrame()

                RDM = RDMs[epochs_trained][layer]['RDM'].get_matrices()
                activations_layer = RDMs[epochs_trained][layer]['dataset'].measurements

                # convert RDM to table with conditions
                from in_vivo.fMRI.utils import RDM_to_table
                RDM_table = RDM_to_table(RDM)

                # perform regression analyses
                from in_vivo.fMRI.utils import RDM_regression
                regression_table = RDM_regression(RDM_table)
                regression_table['benchmark'] = 'occlusion_robustness'
                regression_table['analysis'] = 'regression'
                regression_table['subtype'] = 'regression'
                results_layer = pd.concat([results_layer, regression_table]).reset_index(drop=True)


                # get condition-wise means and errors across subjects
                condition_summary = RDM_table.drop(
                    columns=['exemplar_a', 'exemplar_b', 'occluder_a', 'occluder_b']).groupby(
                    ['analysis', 'level']).agg(['mean','sem']).reset_index()
                condition_summary.columns = ['analysis', 'level', 'value', 'error']
                condition_summary['benchmark'] = 'occlusion_robustness'
                condition_summary['unit'] = 'Euclidean distance'
                condition_summary['subtype'] = 'condwise'
                condition_summary['error_unit'] = 'sem'
                results_layer = pd.concat([results_layer, condition_summary]).reset_index(drop=True)

                # calculate occlusion robustness
                from in_vivo.fMRI.utils import calculate_occlusion_robustness
                for sub_analysis, groupbys in zip(['grp', 'exem'], [['level'],['analysis', 'level', 'exemplar_b']]):
                    df = RDM_table.groupby(groupbys).agg('mean').reset_index()
                    robustness_table = calculate_occlusion_robustness(df)
                    robustness_table['benchmark'] = 'occlusion_robustness'
                    robustness_table['subtype'] = f'index_{sub_analysis}'
                    results_layer = pd.concat([results_layer, robustness_table]).reset_index(drop=True)


                # RSA with human fMRI data
                RDM_flat_offdiag = RDM_table.similarity.values[fMRI.off_diag_mask_flat]

                for fMRI_exp, attns in zip(['exp1', 'exp2', 'exp1+2'],
                                           [[''], ['attn-on_', 'attn-off_', 'attn-on-off_'], ['']]):

                    # combine RDMs for exp1 and exp2
                    if fMRI_exp == 'exp1+2':

                        RDMs_fMRI_path = f'in_vivo/fMRI/exp1/derivatives/RSA/RDMs.pkl'
                        exp1_RDMs = pkl.load(open(RDMs_fMRI_path, 'rb'))
                        RDMs_fMRI_path = f'in_vivo/fMRI/exp2/derivatives/RSA/RDMs/RDMs.pkl'
                        exp2_RDMs = pkl.load(open(RDMs_fMRI_path, 'rb'))
                        RDMs_fMRI_matched = {}
                        """
                        for subject in exp1_RDMs[layer]:
                            RDMs_fMRI_matched[subject] = exp1_RDMs[layer][subject]
                        for subject in exp2_RDMs[layer]:
                            exp2_attn_on = exp2_RDMs[layer][subject][:, :n_conds, :n_conds]
                            if subject in RDMs_fMRI_matched:
                                RDMs_fMRI_matched[subject] = np.concatenate(
                                    [RDMs_fMRI_matched[subject], exp2_attn_on], axis=0)
                            else:
                                RDMs_fMRI_matched[subject] = exp2_attn_on
                        """
                        for subject in exp2_RDMs[layer]:
                            exp2_attn_on = exp2_RDMs[layer][subject][:, :n_conds, :n_conds]
                            RDMs_fMRI_matched[subject] = exp2_attn_on
                        for subject in exp1_RDMs[layer]:
                            if subject not in RDMs_fMRI_matched:
                                RDMs_fMRI_matched[subject] = exp1_RDMs[layer][subject]
                    else:
                        RDMs_fMRI_path = f'in_vivo/fMRI/{fMRI_exp}/derivatives/RSA/RDMs/RDMs.pkl'
                        RDMs_fMRI_all = pkl.load(open(RDMs_fMRI_path, 'rb'))
                        RDMs_fMRI_matched = RDMs_fMRI_all[layer]  # matches ROI to layer

                    n_subjects = len(RDMs_fMRI_matched.keys())

                    for attn in attns:

                        # get fMRI RDMs
                        RDMs_fMRI = np.empty((n_subjects, n_conds, n_conds))

                        for s, subject in enumerate(RDMs_fMRI_matched):

                            if 'exp1' in fMRI_exp:
                                RDMs_fMRI[s, :, :] = np.mean(RDMs_fMRI_matched[subject], axis=0)

                            else:
                                if attn == 'attn-on-off_':
                                    RDMs_attn = np.empty((2, n_conds, n_conds))
                                    for a in range(len(fMRI.attns)):
                                        idxs = np.arange(a * n_conds, (a + 1) * n_conds)
                                        RDMs_attn[a, :, :] = np.mean(
                                            RDMs_fMRI_matched[subject][:, :, idxs][:, idxs, :],
                                            axis=0)
                                    RDMs_fMRI[s, :, :] = np.mean(RDMs_attn, axis=0)
                                else:
                                    if attn == 'attn-off_':
                                        idxs = np.arange(n_conds, (n_conds * 2))
                                    else:
                                        idxs = np.arange(n_conds)
                                    RDMs_fMRI[s, :, :] = np.mean(RDMs_fMRI_matched[subject][:, :, idxs][:, idxs, :],
                                                                 axis=0)  # mean across data splits

                            for RSA_type in ['cRSA', 'frRSA']:

                                # 'image' excludes diagonal, 'exemplar' excludes same exemplar comparisons
                                for cond_set in ['image', 'exemplar']:

                                    RDM_mask = 1-regression_models[cond_set].flatten()

                                    # individual subjects
                                    rs = []
                                    for s in range(RDMs_fMRI.shape[0]):
                                        target_RDM = RDMs_fMRI[s, :, :]
                                        if RSA_type == 'cRSA':
                                            predictor_RDM = RDM
                                        else:
                                            scores, predictor_RDM = frrsa(
                                                target_RDM,
                                                activations_layer,
                                                preprocess=True,
                                                nonnegative=False,
                                                measures=['sqeuclidean', 'mahalanobis'],
                                                cv=[5, 10],
                                                hyperparams=None,
                                                score_type='pearson',
                                                wanted=['predicted_matrix'],
                                                parallel='1',
                                                random_state=None,
                                            )
                                        target_RDM = target_RDM.flatten()[RDM_mask]
                                        predictor_RDM = predictor_RDM.flatten()[RDM_mask]
                                        rs.append(kendalltau(target_RDM, predictor_RDM).correlation)

                                    results_layer = pd.concat([
                                        results_layer,
                                        pd.DataFrame({
                                            'benchmark': ['human_likeness'],
                                            'analysis': [RSA_type],
                                            'subtype': [fMRI_exp],
                                            'level': [f'{attn}ind_{cond_set}'],
                                            'unit': ['kendall'],
                                            'value': [np.mean(rs)],
                                            'error_unit': ['sem'],
                                            'error': [stats.sem(rs)]}),
                                    ]).reset_index(drop=True)

                                # group mean
                                target_RDM = np.mean(RDMs_fMRI, axis=0)  # group mean
                                if RSA_type == 'cRSA':
                                    predictor_RDM = RDM
                                else:
                                    scores, predictor_RDM = frrsa(
                                        target_RDM,
                                        activations_layer,
                                        preprocess=True,
                                        nonnegative=False,
                                        measures=['sqeuclidean', 'mahalanobis'],
                                        cv=[5, 10],
                                        hyperparams=None,
                                        score_type='pearson',
                                        wanted=['predicted_matrix'],
                                        parallel='1',
                                        random_state=None,
                                    )
                                target_RDM = target_RDM.flatten()[RDM_mask]
                                predictor_RDM = predictor_RDM.flatten()[RDM_mask]
                                rs = kendalltau(target_RDM, predictor_RDM).correlation

                                results_layer = pd.concat([
                                    results_layer,
                                    pd.DataFrame({
                                        'benchmark': ['human_likeness'],
                                        'analysis': [RSA_type],
                                        'subtype': [fMRI_exp],
                                        'level': [f'{attn}grp_{cond_set}'],
                                        'unit': ['kendall'],
                                        'value': [rs],
                                        'error_unit': [None],
                                        'error': [None]}),
                                ]).reset_index(drop=True)


                results_layer['model_name'] = model_name
                results_layer['identifier'] = identifier
                results_layer['transfer_dir'] = transfer_dir
                results_layer['epochs_trained'] = epochs_trained
                results_layer['layer'] = layer

                results = pd.concat([results, results_layer]).reset_index(drop=True)

        results.transfer_dir = results.transfer_dir.astype(str)
        counter += 1
        results.to_csv(results_path)

                            
def fMRI_stim_RSA_plots_modelwise(model_dirs): # NEEDS FIXING

    print('Plotting modelwise analyses...')
    analysis_dir = f'in_silico/analysis/results/representation/fMRI_stim'
    results_path = f'{analysis_dir}/RSA.csv'
    results = pd.read_csv(open(results_path, 'r+'), index_col=0)

    
    for m, model_dir in enumerate(model_dirs):

        if 'transfer' in op.basename(model_dir):
            model_name, identifier, transfer_dir = model_dir.split('/')[-3:]
        else:
            model_name, identifier, transfer_dir = model_dir.split('/')[-2:] + ['X']
        outdir_model = f'{model_dir}/fMRI'

        for n, norm in enumerate(fMRI.norms):

            # make plots across all layers
            for measure, measure_string in sim_measures.items():

                for analysis, vars in fMRI.occlusion_robustness_analyses.items():


                    df = results.loc[
                        (results['model_name'] == model_name) &
                        (results['identifier'] == identifier) &
                        (results['transfer_dir'] == transfer_dir) &
                        #(results['epochs_trained'] == 32) &
                        (results['norm'] == norm) &
                        (results['analysis'] == analysis) &
                        (results['level'].isin(vars['conds'])) &
                        (results['sim_measure'] == measure) &
                        (results['layer'].isin(layers_of_interest)), :].copy()
                    df['level'] = df['level'].astype('category').cat.reorder_categories(vars['conds'])
                    df['layer'] = df['layer'].astype('category').cat.reorder_categories(layers_of_interest)
                    df_means = df.pivot(index='layer', columns='level', values='mean')
                    df_sems = df.pivot(index='layer', columns='level', values='error').values
                    df_plot = df_means.plot(kind='bar', yerr=df_sems.transpose(), ylabel=measure_string, rot=0, figsize=figsize_4_layer, color=vars['colours'])
                    fig = df_plot.get_figure()
                    plt.xlabel('model layer')
                    plt.ylim((-.6, 1.1))
                    plt.title(analysis)
                    plt.legend(title='', labels=vars['cond_labels'], bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                    plt.tight_layout()
                    fig.savefig(f'{outdir_model}/{analysis}_norm-{norm}_{measure}.pdf')
                    plt.show()


                # RSA with human fMRI responses

                for fMRI_dataset in fMRI.datasets:

                    # individual subjects
                    df = results.loc[
                        (results['model_name'] == model_name) &
                        (results['identifier'] == identifier) &
                        (results['transfer_dir'] == transfer_dir) &
                        #(results['epochs_trained'] == epochs_trained) &
                        (results['norm'] == norm) &
                        (results['analysis'] == fMRI_dataset) &
                        (results['level'] == 'ind') &
                        (results['sim_measure'] == measure) &
                        (results['layer'].isin(layers_of_interest)), :].copy()
                    figure = plt.plot(figsize=(5, 6))
                    plt.errorbar(x=range(len(layers_of_interest)),
                                 y=df['mean'].values,
                                 yerr=df['error'].values,
                                 color='tab:blue')
                    plt.plot(range(len(layers_of_interest)),
                             df['mean'].values,
                             marker='o',
                             markerfacecolor='white')
                    fig = plt.gcf()
                    plt.xticks(np.arange(len(layers_of_interest)), labels=layers_of_interest)
                    plt.yticks((-.5, 0, .5, 1))
                    plt.xlabel('model/human layer')
                    plt.ylabel(measure_string)
                    plt.ylim((-.4, 1.))
                    plt.title('RSA with human fMRI')
                    plt.tight_layout()
                    fig.savefig(f'{outdir_model}/{fMRI_dataset}_norm-{norm}_ind_{measure}.pdf')
                    plt.show()

                    # group mean
                    df = results.loc[
                        (results['model_name'] == model_name) &
                        (results['identifier'] == identifier) &
                        (results['transfer_dir'] == transfer_dir) &
                        #(results['epochs_trained'] == epochs_trained) &
                        (results['norm'] == norm) &
                        (results['analysis'] == f'fMRI_RSA_{fMRI_dataset}') &
                        (results['level'] == 'grp') &
                        (results['sim_measure'] == measure) &
                        (results['layer'].isin(layers_of_interest)), :].copy()
                    figure = plt.plot(figsize=figsizes['single_4'])
                    plt.plot(range(len(layers_of_interest)),
                             df['mean'],
                             marker='o',
                             markerfacecolor='white')
                    fig = plt.gcf()
                    plt.xticks(np.arange(len(layers_of_interest)), labels=layers_of_interest)
                    plt.yticks((-.5, 0, .5, 1))
                    plt.xlabel('model/human layer')
                    plt.ylabel(measure_string)
                    plt.ylim((-.4, 1.))
                    plt.title('RSA with human fMRI')
                    plt.tight_layout()
                    fig.savefig(f'{outdir_model}/{fMRI_dataset}_norm-{norm}_grp_{measure}.pdf')
                    plt.show()

                """
                # make plots across training epochs
    
                model_results = results[
                    (results['model_name'] == model_name) &
                    (results['identifier'] == identifier) &
                    (results['norm'] == norm)]
    
                epochwise_stats = {'OCI_raw': {},
                                   'OCI_norm': {},
                                   'OII_raw': {},
                                   'OII_norm': {},
                                   'fMRI_RSA_exp1': {},
                                   'fMRI_RSA_exp2_attn-on': {},
                                   'fMRI_RSA_exp2_attn-off': {}}
    
                for layer in layers_of_interest:
    
                    for key in epochwise_stats:
                        epochwise_stats[key][layer] = {'means': []}
    
                    epoch_list = sorted(model_results['epochs_trained'].unique())
                    n_epochs = len(epoch_list)
    
                    for epochs_trained in epoch_list:
    
                        # get data for this model configuration
                        df = model_results[(model_results['epochs_trained'] == epochs_trained) &
                                           (model_results['layer'] == layer)]
    
                        # completion / invariance
                        analyses = fMRI.RDM_analyses
                        for analysis in list(analyses.keys())[:2]:
                            index_label = analyses[analysis]["index_label"]
                            for index_type in ['raw', 'norm']:
                                epochwise_stats[f'{index_label}_{index_type}'][layer]['means'].append(
                                    df['mean'][(df['analysis'] == analysis) &
                                               (df['level'] == f'index_{index_type}')])
    
                        # RSA with human fMRI data
                        for fMRI_dataset in fMRI.datasets:
                            for level in ['ind','grp']:
                                rs = df['mean'][(df['analysis'] == fMRI_dataset) & (df['level'] == level)].item()
                                epochwise_stats[f'{fMRI_dataset}_{level}'][layer]['means'].append(rs)
    
                # make plots
                for plot in epochwise_stats:
    
                    plt.figure(figsize=figsize_epochwise)
    
                    for l, layer in enumerate(layers_of_interest):
                        plt.plot(epoch_list,
                                 epochwise_stats[plot][layer]['means'],
                                 color=tabcols[l],
                                 #alpha=alpha,
                                 marker='o',
                                 markerfacecolor='white',
                                 label=layer)
    
                    fig = plt.gcf()
                    ax = plt.gca()
                    ax.fill_between(epoch_list, 1, 2, color='black', alpha=.2, lw=0)
                    plt.xticks(epoch_list)
                    plt.yticks((-.5, 0, .5, 1))
                    plt.xlabel('epochs trained')
                    plt.ylabel(plot)
                    plt.ylim((-.4, 1.))
                    plt.title(plot)
                    plt.tight_layout()
                    fig.savefig(f'{outdir_model}/{plot}_norm-{norm}_epochwise.pdf')
                    plt.show()
    
            # save legends separately
            f = lambda m, c, a: \
                plt.plot([], [], marker=m, color=c, alpha=a, markerfacecolor='white', ls="solid")[0]
            handles = [f('o', tabcols[c], 1) for c in range(len(layers_of_interest))]
            labels = layers_of_interest
            legend = plt.legend(handles, labels, loc=3)
            export_legend(legend, filename=f'{outdir_model}/legend.pdf')
            """






if __name__ == "__main__":

    os.chdir(op.expanduser("~/david/projects/p022_occlusion"))
    start = time.time()

    from model_contrasts import model_contrasts

    model_dirs = []
    for plot_config, configs in model_contrasts.items():
        if plot_config == 'VSS':
            for config in configs['model_configs']:
                model_dir = f'in_silico/models/{config["model_name"]}/{config["identifier"]}'
                if 'transfer_dir' in config:
                    model_dir += f'/{config["transfer_dir"]}'
                model_dirs.append(model_dir)
    model_dirs = set(model_dirs)

    model_dirs = sorted(glob.glob(f'in_silico/models/cornet_s_*/**/params', recursive=True))
    model_dirs = [op.dirname(x) for x in model_dirs]

    fMRI_stim_RSA(model_dirs, overwrite = False)
    #fMRI_stim_RSA_plots_modelwise(model_dirs)
    fMRI_stim_RSA_plots_model_contrasts()
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')
