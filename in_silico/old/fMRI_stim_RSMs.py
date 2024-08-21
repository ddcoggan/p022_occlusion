import os
import os.path as op
import sys
import glob
import time
from scipy.stats import kendalltau
import pickle as pkl
import numpy as np
from types import SimpleNamespace
import shutil
import matplotlib.pyplot as plt


sys.path.append(op.expanduser('~/david/master_scripts/DNN'))
from utils import get_activations
sys.path.append(op.expanduser('~/david/master_scripts/misc'))
from seconds_to_text import seconds_to_text


from in_vivo.fMRI.utils import fMRI, RSM_plot, layers_of_interest, MDS_plot, sim_measures


def fMRI_stim_RSMs(model_dirs, overwrite=False):

    n_conds = fMRI.n_img

    for m, model_dir in enumerate(model_dirs):

        # image directory
        image_dir = f'in_vivo/fMRI/exp1_orig/stimuli/all_stimuli'
        images = sorted(glob.glob(f'{image_dir}/*'))
        sampler = []
        for cond in fMRI.cond_labels:
            for i, image in enumerate(images):
                if cond in image:
                    sampler.append(i)

        # output directory
        outdir = f'{model_dir}/fMRI'
        if op.isdir(outdir) and overwrite:
            shutil.rmtree(outdir)
        os.makedirs(f'{outdir}/RSMs', exist_ok=True)

        RSMs_path = f"{outdir}/RSMs/RSMs.pkl"
        if op.isfile(RSMs_path or overwrite):
            RSMs = pkl.load(open(RSMs_path, 'rb'))
        else:
            RSMs = {}

        params_paths = sorted(glob.glob(f"{model_dir}/params/*.pt"))[-1:]

        for params_path in params_paths:

            print(params_path)
            epochs_trained = int(params_path[-6:-3])

            if epochs_trained not in RSMs or overwrite:

                RSMs[epochs_trained] = {}

                # get model configuration from config file or params file
                config_path = f'{model_dir}/config.pkl'
                CFG = pkl.load(open(config_path, 'rb'))
                if isinstance(CFG, list):
                    M,D,T = CFG
                else:
                    M = CFG.M
                if 'transfer' in op.basename(model_dir):
                    model_name, identifier, transfer_dir = model_dir.split('/')[-3:]
                else:
                    model_name, identifier, transfer_dir = model_dir.split('/')[-2:] + ['X']
                M.model_name = model_name  # ensure model name is in config
                M.params_path = params_path

                # set hardware parameters
                T = SimpleNamespace(
                    nGPUs = 1,
                    GPUids = 0,
                    batch_size = n_conds,
                    num_workers = 2
                )
                
                for batchnorm in ['test-minibatch', 'train-running']:

                    RSMs[epochs_trained][batchnorm] = {}

                    print('measuring activations...')
                    norm_minibatch = batchnorm == 'test-minibatch'
                    activations = get_activations(M, image_dir, T,
                                                  layers_of_interest=layers_of_interest,
                                                  sampler=sampler, norm_minibatch=norm_minibatch)
    
                    # generate RSMs
                    for norm in fMRI.norms:

                        RSMs[epochs_trained][batchnorm][norm] = {}
        
                        for l, layer in enumerate(activations):
        
                            print(f'analysing {layer} layer with {batchnorm} batch-norm and {norm} norm...')
        
                            RSMs[epochs_trained][batchnorm][norm][layer] = {}
                            activations_layer = activations[layer]
                            activations_layer = np.array(activations_layer.flatten(start_dim=1))
        
                            activations_layer_norm = np.empty_like(activations_layer)
                            if norm == 'all_conds':
                                activations_layer_norm = activations_layer - np.tile(np.mean(activations_layer, axis=0), (n_conds, 1))
                            elif norm == 'occluder':
                                for oc in range(3):
                                    mean_act = np.tile(np.mean(activations_layer[oc::3, :], axis=0), (8, 1))
                                    activations_layer_norm[oc::3, :] = activations_layer[oc::3, :] - mean_act
                            elif norm == 'unoccluded':
                                activations_layer_norm = activations_layer - np.tile(np.mean(activations_layer[::3, :], axis=0), (24, 1))
                            elif norm == 'none':
                                activations_layer_norm = activations_layer
        
                            for measure in sim_measures:
                                if measure == 'Pearson':
                                    RSM = np.corrcoef(activations_layer_norm)
                                else:
                                    RSM = np.empty((n_conds, n_conds))
                                    for cA in range(n_conds):
                                        for cB in range(n_conds):
                                            RSM[cA, cB] = kendalltau(activations_layer_norm[cA, :],
                                                                     activations_layer_norm[cB, :]).correlation
                                assert np.sum(np.isnan(RSM.flatten())) == 0
                                RSMs[epochs_trained][batchnorm][norm][layer][measure] = RSM
        
                            # TSNE plots (must be based on responses, not RSMs)
                            from in_vivo.fMRI.utils import TSNE_plot
                            # just take mean of last split as each split iteration contains all responses
                            outpath = f'{outdir}/TSNE_plots/{l}_{layer}_batchnorm-{batchnorm}.png'
                            os.makedirs(op.dirname(outpath), exist_ok=True)
                            if norm == fMRI.norms[0]:
                                TSNE_plot(activations_layer, outpath)
        
                            # activation magnitudes
                            unoccluded_mags = activations_layer[0::3, :].flatten()
                            occluded_mags = np.array([activations_layer[1::3, :].flatten(), activations_layer[2::3, :].flatten()]).flatten()
                            unocc_nonzeros = (np.count_nonzero(unoccluded_mags) / unoccluded_mags.shape[0]) * 100
                            occ_nonzeros = (np.count_nonzero(occluded_mags) / occluded_mags.shape[0]) * 100
                            fig, ax = plt.subplots(figsize=(7,6))
                            upper = np.max([np.max(unoccluded_mags), np.max(occluded_mags)])
                            lower = np.min([np.min(unoccluded_mags), np.min(occluded_mags)])
                            bins = np.linspace(lower, upper, 32)
                            ax.hist(unoccluded_mags, bins, alpha=0.5, label=f'unoccluded ({int(unocc_nonzeros)}% > 0)')
                            ax.hist(occluded_mags, bins, alpha=0.5, label=f'occluded ({int(occ_nonzeros)}% > 0)')
                            ax.legend(loc='upper right')
                            ax.set_xlabel('response magnitude')
                            ax.set_ylabel('unit count')
                            ax.set_title(layer)
                            outpath = f'{outdir}/activation_magnitudes/{layer}_batchnorm-{batchnorm}.png'
                            os.makedirs(op.dirname(outpath), exist_ok=True)
                            plt.tight_layout()
                            plt.savefig(outpath)
                            plt.show()
        
                # save RSMs and contrasts for this model
                pkl.dump(RSMs, open(f"{outdir}/RSMs/RSMs.pkl", "wb"))


def fMRI_stim_RSMs_plot(model_dirs, overwrite=False):

    # start of model loop
    for m, model_dir in enumerate(model_dirs):

        print(f'Plotting RSMs, MDS | {model_dir}')

        RSMs_dir = f'{model_dir}/fMRI/RSMs'
        MDS_dir = f'{model_dir}/fMRI/MDS'
        os.makedirs(MDS_dir, exist_ok=True)
        RSMs_path = f'{RSMs_dir}/RSMs.pkl'
        if op.isfile(RSMs_path):
            RSMs = pkl.load(open(RSMs_path, 'rb'))
            last_epoch = sorted(list(RSMs.keys()))[-1]
            for batchnorm in ['test-minibatch', 'train-running']:
                for norm in fMRI.norms:
                    for l, layer in enumerate(RSMs[last_epoch][batchnorm][norm]):
                        for measure, measure_string in sim_measures.items():

                            # RSM
                            RSM = RSMs[last_epoch][batchnorm][norm][layer][measure]
                            outpath = f'{RSMs_dir}/{batchnorm}/{norm}/{measure}/{l}_{layer}.pdf'
                            os.makedirs(op.dirname(outpath), exist_ok=True)
                            fancy = model_dir.endswith('cornet_s_custom/base-model')
                            if not op.isfile(outpath) or overwrite:
                                RSM_plot(RSM, outpath=outpath, measure=measure_string, fancy=fancy)

                            # MDS
                            outpath = f'{MDS_dir}/{batchnorm}/{norm}/{measure}/{l}_{layer}.pdf'
                            os.makedirs(op.dirname(outpath), exist_ok=True)
                            if not op.isfile(outpath) or overwrite:
                                MDS_plot(RSM, outpath=outpath)


if __name__ == "__main__":

    start = time.time()

    from model_contrasts import model_contrasts

    model_dirs = []
    for config in model_contrasts['VSS']['model_configs']:
        model_dir = f'in_silico/models/{config["model_name"]}/{config["identifier"]}'
        if 'transfer_dir' in config:
            model_dir += f'/{config["transfer_dir"]}'
        model_dirs.append(model_dir)

    model_dirs = sorted(glob.glob(f'in_silico/models/cornet_s_*/**/params', recursive=True))
    model_dirs = [op.dirname(x) for x in model_dirs]
    
    fMRI_stim_RSMs(model_dirs, overwrite=True)
    #fMRI_stim_RSMs_plot(model_dirs, overwrite=True)

    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')
