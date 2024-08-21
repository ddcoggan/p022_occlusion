"""
configure a model for training
"""
import os
import os.path as op
import glob
from types import SimpleNamespace
from argparse import Namespace
import torch
import yaml
import pickle as pkl
import sys
import datetime

sys.path.append(op.expanduser(f'~/david/masterScripts/misc'))
from namespace_utils import Loader, Dumper, object_hook, nsEncoder

base_dir = op.abspath('in_silico/data')
model_dirs = glob.glob(f'{base_dir}/cornet_flab/*')
sys.path.insert(0, '/home/tonglab/david/projects/p022_occlusion/in_silico/data/cornet_s_custom/head-deep_occ-beh_task-cont_cont-trans')

lut = {
    'modelName': 'model_name',
    'times': 'R',
    'outDir': 'model_dir',
    'lastEpoch': 'last_epoch',
    'RF': 'K',
    'propOccluded': 'prop_occluded',
    'occlusion': 'Occlusion',
    'blur': 'Blur',
    'noise': 'Noise',
    'datasetPath': 'dataset_path',
    'datasetName': 'dataset',
    'optimizerName': 'optimizer_name',
    'batchSize': 'batch_size',
    'learningRate': 'learning_rate',
    'nEpochs': 'num_epochs',
    'saveInterval': 'save_interval'
}
occlusion_params = ['occluder', 'visibility', 'coverage', 'invert', 'prop_occluded', 'propOccluded', 'colour','colours','type','label','visLabel']
remove_params = ['outdir', 'instance', 'model', 'dataset_path', 'visLabel', 'label']
for model_dir in model_dirs:

    print(model_dir)

    # get model configuration from config file or params file
    config_path = f'{model_dir}/config_old.pkl'
    params_paths = sorted(glob.glob(f"{model_dir}/params/*.pt"))
    if op.isfile(config_path):
        CFG = pkl.load(open(config_path, 'rb'))
    elif params_paths:
        import torch
        params_path = params_paths[-1]
        CFG = torch.load(params_path)['config']
    else:
        Exception('No model configuration found.')

    if isinstance(CFG, dict):
        M = SimpleNamespace()
        for key, value in CFG['modelParams'].items():
            if key in lut:
                new_key = lut[key]
            else:
                new_key = key
            if isinstance(value, tuple):
                new_value = list(value)
            else:
                new_value = value
            setattr(M, new_key, new_value)
        D = SimpleNamespace()
        occ_ns = SimpleNamespace()
        occ_exists = False
        for key, value in CFG['datasetParams'].items():
            if key in lut:
                new_key = lut[key]
            else:
                new_key = key
            if key == 'coverage':
                new_key = 'visibility'
                new_value = 1 - value
            else:
                new_value = value
            if new_value == 'imagenet1000':
                new_value = 'ILSVRC2012'
            if isinstance(value, dict):
                new_space = SimpleNamespace()
                for subkey, subvalue in value.items():
                    if subkey == 'coverage':
                        new_subkey = 'visibility'
                        new_subvalue = 1 - subvalue
                    else:
                        new_subvalue = subvalue
                    if subkey in lut:
                        new_subkey = lut[subkey]
                    else:
                        new_subkey = subkey
                    setattr(new_space, new_subkey, new_subvalue)
                setattr(D, new_key, new_space)
            else:
                if key in occlusion_params:
                    setattr(occ_ns, new_key, new_value)
                    occ_exists = True
                else:
                    setattr(D, new_key, new_value)
        if occ_exists:
            setattr(D, 'Occlusion', occ_ns)

        T = SimpleNamespace()
        for key, value in CFG['trainingParams'].items():
            if key in lut:
                new_key = lut[key]
            else:
                new_key = key
            setattr(T, new_key, value)

    else:
        if isinstance(CFG, list):
            M_orig, D_orig, T_orig = CFG
        else:
            try:
                M_orig, D_orig, T_orig = CFG.M, CFG.D, CFG.T
            except:
                M_orig, D_orig, T_orig = CFG.m, CFG.d, CFG.t

        M = SimpleNamespace()
        for key, value in M_orig.__dict__.items():
            if key in lut:
                new_key = lut[key]
            else:
                new_key = key
            setattr(M, new_key, value)

        D = SimpleNamespace()
        for key, value in D_orig.__dict__.items():
            if key in lut:
                new_key = lut[key]
            else:
                new_key = key
            if key == 'coverage':
                new_key = 'visibility'
                new_value = 1 - value
            else:
                new_value = value
            if new_value == 'imagenet1000':
                new_value = 'ILSVRC2012'
            if isinstance(value, Namespace):
                new_space = SimpleNamespace()
                for subkey, subvalue in value.__dict__.items():
                    if subkey == 'coverage':
                        new_subkey = 'visibility'
                        new_subvalue = 1-subvalue
                    else:
                        new_subvalue = subvalue
                    if subkey in lut:
                        new_subkey = lut[subkey]
                    else:
                        new_subkey = subkey
                    setattr(new_space, new_subkey, new_subvalue)
                setattr(D, new_key, new_space)
            else:
                occ_ns = SimpleNamespace()
                occ_exists = False
                if key in occlusion_params:
                    setattr(occ_ns, new_key, new_value)
                    occ_exists = True
                else:
                    setattr(D, new_key, new_value)
                if occ_exists:
                    setattr(D, 'Occlusion', occ_ns)

        T = SimpleNamespace()
        for key, value in T_orig.__dict__.items():
            if key in lut:
                new_key = lut[key]
            else:
                new_key = key
            setattr(T, new_key, value)

    setattr(M, 'identifier', op.basename(model_dir))
    setattr(M, 'model_dir', model_dir)

    CFG = SimpleNamespace(M=M, D=D, T=T)
    pkl.dump(CFG, open(f'{model_dir}/config.pkl', 'wb'))#, Dumper=Dumper)
    #CFG = yaml.load(open(f'{model_dir}/config.yaml', 'r+'), Loader=Loader)

    # save viewable text version of final configuration (appends to file in cases where training stopped and resumed)
    if CFG.M.identifier != 'occ-beh':
        os.makedirs(CFG.M.model_dir, exist_ok=True)
        config_txt = f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n\n'
        for param_type, param_space in zip(['model', 'dataset', 'training'], [CFG.M, CFG.D, CFG.T]):
            config_txt += f'### {param_type} ###\n'
            for param_name, param in param_space.__dict__.items():
                if not param_name.startswith('_') and param_name not in remove_params:
                    if type(param) is SimpleNamespace:
                        param_type_printed = False
                        for subparam_name, subparam in param.__dict__.items():
                            if not subparam_name.startswith('_') and subparam_name not in remove_params:
                                if not param_type_printed:
                                    config_txt += f'{param_name.ljust(32)}{subparam_name.ljust(32)}{subparam}\n'
                                    param_type_printed = True
                                else:
                                    config_txt += f'{subparam_name.ljust(32).rjust(64)}{subparam}\n'  # if param is another class
                    else:
                        config_txt += f'{param_name.ljust(32)}{param}\n'  # if param is a parameter
            config_txt += '\n\n'
        config_txt += '\n\n\n\n'
        config_path_txt = op.join(CFG.M.model_dir, 'config.txt')
        with open(config_path_txt, 'w+') as c:
            c.write(config_txt)
