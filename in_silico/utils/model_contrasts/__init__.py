# /usr/bin/python
# Created by David Coggan on 2024 01 04
import os
import os.path as op
import torch
import torchvision
import numpy as np
import h5py
import glob

# get all model contrasts and place in dict
from .cornet_rt import models as cornet_rt
from .cornet_s_custom_base import models as cornet_s_custom_base
from .cornet_s_custom_large import models as cornet_s_custom_large
from .cornet_s_custom_recurrence import models as cornet_s_custom_recurrence
from .cornet_s_V1 import models as cornet_s_V1
from .cognet import models as cognet
from .public_models import models as public_models
from .cornet_s_unshared import models as cornet_s_unshared
from .vit import models as vit
from .VSS_2024_abstract import models as VSS_2024_abstract
from .VSS_2024_poster import models as VSS_2024_poster
from .CCN_2024_abstract import models as CCN_2024_abstract
from .CCN_2024_poster_exp1 import models as CCN_2024_poster_exp1
from .CCN_2024_poster_exp2 import models as CCN_2024_poster_exp2
from .generative_models import models as generative_models
from .main_effects import model_contrasts as main_effects
from .main_effects import effect_colors

model_contrasts = dict(
    #public_models=public_models,
    #VSS_2024_abstract=VSS_2024_abstract,
    #VSS_2024_poster=VSS_2024_poster,
    #CCN_2024_abstract=CCN_2024_abstract,
    CCN_2024_poster_exp1=CCN_2024_poster_exp1,
    CCN_2024_poster_exp2=CCN_2024_poster_exp2,
    #cornet_rt=cornet_rt,
    #cornet_s_custom_base=cornet_s_custom_base,
    #cornet_s_custom_large=cornet_s_custom_large,
    #cornet_s_custom_recurrence=cornet_s_custom_recurrence,
    #cornet_s_unshared=cornet_s_unshared,
    #cornet_s_V1=cornet_s_V1,
    #cognet=cognet,
    #vit=vit,
    #**main_effects,
)
model_base = op.expanduser('~/david/models')

# ensure weights for publicly available models are downloaded
if 'public_models' in model_contrasts:
    for model, info in public_models.items():
        params_dir = f'{model_base}/{info["path"]}/params'
        os.makedirs(params_dir, exist_ok=True)
        if not len(os.listdir(params_dir)):
            model_name = info["path"].split('/')[0]
            model = getattr(torchvision.models, model_name)(weights=info['weights'])
            torch.save(model.state_dict(), f'{params_dir}/{info["weights"]}.pt')

# list all unique model directories
model_dirs = set()
for contrast, models in model_contrasts.items():
    for model, info in models.items():
        model_dirs.add(f'{model_base}/{info["path"]}')
model_dirs = sorted(model_dirs)

# specify layer mapping to human brain regions
identity_mapping = {'V1': 'V1', 'V2': 'V2', 'V4': 'V4', 'IT': 'IT'}
identity_mapping_output = {'V1': 'V1.output', 'V2': 'V2.output',
                           'V4': 'V4.output', 'IT': 'IT.output'}
cognet_mapping = {'V1': 'V1.f', 'V2': 'V2.f', 'V4': 'V4.f', 'IT': 'IT.f'}
region_to_layer = {
    'cornet_s': identity_mapping,
    'cornet_s_unshared': identity_mapping,
    'cornet_s_V1': identity_mapping,
    'cornet_s_hw3': identity_mapping,
    'cornet_s_hw7': identity_mapping,
    'cornet_s_hd2_hw3': identity_mapping,
    'cornet_s_custom': identity_mapping_output,
    'cornet_rt': identity_mapping,
    'cornet_rt_hw3': identity_mapping,
    'cornet_z': identity_mapping,
    'cognet_v9': cognet_mapping,
    'cognet_v10': cognet_mapping,
    'alexnet': {
        'V1': 'features.2',
        'V2': 'features.7',
        'V4': 'features.7',
        'IT': 'features.12'},
    'resnet101': {
        'V1': 'layer3.6',
        'V2': 'layer3.12',
        'V4': 'layer3.18',
        'IT': 'layer4.2'},
    'resnext101_32x8d': {
        'V1': 'layer3.6',
        'V2': 'layer3.12',
        'V4': 'layer3.18',
        'IT': 'layer4.2'},
    'resnet152': {
        'V1': 'layer2.7',
        'V2': 'layer3.12',
        'V4': 'layer3.25',
        'IT': 'layer4.2'},
    'vgg19': {
        'V1': 'features.9',
        'V2': 'features.18',
        'V4': 'features.27',
        'IT': 'features.36'},
    'vit_b_16': {
        'V1': 'encoder.layers.encoder_layer_0',
        'V2': 'encoder.layers.encoder_layer_3',
        'V4': 'encoder.layers.encoder_layer_7',
        'IT': 'encoder.layers.encoder_layer_11.ln_2'},
    'pix2pix': {
        'V1': 'encoder1',
        'V2': 'encoder3',
        'V4': 'encoder5',
        'IT': 'encoder8'},
}
layer_to_region = {model_key: {v: k for k, v in model_vals.items()} for
                   model_key, model_vals in region_to_layer.items()}

# recurrent models return a 2-level activation dict (layer, cycle)
rec_models = ['cornet_s_custom', 'cornet_rt_hw3']

# list all unique generative model directories
model_dirs_gen = []
for model, info in generative_models.items():
    model_dirs_gen.append(f'{model_base}/{info["path"]}')
model_dirs_gen = sorted(set(model_dirs_gen))

# convert prednet weights from keras to torch format
params_path = f'{model_base}/prednet/pretrained/params/pretrained.pt'
if not op.isfile(params_path):

    weights_file = op.expanduser('~/david/repos/PredNet_pytorch/'
                                 'model_data_keras2/prednet_kitti_weights.hdf5')
    weights_f = h5py.File(weights_file, 'r')
    pred_weights = weights_f['pred_net_1']['pred_net_1']	# contains 23 item: 4x4(i,f,c,o for 4 layers) + 4(Ahat for 4 layers) + 3(A for 4 layers)
    keras_items = ['bias:0', 'kernel:0']
    pytorch_items = ['weight', 'bias']
    ks_mods = ['a', 'ahat', 'c', 'f', 'i', 'o']
    ks_mods = ['layer_' + m + '_' + str(i) for m in ks_mods for i in range(4)]
    ks_mods.remove('layer_a_3')
    assert len(ks_mods) == 4 * 4 + 4 + 3
    
    pt_mods_1 = ['A', 'Ahat']
    pt_mods_2 = ['c', 'f', 'i', 'o']
    pt_mods_1 = [m + '.' + str(2 * i) + '.' + item for m in pt_mods_1 for i in range(4) for item in pytorch_items]
    pt_mods_1.remove('A.6.weight')
    pt_mods_1.remove('A.6.bias')
    pt_mods_2 = [m + '.' + str(i) + '.' + item for m in pt_mods_2 for i in range(4) for item in pytorch_items]
    pt_mods = pt_mods_1 + pt_mods_2
    assert len(pt_mods) == (4 * 4 + 4 + 3) * 2
    
    weight_dict = dict()
    for i in range(len(ks_mods)):
        weight_dict[pt_mods[i * 2 + 1]] = pred_weights[ks_mods[i]]['bias:0'][:]
        weight_dict[pt_mods[i * 2]] = np.transpose(pred_weights[ks_mods[i]]['kernel:0'][:], (3, 2, 1, 0))
    for k, v in weight_dict.items():
        weight_dict[k] = torch.from_numpy(v).float().cuda()
    torch.save(weight_dict, params_path)




