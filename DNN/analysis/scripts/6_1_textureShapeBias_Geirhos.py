#### Summary ###########################################################################################################
# Check Texture versus Shape bias - Geirhos.
########################################################################################################################

import os
import sys
import random
import time
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import collections
import csv
import glob
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from PIL import Image

sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
import zoo
from DNN.analysis.scripts.texture_vs_shape.code import probabilities_to_decision

overwrite = True

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""
    # Override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # Make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path, index))
        return tuple_with_path

# def main():
def main(model_path):

    model_name = 'alexnet'
    pretrained = False
    num_categories = 1000
    train_batch_size = 256
    val_batch_size = 1
    start_epoch = 0
    num_epochs = 70 # 100
    save_every_epoch = 10
    initial_learning_rate = 1e-3 # SGD, 0.01; Adam, 0.0001
    gpu_ids = [1]
    input_size = [224, 224]

    #### Create/Load model #############################################################################################
    # 1. If pre-trained models used without pre-trained weights. e.g., model = models.vgg19()
    # 2. If pre-trained models used with pre-trained weights. e.g., model = models.vgg19(pretrained=True)
    # 3. If our models used.
    ####################################################################################################################

    # get model
    if not model_name.startswith('cornet'):
        model = getattr(zoo, model_name)(pretrained=pretrained)
    else:
        model = getattr(cornet, model_name)
        model = model(pretrained=pretrained)


    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
    elif len(gpu_ids) == 1:
        device = torch.device('cuda:%d'%(gpu_ids[0]))
        torch.cuda.set_device(device)
        model.cuda()
        model.to(device)

    loss_function = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)

    weightFiles = sorted(glob.glob(os.path.join(model_path, 'params/*.pt')))
    weightsPath = weightFiles[-1]
    weights = torch.load(weightsPath)

    # adapt weights dict to not include 'module' in keys
    newDict = {}
    for key in weights['model'].keys():
        newDict[str(key)[7:]] = weights['model'][key]

    del weights['model']
    weights['model'] = newDict

    model.load_state_dict(weights['model'])
    optimizer.load_state_dict(weights['optimizer'])


    #### Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[51], gamma=0.1, last_epoch=start_epoch-1)
    lr_scheduler.step()

    #### Add softmax layer
    if model_name == 'ResNet' or model_name == 'GoogLeNet' or model_name == 'Inception3':
        if len(gpu_ids) > 1:
            model.module.fc = nn.Sequential(model.module.fc, nn.Softmax(dim=1),)
        else:
            model.fc = nn.Sequential(model.fc, nn.Softmax(dim=1),)
    else:
        if len(gpu_ids) > 1:
            model.module.classifier = nn.Sequential(*list(model.module.classifier) + [nn.Softmax(dim=1)])
        else:
            model.classifier = nn.Sequential(*list(model.classifier) + [nn.Softmax(dim=1)])

    #### Data loader ###################################################################################################
    # val_dataset = torchvision.datasets.ImageFolder(
    val_dataset = ImageFolderWithPaths("DNN/analysis/scripts/texture_vs_shape/stimuli/style-transfer-preprocessed-512",
        transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.Resize(input_size[1:]),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]) # grayscale
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # rgb
        ]))
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=False)

    # # val_dataset_ref = torchvision.datasets.ImageFolder(
    # val_dataset_ref = ImageFolderWithPaths(
    #     "/home/tonglab/Documents/Data/ILSVRC2012/images/val")
    # val_loader_ref = torch.utils.data.DataLoader(val_dataset_ref, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=False)

    #### Val ###########################################################################################################
    model.eval()
    csv_filename = f'{model_path}/texture_shape_bias.csv'
    if os.path.exists(csv_filename):
        print('The file exists')
    else:
        with open(csv_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["subj", "session", "trial", "rt", "object_response", "category", "condition", "imagename"])

    for batch_index, (inputs, targets, paths, indices) in enumerate(val_loader):

        image_index = []
        for path, target in zip(paths, targets):
            image_index.append(val_loader.dataset.samples.index((path, target)))

        if len(gpu_ids) >= 1:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        inputs = inputs.repeat(1, 3, 1, 1)  # Grayscale to RGB

        softmax_output = model(inputs)
        softmax_output_np = np.squeeze(np.array(softmax_output.cpu().detach()))
        mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
        decision_from_16_classes = mapping.probabilities_to_decision(softmax_output_np)

        target_category = paths[0].split('/')
        target_category = target_category[-2]

        with open(csv_filename, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["CNN", "1", image_index[0], "NaN", decision_from_16_classes, target_category, 0, paths[0]])

def is_correct(output, target, topk=1):
    with torch.no_grad():
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        num_correct = correct[:topk].contiguous().view(-1).float().sum(0, keepdim=True) # top5 glitch

        return correct, num_correct

categories = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
occlusionMethods = ['naturalTextured2']#[os.path.basename(x) for x in sorted(glob.glob('DNN/images/occluders/*'))]
occlusionMethods.append('unoccluded')
coverages = [.1,.2,.4,.8]
model = 'alexnet'
dataset = 'imagenet1000'

values = []
for occlusionMethod in occlusionMethods:

    modelDir = f'DNN/data/{model}/{dataset}/fromPretrained/{occlusionMethod}'

    outFile = f'{modelDir}/texture_shape_bias.csv'

    if not os.path.isfile(outFile) or overwrite:
        main(modelDir)

    modelDecision = np.zeros(4)
    check = np.ones(16)
    decisionMatrix = pd.read_csv(outFile)

    for row in range(len(decisionMatrix)):
        prediction = decisionMatrix['object_response'][row]
        info = decisionMatrix['imagename'][row].split('/')[-1].split('-')
        shape = info[0][:3] # shortest label is 3 chars long and first 3 chars distinguish all categories
        texture = info[1][:3]
        if prediction != shape and prediction != texture:
            modelDecision[0] += 1
        elif prediction != shape and prediction == texture:
            modelDecision[1] += 1
        elif prediction == shape and prediction != texture:
            modelDecision[2] += 1
        elif prediction == shape and prediction == texture:
            modelDecision[3] += 1
    shapeBias = modelDecision[2] / (modelDecision[1] + modelDecision[2])
    print(shapeBias)
    values.append(shapeBias)

# plot
fig, ax = plt.subplots(figsize=(2.5, 3))
plt.bar(np.arange(2), values)
plt.tick_params(direction='in')
plt.xticks(np.arange(2), labels = occlusionMethods, rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel('occlusion type trained on')
ax.set_ylabel('shape bias index')
plt.ylim(0, 0.35)
plt.title(f'{model} trained on {dataset.split("_")[0]}', size=8)
plt.tight_layout()
outDir = f'DNN/analysis/results/shapeTextureBias/{model}/{dataset}/Geirhos'
os.makedirs(outDir, exist_ok=True)
fig.savefig(os.path.join(outDir, f'{model}_{dataset}.png'))
plt.show()
plt.close()


        


