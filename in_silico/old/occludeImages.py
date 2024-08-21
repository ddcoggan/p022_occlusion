import numpy as np
import glob
import os
from PIL import Image, ImageDraw
import random
import datetime
import torch
from torch import Tensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def occImage(images=None, method=None, coverage=.5, colour=(0,0,0), invert=False):

    r"""Adds occluders to image.

        Arguments:
            image (tensor):
            method (string): type of occlusion to apply. Options include:
                             barHorz (horizontal bars), polkadot, orientedNoise
            coverage (float, range 0:1): proportion of image to be occluded
            colour (tuple of length 3, range 1:255): RGB colour for occluded pixels
                                                    can be list of different colours
                                                    iterated over randomly.
            invert (bool): invert the occlusion pattern or not.

        Returns:
            occluded image (tensor)"""

    occImagePaths = sorted(glob.glob(f'DNN/images/occluders/{method}/{int(coverage*100)}/*.png'))
    occImagePath = random.choice(occImagePaths)
    occImage = torch.tensor(np.array(Image.open(occImagePath))).permute(2, 0, 1)/255 # load image, put in tensor, reshape, binarise
    occImageInv = 1 - occImage # get inverse of image

    # image size
    C, H, W = occImage.size()
    if occImage.size() != image.size():
        raise Exception('Occluder and image are different sizes!')

    # occluder colour
    if method != 'natural':
        if type(colour) == list:
            fillCol = torch.tensor(colour[np.random.randint(len(colour))])
        elif type(colour) == tuple:
            fillCol = torch.tensor(colour)
        colouredImage = fillCol.repeat(H, W).view([H, W, 3]).permute(2, 0, 1)
        if invert:
            image *= occImageInv  # zero masked pixels
            colouredMask = colouredImage * occImage  # create coloured mask
        else:
            image *= occImage  # zero masked pixels
            colouredMask = colouredImage * occImageInv  # create coloured mask
        image += colouredMask
        return image

    # if occluding with another image
    else:
        occImage = torch.tensor(np.array(Image.open(occImagePath))).permute(2, 0, 1)/255 # load image, put in tensor, reshape
        occImageBin = torch.Tensor.repeat(torch.where(occImage.sum(0) < 3, 1, 0), (3, 1, 1))
        occImageBinInv = 1-occImageBin
        imageMasked = image * occImageBinInv
        occImageMasked = occImage * occImageBin
        image = imageMasked + occImageMasked
        return image

#plt.imshow(image.permute(1, 2, 0))
#plt.show()
class occludeImage(torch.nn.Module):

    def __init__(self, method, coverage, colour, invert):
        super().__init__()
        self.method = method
        self.coverage = coverage
        self.colour = colour
        self.invert = invert

    def forward(self, tensor: Tensor) -> Tensor:

        return occImage(tensor, self.method, self.coverage, self.colour, self.invert)

