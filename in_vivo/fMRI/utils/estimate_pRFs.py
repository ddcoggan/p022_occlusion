# /usr/bin/python
# Created by David Coggan on 2023 06 28
import itertools
import os
import os.path as op
import numpy as np
import pandas as pd

from config import PROJ_DIR

def ecc_to_size(ecc, method):

    def voxel_size_adjustment(voxel_size, ndims=1):
        if ndims == 1:
            scale_factor = 2 / np.mean(voxel_size)
        elif ndims == 2:
            my_area = 4 * 6  # 2mm isotropic
            their_area = np.sum([x * y for x, y in itertools.combinations(
                voxel_size, 2)] * 2)
            scale_factor = my_area / their_area
        return scale_factor

    if method == 'dumoulin':
        scale = voxel_size_adjustment([3, 2.5, 2.5], ndims=1)
        unit = 'sigma'
        a = np.array([2, .4])
        b = np.array([12, 1])
        slope = scale * (b[1] - a[1]) / (b[0] - a[0])
        intercept = scale * (a[1] - (a[0] * slope))
    elif method == 'HCP':
        scale = voxel_size_adjustment([1.6, 1.6, 1.6], ndims=1)
        unit = 'sigma'
        intercept = .1856 * scale
        slope = .02676 * scale
    else:  # poltoratski
        unit = 'FWHM'
        #intercept, slope = .3424, .199
        intercept, slope = .401, .165

    size = intercept + slope * ecc

    if unit == 'sigma':
        size *= 2.355

    return size


def estimate_pRFs():

    eccs = np.arange(1, 5)
    data = {'dumoulin': [], 'poltoratski': [], 'HCP': []}

    for method, ecc in itertools.product(data, eccs):
        data[method].append(ecc_to_size(ecc, method))
    data['eccentricity'] = eccs

    pRF_estimates = pd.DataFrame(data)
    outpath = f'{PROJ_DIR}/in_vivo/fMRI/figures/pRF_estimates.csv'
    pRF_estimates.to_csv(outpath, index=False)


if __name__ == "__main__":

    estimate_pRFs()
