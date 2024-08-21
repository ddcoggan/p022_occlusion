'''
This scripts tiles sample training inputs for quick inspection
'''

import os.path as op
import glob
import sys

sys.path.append(op.expanduser('~/david/master_scripts/image'))
from image_processing import tile

def tile_sample_inputs(model_dirs, overwrite=False):

    # start of model loop
    for m, model_dir in enumerate(model_dirs):

        # image directory
        imagedir = f'{model_dir}/sample_train_inputs'
        image_paths = sorted(glob.glob(f'{imagedir}/*.png'))[:30]
        outpath = f'{imagedir}/tiled.png'

        if image_paths and (not op.isfile(outpath) or overwrite):
            tile(image_paths, outpath, base_gap=8, num_cols=8)

        # do this for pretraining too if this is a transfer-learned model
        if 'transfer' in model_dir:
            imagedir_orig = f'{op.dirname(model_dir)}/sample_train_inputs'
            image_paths = sorted(glob.glob(f'{imagedir_orig}/*.png'))[:50]
            outpath = f'{imagedir_orig}/tiled.png'
            if image_paths and (not op.isfile(outpath) or overwrite):
                tile(image_paths, outpath, num_cols=8, base_gap=8, colgap=32,
                     colgapfreq=2, rowgap=32, rowgapfreq=1)


if __name__ == "__main__":

    import time
    from .config import model_dirs

    sys.path.append(op.expanduser('~/david/master_scripts/misc'))
    from seconds_to_text import seconds_to_text

    start = time.time()
    tile_sample_inputs(model_dirs, overwrite=True)
    finish = time.time()

    print(f'analysis took {seconds_to_text(finish - start)} to complete')
