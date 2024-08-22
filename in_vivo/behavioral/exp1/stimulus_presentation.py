# Image Classification experiment

import os, glob, sys, csv
import time
import serial
import numpy as np
import random
import pandas as pd
from PIL import Image
import pickle
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from psychopy import visual, event, core, monitors, gui, logging
from psychopy.hardware import keyboard

"""
This is a psychopy stimulus presentation script used for behavioral exp 1.

Design:
- Subjects viewed object images sequentially, performed 8 AFC classification.
- Each trial begins with a 500ms fixation period, followed by a 100ms 
  stimulus presentation, then a pink-noise mask until a response is made.
- Objects are from 8 imagenet classes, 4 animate, 4 inanimate. See the CFG 
  class in the analysis script for more details e.g., their Imagenet synsets.
- 784 object images (98 per class) were selected (all 50 validation 
  images per class, plus 48 additional images from the training set).
- 32 object images were selected for a short practice session to 
  familiarize the subject with the task and response key mapping.
- Remaining 752 object images were used in the main experiment.
- 90 occluder conditions: 9 types * 5 coverage levels * 2 colors.
- 1 trial for each combination of object class and occluder condition (720)
- An additional 4 trials per object class without occlusion (32).

Comments:
- Occluder level or coverage is more commonly implemented elsewhere as 
  'visibility', which is 1 - coverage.
- Naming conventions are slightly different elsewhere, but this is already 
  handled and so not updated here. 
- A coding error found after data collection led to the same image being 
  used for all 4 unoccluded trials of each object class.
- The pairing of object images and occluders was randomized for each subject, 
  a measure chosen to increase SNR of the effects of different occluder 
  features on classification accuracy (through increasing the 
  number of unique stimuli from 752 to 22560). The trade-off is that it is not 
  possible to establish human-human similarity at the level of individual 
  trials (e.g., error consistency), although this is still possible at the 
  level of conditions (e.g. set of accuracy scores for the different 
  conditions).
- Short presentation times and noise-masking were implemented:
    1. to limit extensive recurrent processing by humans, thereby affording  
       shallower/feedforward models a better chance of modelling responses.
    2. to more closely align the inflection points in the accuracy/visibility 
       performance curves of humans and models. Pilot studies showed that, 
       if given unlimited time to view an image, humans approach ceiling 
       performance at low visibility levels. At these levels, many CNNs would
       not perform substantially better than chance, limiting the 
       interpretability of their performance on human similarity measures.
    3. to increase the number of trials that can be completed in a session.
"""

#################
# configuration #
#################

# get experiment info
exp_info = {'subject': '',
           'date': time.strftime('%y%m%d.%H:%M:%S')}
dlg = gui.DlgFromDict(exp_info, title="Tonglab Experiment",
                      fixed=['date'], order=['subject', 'date'])
if not dlg.OK:
    sys.exit(0)

# directories
proj_dir = '/Users/tonglab/Desktop/Dave/p022/behavioural'
log_dir = os.path.join(proj_dir, 'exp1/data/logFiles', exp_info['subject'])
os.makedirs(log_dir, exist_ok=True)

# timing (in seconds)
refresh_rate = 1/60
block_intro_duration = 2
ISI_duration = 0.5
stim_duration = 0.100 # will be rounded based on screen refresh rate (to 16.7 ms for 60Hz)
stim_duration_flips = round(stim_duration/refresh_rate)

# stimulus and display
image_size = 256  # in pixels
image_size_deg = 10  # in degrees of visual angle
background_colour = (0,0,0) # black=(-1,-1,-1); mid-gray=(0,0 0); white=(1,1,1)
monitor_name = 'Dell U2715Hc'

# conditions
occlusion_types = ['unoccluded', 'barHorz04', 'barVert04', 'barObl04', 'mudsplash', 'polkadot','polkasquare','crossBarOblique','crossBarCardinal', 'naturalUntexturedCropped2']
occlusion_levels = [.9,.8,.6,.4,.2]  # approximate occluder coverage levels
occluder_lums = {'black': 0, 'white': 1}  # 0=black, 1=white
categories = ['bear', 'bison', 'elephant', 'hare', 'jeep', 'lamp', 'sportsCar', 'teapot']
trials_per_cond = 1  # number of images for each combination of the above variables
trials_per_cat_unoccluded = 4  # number of unoccluded images per category

# responses
response_keys = ['1', '2', '3', '4', '5', '6', '7', '8']
center_x = -8
response_locs_x = np.array([-10, -8, -6, -4, 4, 6, 8, 10]) + center_x  # x axis locations of responses
response_locs_y = [-6, -7, -8, -9, -9, -8, -7, -6]

########################
# end of configuration #
########################

# get random seed based on time
rand_seed_path = f'{log_dir}/randSeed.pkl'
if os.path.isfile(rand_seed_path):
    rand_seed = pickle.load(open(rand_seed_path, 'rb'))
else:
    rand_seed = time.time()  # current time is the random seed
    pickle.dump(rand_seed, open(rand_seed_path, 'wb'))

# set whether animate keys are on left or right
animate_left = int(exp_info['subject']) % 2

# randomly shuffle category keys
animate_categories, inanimate_categories = [categories[:4], categories[4:]]
animate_categories_shuffle = animate_categories.copy()
random.Random(rand_seed).shuffle(animate_categories_shuffle)
inanimate_categories_shuffle = inanimate_categories.copy()
random.Random(rand_seed).shuffle(inanimate_categories_shuffle)
if animate_left == 1:
    categories_shuffle = animate_categories_shuffle + inanimate_categories_shuffle
else:
    categories_shuffle = inanimate_categories_shuffle + animate_categories_shuffle

n_images_per_cat = len(glob.glob(os.path.join(proj_dir, 'objects', categories[0])))
trials_dict = {'occlusion_type': [],
               'occlusion_level': [],
               'occlusion_colour': [],
               'category': [],
               'object_path': [],
               'occluder_path': [],
               'occluded_object_path': [],
               'noise_path': []}
trial_counter = 0
noise_paths = glob.glob(os.path.join(proj_dir, 'images/noise/*'))
random.Random(rand_seed).shuffle(noise_paths)

for c in categories:
    exemplar_counter = 0
    object_paths = glob.glob(os.path.join(proj_dir, 'images/objects', c, '*.*'))
    random.Random(rand_seed).shuffle(object_paths)
    for ot in occlusion_types:

        # unoccluded
        if ot == 'unoccluded':
            for t in range(trials_per_cat_unoccluded):

                ## resize to desired image size, cropping if necessary to avoid stretching/squeezing
                object_path = object_paths[exemplar_counter] # use following index if trials > object images: [exemplar_counter % len(object_paths)]
                out_dir = f'{log_dir}/stimuli'
                os.makedirs(out_dir, exist_ok=True)
                object_pil = Image.open(object_path).convert('L')
                image_size = 256  # desired size
                old_im_size = object_pil.size
                min_length = min(old_im_size)
                smallest_dim = old_im_size.index(min_length)
                biggest_dim = np.setdiff1d([0, 1], smallest_dim)[0]
                new_max_length = int((image_size / old_im_size[smallest_dim]) * old_im_size[biggest_dim])
                new_shape = [0, 0]
                new_shape[smallest_dim] = image_size
                new_shape[biggest_dim] = new_max_length
                resized_image = object_pil.resize(new_shape)
                left = int((new_shape[0] - image_size) / 2)
                right = new_shape[0] - left
                top = int((new_shape[1] - image_size) / 2)
                bottom = new_shape[1] - top
                cropped_object = resized_image.crop((left, top, right, bottom))
                out_path = f'{out_dir}/{c}_{ot}_0_nan_{t}.png'
                cropped_object.save(out_path)

                trials_dict['occlusion_type'].append(ot)
                trials_dict['occlusion_level'].append(0)
                trials_dict['occlusion_colour'].append(None)
                trials_dict['category'].append(c)
                trials_dict['object_path'].append(object_path)
                trials_dict['occluder_path'].append(None)
                trials_dict['occluded_object_path'].append(out_path)
                trials_dict['noise_path'].append(noise_paths[trial_counter % len(noise_paths)])

                trial_counter += 1

        # occluded
        else:
            for ol in occlusion_levels:
                occluder_paths = glob.glob(os.path.join(proj_dir, 'images/occluders', ot, f'{int(ol * 100)}/*'))
                for lum in occluder_lums:
                    l = occluder_lums[lum]
                    for t in range(trials_per_cond):
                        object_path = random.Random(rand_seed + trial_counter).sample(object_paths, 1)[0]
                        occluder_path = random.Random(rand_seed + trial_counter).sample(occluder_paths, 1)[0]
                        trials_dict['occlusion_type'].append(ot)
                        trials_dict['occlusion_level'].append(ol)
                        trials_dict['occlusion_colour'].append(lum)
                        trials_dict['category'].append(c)
                        trials_dict['object_path'].append(object_path)
                        trials_dict['occluder_path'].append(occluder_path)

                        # make occluded image
                        ## load image and center crop to square
                        object_pil = Image.open(trial.object_path).convert('L')
                        old_im_size = object_pil.size
                        min_length = min(old_im_size)
                        smallest_dim = old_im_size.index(min_length)
                        biggest_dim = np.setdiff1d([0, 1], smallest_dim)[0]
                        new_max_length = int(
                            (image_size / old_im_size[smallest_dim]) *
                            old_im_size[biggest_dim])
                        new_shape = [0, 0]
                        new_shape[smallest_dim] = image_size
                        new_shape[biggest_dim] = new_max_length
                        resized_image = object_pil.resize(new_shape)
                        left = int((new_shape[0] - image_size) / 2)
                        right = new_shape[0] - left
                        top = int((new_shape[1] - image_size) / 2)
                        bottom = new_shape[1] - top
                        processed_object = resized_image.crop(
                            (left, top, right, bottom))

                        ## convert to array and combine with occluder
                        object_array = np.array(processed_object) / 255  # load image, put in array, downscale
                        object_array = object_array[:image_size, :image_size]  # fixes occasional enlargening
                        occluder_pil = Image.open(occluder_path).convert('RGBA')
                        occluder_array = np.array(occluder_pil) / 255  # load image, put in array, downscale
                        alpha_layer = occluder_array[:, :, 3]
                        alpha_layer_inv = 1 - alpha_layer
                        new_image_array = np.zeros_like(object_array)
                        new_image_array += object_array * alpha_layer_inv
                        new_image_array += l * alpha_layer
                        new_image_array *= 255
                        out_image = Image.fromarray(new_image_array.astype(np.uint8))
                        out_dir = f'{log_dir}/stimuli'
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = f'{out_dir}/{c}_{ot}_{ol}_{lum}_{t}.png'
                        out_image.save(out_path)
                        trials_dict['occluded_object_path'].append(out_path)
                        trials_dict['noise_path'].append(noise_paths[trial_counter % len(noise_paths)])
                        trial_counter += 1

trials = pd.DataFrame(trials_dict)
trials_shuffle = trials.sample(frac=1, random_state=int(rand_seed)).reset_index()
pickle.dump(trials_shuffle, open(f'{log_dir}/trials.pkl', 'wb'))

# allow quit function
def allow_quit():
    for keys in event.getKeys():
        if keys in ['escape', 'q']:
            win.close()
            core.quit()

# Create window and image
kb = keyboard.Keyboard()
win = visual.Window(size=monitors.Monitor(monitor_name).getSizePix(),
                    fullscr=True,
                    allowGUI=False,
                    monitor='testMonitor',
                    color=background_colour)
load_msg = visual.TextStim(win, text='Loading images...', height=1, units='degFlat')
load_msg.draw()
win.flip()
cache_images = []
cache_noise = []
for t in range(len(trials)):
    cache_images.append(visual.ImageStim(win, image=trials_shuffle['occluded_object_path'][t], size=image_size_deg, units='degFlat', pos=(0, 0)))
    cache_noise.append(visual.ImageStim(win, image=trials_shuffle['noise_path'][t], size=image_size_deg, units='degFlat', pos=(0, 0)))

# Load our fixation cross image too
fixation_outer = visual.Circle(win, size=(.5), units='degFlat', pos=(0, 0), fillColor='white')
fixation_inner = visual.Circle(win, size=(.25), units='degFlat', pos=(0, 0), fillColor='black')

# Wait for a key press before starting
visual.TextStim(win, text='Experiment ready.\n\nPress spacebar when ready to begin...', height=1, units='degFlat').draw()
win.flip()
event.waitKeys(keyList=['space'])

# start with 2s fixation before first image
fixation_outer.draw()
fixation_inner.draw()
win.flip()
core.wait(2)
experiment_clock = core.Clock()
logging.setDefaultClock(experiment_clock)

responses=[]
correct=[]
RTs=[]
break_counter = 0

cache_object = visual.ImageStim(win, image=trials_shuffle['occluded_object_path'][0], size=image_size_deg, units='degFlat', pos=(0, 0))
cache_noise = visual.ImageStim(win, image=trials_shuffle['noise_path'][0], size=image_size_deg, units='degFlat', pos=(0, 0))

try:
    for t in range(len(trials)):

        kb.clock.reset()  # start RT timer

        for f in range(stim_duration_flips):
            # draw image
            cache_object.draw()

            # draw fixation
            fixation_outer.draw()
            fixation_inner.draw()

            # draw response keys WARNING! skipped as this cannot be performed in a single flip on 417D Mac Pro
            # for k, key in enumerate(response_keys):
            #    visual.TextStim(win, text=f'{key} - {categories_shuffle[k]}', pos=(responseLocsX[k], responseLocsY[k]), height=1, units='degFlat', anchorHoriz='left').draw()

            # show stimulus
            win.flip()

            # print(time.time()) # use this to investigate stimulus timing

        # replace with noise image, and do this without drawing response map to quickly remove the stimulus from the screen
        cache_noise.draw()
        fixation_outer.draw()
        fixation_inner.draw()
        win.flip()

        # now draw the full window with response map
        cache_noise.draw()

        # draw fixation
        fixation_outer.draw()
        fixation_inner.draw()

        # draw response keys
        for k, key in enumerate(response_keys):
            visual.TextStim(win, text=f'{key} - {categories_shuffle[k]}', pos=(response_locs_x[k], response_locs_y[k]),
                            height=1, units='degFlat', anchorHoriz='left').draw()

        # show stimulus
        win.flip()

        # wait for valid key press and record response
        response_key = event.waitKeys(keyList=response_keys)
        RT = kb.clock.getTime()
        RTs.append(RT)
        response_cat = categories_shuffle[int(response_key[0])-1]
        responses.append(response_cat)
        if response_cat == trials_shuffle['category'][t]:
            correct.append(1)
        else:
            correct.append(0)
        #print(f'category: {trials_shuffle["category"][t]}, response: {response_cat}, RT: {RT}')

        # check for a quit key
        allow_quit()

        # Now show the fixation cross image
        fixation_outer.draw()
        fixation_inner.draw()
        win.flip()

        # ISI (load next image while waiting)
        ISI_start = experiment_clock.getTime()
        if t+1 < len(trials_shuffle):
            cache_object = visual.ImageStim(win, image=trials_shuffle['occluded_object_path'][t+1], size=image_size_deg,
                                            units='degFlat', pos=(0, 0))
            cache_noise = visual.ImageStim(win, image=trials_shuffle['noise_path'][t+1], size=image_size_deg, units='degFlat',
                                           pos=(0, 0))
        while experiment_clock.getTime() < ISI_start + ISI_duration:
            core.wait(0.017) # wait one frame refresh

        # self-timed break
        if (t+1) % int(len(trials)/4) == 0 and break_counter < 3:
            break_counter += 1
            visual.TextStim(win, text=f'block {break_counter} of 4 complete, press spacebar to continue', pos=(center_x, 0),
                            height=1, units='degFlat', anchorHoriz='left').draw()

            win.flip()
            event.waitKeys(keyList=['space'])

            fixation_outer.draw()
            fixation_inner.draw()
            win.flip()
            core.wait(2)

except KeyboardInterrupt:
    win.close()
    pickle.dump(trials_shuffle, open(f'{log_dir}/trials.pkl', 'wb'))
    core.quit()

trials_shuffle['response'] = responses
trials_shuffle['RT'] = RTs
trials_shuffle['correct'] = correct
pickle.dump(trials_shuffle, open(f'{log_dir}/trials.pkl', 'wb'))

print(f'experiment duration: {experiment_clock.getTime()} | mean accuracy: {np.mean(correct)}')

trials_shuffle['occlusion_type'] = trials_shuffle['occlusion_type'].astype('category')
trials_shuffle['visibility'] = np.round((1-trials_shuffle['occlusion_level'])*100).astype('int').astype('category')

for metric in ['correct','RT']:

    # get unoccluded performance
    unoccPerf = trials_shuffle[metric][trials_shuffle['occlusion_type'] == 'unoccluded'].mean()

    # remove unoccluded data
    occludedTrials = trials_shuffle[trials_shuffle['occlusion_type'] != 'unoccluded']
    pd.options.mode.chained_assignment = None # stop SettingWithCopyWarning
    occludedTrials['occlusion_type'] = occludedTrials['occlusion_type'].cat.remove_unused_categories() # drop unoccluded level
    occludedTrials['visibility'] = occludedTrials['visibility'].cat.remove_unused_categories() # drop unoccluded level

    # get means and sems
    trialsMean = occludedTrials.groupby(['occlusion_type','visibility'], as_index=False).mean()
    trialsSEM = occludedTrials.groupby(['occlusion_type','visibility'], as_index=False).sem()

    plt.figure(figsize = (10,3))
    colours = list(mcolors.TABLEAU_COLORS.keys())
    for v, visibility in enumerate(trialsMean['visibility'].unique()):
        means = trialsMean[metric][trialsMean['visibility'] == visibility]
        sems = trialsSEM[metric][trialsSEM['visibility'] == visibility]
        barWidth = 1/6
        xshift = (v*barWidth)-(2*barWidth)
        plt.bar(np.arange(len(occlusion_types)-1)+xshift, means, yerr=sems, width = barWidth, color = mcolors.TABLEAU_COLORS[colours[v]], label = visibility)
    plt.xticks(np.arange(len(occlusion_types)-1), labels=occlusion_types[1:], rotation=25, ha='right')
    plt.tick_params(direction = 'in')
    plt.axhline(y=unoccPerf, color=mcolors.TABLEAU_COLORS[colours[v+1]], linestyle = 'dashed')
    if metric == 'correct':
        plt.axhline(y=1/8, color = 'k', linestyle = 'dotted')
    plt.title(f'behavioural performance ({metric}), subject {exp_info["subject"]}')
    plt.legend(title='visibility (%)',bbox_to_anchor=(1.04,1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/performance_{metric}.png')
    plt.show()

win.close()
core.quit()


