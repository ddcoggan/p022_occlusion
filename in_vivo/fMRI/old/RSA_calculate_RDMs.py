#!/usr/bin/python
"""
script for making and plotting RDMs
"""

import os
import os.path as op
import glob
import numpy as np
import pickle as pkl
import nibabel as nib
import json
import time

from utils.seconds_to_text import seconds_to_text
from in_vivo.fMRI.utils import fMRI

def get_responses(subject, region, localiser):
	thr = 3.1

	subj_dir = f'derivatives/FEAT/sub-{subject}'
	run_dirs = sorted(glob.glob(f'{subj_dir}/ses-?/task-occlusion/*.feat'))
	n_runs = len(run_dirs)

	# load mask
	roi_path = f'derivatives/masks/MNI152_2mm/{region}.nii.gz'
	roi_data = nib.load(roi_path).get_fdata().flatten()

	# load localizer contrast map
	loc_path = f'{subj_dir}/task-occlusion_runs-all.gfeat/cope25.feat/stats/zstat1.nii.gz'
	loc_data = nib.load(loc_path).get_fdata().flatten()

	# restrict roi mask to all voxels above threshold in localizer contrast
	mask = np.array( (roi_data * loc_data) >= thr, dtype=int)
	n_voxels = np.count_nonzero(mask)

	"""
    # restrict roi mask to top n voxels above threshold in localizer contrast
    n_vox_target = 512  # number of voxels to select from each ROI
    thr_nvox = np.argsort(roi_data * loc_data)[n_vox_target]  # find lowest z value for top n_voxels
    thr_final = max(thr, thr_nvox)  # use top n_vox or all vox above threshold, whichever is fewer
    mask = np.array( (roi_data * loc_data) >= thr_final, dtype=int)
	n_voxels = np.count_nonzero(mask)
    """

	# collate voxelwise responses
	responses = np.empty((n_runs, fMRI.n_img, n_voxels))
	for r, run_dir in enumerate(run_dirs):
		for c, label in enumerate(fMRI.cond_labels):
			pe_path = f'{run_dir}/reg_standard/stats/cope{c+ 1}.nii.gz'
			pe_data = nib.load(pe_path).get_fdata().flatten()
			responses[r, c, :] = pe_data[mask == 1]

	return responses


def calculate_RDMs(responses, norm, norm_method):

	# get responses shape
	n_runs, n_conds, n_vox = responses.shape

	# split the data
	from sklearn.model_selection import train_test_split
	n_splits = 8
	RDMs_split = np.empty((n_splits, n_conds, n_conds))
	for split in range(n_splits):

		responses_split = np.empty((2, n_runs // 2, n_conds, n_vox))
		responses_split[0], responses_split[1] = train_test_split(responses, test_size=0.5, random_state=split)
		responses_split = np.mean(responses_split, axis=1) # average across runs

		mean_response = np.zeros_like(responses_split)
		std_response = np.ones_like(responses_split)

		if norm == 'all-conds':
			mean_response = np.tile(np.mean(responses_split, axis=1, keepdims=True), (1, n_conds, 1))
			std_response = np.tile(np.std(responses_split, axis=1, keepdims=True), (1, n_conds, 1))

		elif norm == 'occluder':
			for oc in range(3):
				mean_response[:, oc::3, :] = np.tile(np.mean(responses_split[:, oc::3, :], axis=1, keepdims=True),
													 (1, 8, 1))
				std_response[:, oc::3, :] = np.tile(np.std(responses_split[:, oc::3, :], axis=1, keepdims=True),
													(1, 8, 1))

		elif norm == 'unoccluded':
			mean_response = np.tile(np.mean(responses_split[:, ::3, :], axis=1, keepdims=True), (1, 24, 1))
			std_response = np.tile(np.std(responses_split[:, ::3, :], axis=1, keepdims=True), (1, 24, 1))

		# subtract mean
		responses_split -= mean_response

		# to get z-score, divide by std
		if norm_method == 'z-score':
			responses_split /= std_response

		# calculate RDM based on Euclidean distance, normalized by number of voxels
		from sklearn.metrics import euclidean_distances as dist
		RDMs_split[split] = np.sqrt(dist(responses_split[0], responses_split[1], squared=True) / n_vox)

	# mean across splits
	RDM = np.mean(RDMs_split, axis=0)

	return RDM



def main(overwrite_responses=False, overwrite_RDMs=False, overwrite_plots=False):
	fMRI.regions_of_interest.pop('V3')
	subjects = list(json.load(open('participants.json', 'r+')).keys())


	out_dir = f'derivatives/RSA'
	os.makedirs(out_dir, exist_ok=True)

	responses_path = f'{out_dir}/responses.pkl'
	if op.isfile(responses_path):
		responses = pkl.load(open(responses_path, 'rb'))
	else:
		responses = {region: {} for region in fMRI.regions_of_interest}

		# get responses for each subject
		for region in fMRI.regions_of_interest:
			for subject in subjects:
				if subject not in responses[region] or overwrite_responses:
					print(f'Getting responses | {region} | {subject} ')
					responses_subject = get_responses(subject, region)
					responses[region][subject] = responses_subject

				# make TSNE plots
				TSNE_dir = f'{out_dir}/TSNE'
				os.makedirs(TSNE_dir, exist_ok=True)
				outpath = f'{TSNE_dir}/{subject}_{region}.png'
				if not op.isfile(outpath) or overwrite_plots:
					print(f'Generating TSNE plots | region: {region} | subject {subject}')
					from in_vivo.fMRI.utils import TSNE_plot
					responses_TSNE = np.mean(responses_subject, axis=0)  # mean across runs
				TSNE_plot(responses_TSNE, outpath)

	# save responses
	pkl.dump(responses, open(responses_path, 'wb'))

	# calculate RDMs, plot RDMs, and plot MDS
	for norm in fMRI.norms:
		for norm_method in fMRI.norm_methods:

			analysis_dir = f'{out_dir}/norm-{norm}_{norm_method}'
			os.makedirs(analysis_dir, exist_ok=True)
			RDMs_path = f'{analysis_dir}/RDMs.pkl'
			if op.isfile(RDMs_path):
				RDMs = pkl.load(open(RDMs_path, 'rb'))
			else:
				RDMs = {}

			for region in fMRI.regions_of_interest:
				if region not in RDMs or overwrite_RDMs:
					RDMs[region] = np.empty((len(subjects) + 1, fMRI.n_img, fMRI.n_img))

					# calculate RDMs
					for s, subject in enumerate(subjects + ['group']):

						if subject != 'group':
							# individual subjects
							n_vox = responses[region][subject].shape[2]
							print(f'Calculating RDM | {region} | {subject} | norm-{norm} | {norm_method} | {n_vox} voxels')
							subject_responses = responses[region][subject]
							RDM = calculate_RDMs(subject_responses, norm, norm_method)
						else:
							# group mean
							print(f'Calculating RDM | {region} | {subject} | norm-{norm} | {norm_method}')
							RDM = np.mean(RDMs[region][:-1], axis=0)
						RDMs[region][s] = RDM

				# make plots
				for s, subject in enumerate(subjects + ['group']):

					RDM = RDMs[region][s]

					# plot RDMs
					from in_vivo.fMRI.utils import RDM_plot
					RDMs_dir = f'{analysis_dir}/RDMs'
					os.makedirs(RDMs_dir, exist_ok=True)
					outpath = f'{RDMs_dir}/{subject}_{region}.png'
					if not op.isfile(outpath) or overwrite_plots:
						print(f'Generating RDM plots | {region} | {subject} | norm-{norm} | {norm_method}')
						RDM_plot(RDM, vmin=None, vmax=None, fancy=False,
								 title=f'subject: {subject}, ROI: {region}',
								 labels=fMRI.cond_labels,
								 outpath=outpath, measure='Euclidean distance')

					# plot MDS
					from in_vivo.fMRI.utils import MDS_plot
					MDS_dir = f'{analysis_dir}/MDS'
					os.makedirs(MDS_dir, exist_ok=True)
					outpath = f'{MDS_dir}/{subject}_{region}.png'
					if not op.isfile(outpath) or overwrite_plots:
						print(f'Generating MDS plots | {region} | {subject} | norm-{norm} | {norm_method}')
						MDS_plot(RDM, title=f'subject: {subject}, ROI: {region}',
								 outpath=outpath)

			# save RDMs
			pkl.dump(RDMs, open(RDMs_path, 'wb'))


if __name__ == "__main__":
	#time.sleep(2 * 60 * 60)
	os.chdir(op.expanduser("~/david/projects/p022_occlusion/in_vivo/fMRI/exp1_orig"))
	start = time.time()
	overwrite_responses = False
	overwrite_RDMs = False
	overwrite_plots = True
	main(overwrite_responses, overwrite_RDMs, overwrite_plots)
	finish = time.time()
	print(f'analysis took {seconds_to_text(finish - start)} to complete')
	
