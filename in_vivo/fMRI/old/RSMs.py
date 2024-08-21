#!/usr/bin/python
"""
script for making and plotting RSMs
"""

import os
import os.path as op
import sys
import glob
import numpy as np
import pickle
import nibabel as nib
import json
import time

sys.path.append(op.expanduser('~/david/masterScripts/misc'))
from seconds_to_text import seconds_to_text
from in_vivo.fMRI.utils import RSM_plot, fMRI, layers_of_interest, MDS_plot


subjects = json.load(open('participants.json', 'r+'))
thr = 3.1  # z threshold for localiser data
n_conds = fMRI.n_img
masks = layers_of_interest

def make_RSMs():

	for localiser in fMRI.localisers:
		for norm in fMRI.norms_exp2:
			RSMs = {}
			for mask, mask_label in zip(masks, fMRI.masks):

				RSMs[mask] = {}

				for subject in subjects:

					subj_dir = f'derivatives/FEAT/sub-{subject}'
					n_splits = len(glob.glob(f'{subj_dir}/task-occlusionAttnOn_split-*A.gfeat'))
					RSMs[mask][subject] = np.empty((n_splits, n_conds*2, n_conds*2))

					# load mask and restrict to active voxels in localiser contrast
					mask_path_orig = glob.glob(f'{fMRI.mask_dir_std}/*/{mask_label}.nii.gz')[0]
					mask_path = f'derivatives/masks/MNI152_2mm/{mask_label}.nii.gz'
					if not op.isfile(mask_path):
						os.makedirs(op.dirname(mask_path), exist_ok=True)
						os.system(f'ln -s {mask_path_orig} {mask_path}')
					mask_data = nib.load(mask_path).get_fdata().flatten()
					
					# all conditions in localiser scan
					if localiser == 'localiser':
						loc_path = f'{subj_dir}/task-objectLocaliser_runs-all.gfeat/cope12.feat/stats/zstat1.nii.gz'

					# all conditions in occlusion scan (with attention on)
					elif localiser == 'all_conds':
						loc_path = f'{subj_dir}/task-occlusionAttnOn_runs-all.gfeat/cope25.feat/stats/zstat1.nii.gz'

					loc_data = nib.load(loc_path).get_fdata().flatten()
					loc_data = np.array(loc_data >= thr, dtype=int)
					loc_mask_data = loc_data * mask_data
					n_voxels = np.count_nonzero(loc_mask_data)

					print(f'subject: {subject} | mask: {mask} | n_voxels: {n_voxels} | localiser: {localiser} | normalisation: {norm}')


					# experimental data
					for s in range(n_splits):
						responses = np.empty((2, n_conds*2, n_voxels))
						for si, side in enumerate(['A', 'B']):
							for a, attn in enumerate(fMRI.attns_path):
								for c in range(n_conds):
									cope_path = f'{subj_dir}/task-occlusion{attn}_split-{s}{side}.gfeat/cope{c+1}.feat/stats/cope1.nii.gz'
									cope_data = nib.load(cope_path).get_fdata().flatten()
									responses[si, (a*n_conds) + c, :] = cope_data[loc_mask_data > 0]

						# normalisation
						mean_response = np.zeros_like(responses)

						if norm == 'all_conds':
							mean_response = np.tile(np.mean(responses, axis=1, keepdims=True), (1, n_conds*2, 1))

						elif norm == 'attention':
							for attn in range(2):
								idxs = np.arange(n_conds) + (attn*n_conds)
								mean_response[:, idxs, :] = np.tile(np.mean(responses[:, idxs, :], axis=1, keepdims=True), (1, n_conds, 1))

						elif norm == 'occluder-attention':
							for attn in range(2):
								for oc in range(3):
									idxs = np.arange((attn*n_conds) + oc, ((attn+1)*n_conds), 3)
									mean_response[:, idxs, :] = np.tile(np.mean(responses[:, idxs, :], axis=1, keepdims=True), (1, 8, 1))

						responses_norm = responses - mean_response

						# get correlations
						RSM = np.corrcoef(responses_norm[0, :, :], responses_norm[1, :, :])[n_conds*2:,:n_conds*2]
						RSM_diag_mean = RSM + RSM.transpose() / 2
						RSMs[mask][subject][s, :, :] = RSM_diag_mean
						RSMs[mask][subject][s,:,:] = np.corrcoef(responses_norm[0, :, :], responses_norm[1, :, :])[n_conds*2:,:n_conds*2]

			out_dir = f'derivatives/RSA/within-subjects/loc-{localiser}/norm-{norm}'
			os.makedirs(out_dir, exist_ok=True)
			pickle.dump(RSMs, open(f'{out_dir}/RSMs.pkl', 'wb'))


def plot_RSMs():

	for localiser in fMRI.localisers:
		for norm in fMRI.norms_exp2:

			analysis_dir = f'derivatives/RSA/within-subjects/loc-{localiser}/norm-{norm}'
			RSMs = pickle.load(open(f'{analysis_dir}/RSMs.pkl', 'rb'))
			RSMs_dir = f'{analysis_dir}/RSMs'
			os.makedirs(RSMs_dir, exist_ok=True)

			for mask in masks:

				all_mats = np.empty((len(subjects), n_conds*2, n_conds*2))

				# individual plots
				for s, subject in enumerate(subjects):

					matrix = np.mean(RSMs[mask][subject], axis=0) # mean across splits

					# RSM of all conditions
					RSM_plot(matrix, vmin=-.8, vmax=.8, title=f'subject: {subject}, ROI: {mask}', labels=fMRI.cond_labels_attn,
							 outpath=f'{RSMs_dir}/{subject}_{mask}.pdf')

					# attention-wise RSMs
					for a, attn in enumerate(fMRI.attns):
						idxs = np.arange(a * n_conds, (a + 1) * n_conds)
						matrix_attn = matrix[idxs, :][:, idxs]
						RSM_plot(matrix_attn, vmin=-.8, vmax=.8, title=f'subject: {subject}, ROI: {mask}, {attn}', labels=fMRI.cond_labels,
								 outpath=f'{RSMs_dir}/{subject}_{mask}_{attn}.pdf')
					all_mats[s, :, :] = matrix

				# group plots
				matrix = np.mean(all_mats, axis=0)

				# RSM of all conditions
				RSM_plot(matrix, vmin=-.3, vmax=.3, labels=fMRI.cond_labels_attn,
						 title=f'subject: group, ROI: {mask}',
						 outpath=f'{RSMs_dir}/group_{mask}.pdf')

				# attention-wise RSMs
				for a, attn in enumerate(fMRI.attns):
					idxs = np.arange(a * n_conds, (a + 1) * n_conds)
					matrix_attn = matrix[idxs, :][:, idxs]
					RSM_plot(matrix_attn, vmin=-.3, vmax=.3, title=f'subject: group, ROI: {mask}, {attn}', labels=fMRI.cond_labels,
							 outpath=f'{RSMs_dir}/group_{mask}_{attn}.pdf')

					# MDS plots
					MDS_dir = f'{analysis_dir}/MDS'
					os.makedirs(MDS_dir, exist_ok=True)
					MDS_plot(matrix, title=f'subject: group, ROI: {mask}',
							 outpath=f'{MDS_dir}/group_{mask}_{attn}.pdf')


if __name__ == "__main__":

	#time.sleep(2 * 60 * 60)
	os.chdir(op.expanduser("/in_vivo/fMRI/exp2"))
	start = time.time()
	make_RSMs()
	plot_RSMs()
	finish = time.time()
	print(f'analysis took {seconds_to_text(finish - start)} to complete')
