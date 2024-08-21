import sys
import os
import os.path as op
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import json

import time
#time.sleep(3600)

sys.path.append(op.expanduser('~/david/masterScripts/misc'))
from seconds_to_text import seconds_to_text
from plot_utils import export_legend, custom_defaults
plt.rcParams.update(custom_defaults)

from in_vivo.fMRI.utils import fMRI, regression_models, layers_of_interest

subjects = json.load(open('participants.json', 'r+'))
cross_val = 'within-subjects'
regression_model_labels = list(regression_models.keys())
masks = layers_of_interest
cond_labels = fMRI.cond_labels

def RSA():

	for localiser in fMRI.localisers:
		for norm in fMRI.norms_exp2:

			print(f'localiser: {localiser} | normalisation: {norm}')

			analysis_dir = f'derivatives/RSA/{cross_val}/loc-{localiser}/norm-{norm}'
			RSMs = pkl.load(open(f'{analysis_dir}/RSMs.pkl', 'rb'))

			table_dict = {'subject': [], 'region': [], 'attention': [], 'contrast': [], 'level': [], 'mean': [], 'sem': []}
			for a, attn in enumerate(fMRI.attns):
				for mask in masks:
					for s, subject in enumerate(subjects):

						# get RSM and apply z-transform
						idxs = np.arange(a * fMRI.n_img, (a + 1) * fMRI.n_img)
						matrix_z = np.arctanh(np.mean(RSMs[mask][subject][:, idxs, :][:,:,idxs], axis=0))

						# 2x2 exemplar by occluder position (occluded images only)
						EsOs = []  # occluded images only, same exemplar, same occluder
						EsOd = []  # occluded images only, same exemplar, different occluder
						EdOs = []  # occluded images only, different exemplar, same occluder
						EdOd = []  # occluded images only, different exemplar, different occluder

						# 2x2, exemplar by one occluded yes/no
						EsUb = []  # unoccluded images only, same exemplar
						EdUb = []  # unoccluded images only, different exemplar
						EsU1 = []  # occluded v unoccluded, same exemplar
						EdU1 = []  # occluded v unoccluded, different exemplar

						for c_a, cond_label_a in enumerate(cond_labels):

							exemplar_a = cond_label_a.split('_')[0]
							occluder_a = cond_label_a.split('_')[1]

							for c_b, cond_label_b in enumerate(cond_labels):

								exemplar_b = cond_label_b.split('_')[0]
								occluder_b = cond_label_b.split('_')[1]

								z = matrix_z[c_a, c_b]

								# 2x2 exemplar by occluder position (occluded images only)
								if occluder_a != 'none' and occluder_b != 'none':
									if exemplar_a == exemplar_b and occluder_a == occluder_b:
										EsOs.append(z)
									elif exemplar_a != exemplar_b and occluder_a == occluder_b:
										EdOs.append(z)
									elif exemplar_a == exemplar_b and occluder_a != occluder_b:
										EsOd.append(z)
									elif exemplar_a != exemplar_b and occluder_a != occluder_b:
										EdOd.append(z)

								# 2x2, exemplar by one occluded yes/no
								elif occluder_a == 'none' and occluder_b == 'none':  # neither occluded
									if exemplar_a == exemplar_b:
										EsUb.append(z)
									else:
										EdUb.append(z)
								elif int(occluder_a == 'none') + int(occluder_b == 'none') == 1:  # one occluded
									if exemplar_a == exemplar_b:
										EsU1.append(z)
									else:
										EdU1.append(z)

						# 2x2 exemplar by occluder position (occluded images only)
						## same exemplar, same occluder
						table_dict['subject'].append(subject)
						table_dict['region'].append(mask)
						table_dict['attention'].append(attn)
						table_dict['contrast'].append('object_completion')
						table_dict['level'].append('EsOs')
						table_dict['mean'].append(np.mean(EsOs))
						table_dict['sem'].append(stats.sem(EsOs))

						## same exemplar, different occluder
						table_dict['subject'].append(subject)
						table_dict['region'].append(mask)
						table_dict['attention'].append(attn)
						table_dict['contrast'].append('object_completion')
						table_dict['level'].append('EsOd')
						table_dict['mean'].append(np.mean(EsOd))
						table_dict['sem'].append(stats.sem(EsOd))

						## different exemplar, same occluder
						table_dict['subject'].append(subject)
						table_dict['region'].append(mask)
						table_dict['attention'].append(attn)
						table_dict['contrast'].append('object_completion')
						table_dict['level'].append('EdOs')
						table_dict['mean'].append(np.mean(EdOs))
						table_dict['sem'].append(stats.sem(EdOs))

						## different exemplar, different occluder
						table_dict['subject'].append(subject)
						table_dict['region'].append(mask)
						table_dict['attention'].append(attn)
						table_dict['contrast'].append('object_completion')
						table_dict['level'].append('EdOd')
						table_dict['mean'].append(np.mean(EdOd))
						table_dict['sem'].append(stats.sem(EdOd))

						# 2x2, exemplar by one occluded yes/no
						## same exemplar, both unoccluded
						table_dict['subject'].append(subject)
						table_dict['region'].append(mask)
						table_dict['attention'].append(attn)
						table_dict['contrast'].append('occlusion_invariance')
						table_dict['level'].append('EsUb')
						table_dict['mean'].append(np.mean(EsUb))
						table_dict['sem'].append(stats.sem(EsUb))

						## same exemplar, one occluded
						table_dict['subject'].append(subject)
						table_dict['region'].append(mask)
						table_dict['attention'].append(attn)
						table_dict['contrast'].append('occlusion_invariance')
						table_dict['level'].append('EsU1')
						table_dict['mean'].append(np.mean(EsU1))
						table_dict['sem'].append(stats.sem(EsU1))

						## different exemplar, one unoccluded
						table_dict['subject'].append(subject)
						table_dict['region'].append(mask)
						table_dict['attention'].append(attn)
						table_dict['contrast'].append('occlusion_invariance')
						table_dict['level'].append('EdUb')
						table_dict['mean'].append(np.mean(EdUb))
						table_dict['sem'].append(stats.sem(EdUb))

						## different exemplar, one unoccluded
						table_dict['subject'].append(subject)
						table_dict['region'].append(mask)
						table_dict['attention'].append(attn)
						table_dict['contrast'].append('occlusion_invariance')
						table_dict['level'].append('EdU1')
						table_dict['mean'].append(np.mean(EdU1))
						table_dict['sem'].append(stats.sem(EdU1))

						## regression
						for regression_model in regression_models:
							model_flat = regression_models[regression_model].flatten()
							model_flat_off_diag = model_flat[
								np.isfinite(model_flat)]  # remove where model flat == 0

							matrix_flat = matrix_z.flatten()
							matrix_flat_off_diag = matrix_flat[
								np.isfinite(model_flat)]  # remove where model flat == 0

							regr = LinearRegression()
							fit = \
							regr.fit(model_flat_off_diag.reshape(-1, 1), matrix_flat_off_diag.reshape(-1, 1)).coef_[
								0][0]

							## different exemplar, one unoccluded
							table_dict['subject'].append(subject)
							table_dict['region'].append(mask)
							table_dict['attention'].append(attn)
							table_dict['contrast'].append('regression')
							table_dict['level'].append(regression_model)
							table_dict['mean'].append(fit)
							table_dict['sem'].append(np.nan)

			table_all = pd.DataFrame(table_dict)
			table_all.to_csv(f'{analysis_dir}/contrasts.csv')


def RSA_plots():

	fig_size_ind = (8, 4)
	fig_size_grp = (7, 4)
	colours = list(mcolors.TABLEAU_COLORS)
	all_colours = np.array([0, 1, 2, 3, 4, 6, 8, 9, 0, 1, 2, 3]).reshape((3, 4)) # colour idxs for completion, invariance, regression conditions

	legend_labels = {
		'object_completion': [
			'exemplar same, occluder same',
			'exemplar same, occluder different',
			'exemplar different, occluder same',
			'exemplar different, occluder different'],
		'occlusion_invariance': [
			'exemplar same, both unoccluded',
			'exemplar same, one unoccluded',
			'exemplar different, both unoccluded',
			'exemplar different, one unoccluded'],
		'regression': regression_model_labels}

	for localiser in fMRI.localisers:
		for norm in fMRI.norms_exp2:

			print(f'Plotting | localiser: {localiser} | normalisation: {norm}')

			analysis_dir = f'derivatives/RSA/{cross_val}/loc-{localiser}/norm-{norm}'
			table = pd.read_csv(open(f'{analysis_dir}/contrasts.csv', 'r+'))

			if norm == 'none':
				ylims = (0, 1)
			else:
				ylims = (-.1, .3)


			for a, attn in enumerate(fMRI.attns):


				for c, contrast in enumerate(['object_completion', 'occlusion_invariance', 'regression']):

					out_dir = f'{analysis_dir}/{contrast}'
					os.makedirs(out_dir, exist_ok=True)

					ylabel = ['correlation (Z)', 'correlation (Z)', r"regression coefficient ($\beta$)"][c]
					cols = [colours[col] for col in all_colours[c,:]]
					levels = [['EsOs', 'EsOd', 'EdOs', 'EdOd'], ['EsUb', 'EsU1', 'EdUb', 'EdU1'], regression_model_labels][c]

					# legend
					labels = legend_labels[contrast]
					f = lambda m, c: plt.plot([], [], marker=m, markerfacecolor=c, color='white')[0]
					handles = [f('s', cols[c]) for c in range(4)]
					legend = plt.legend(handles, labels, loc=3)
					export_legend(legend, filename=f'{out_dir}/legend.pdf')
					plt.close()

					# collate plots across ROIs
					all_region_vals = np.empty(shape=(len(masks), len(subjects), 4))  # for plotting all ROIs at once

					# individual level plots for each ROI
					for m, mask in enumerate(masks):

						df = table.loc[
								(table['region'] == mask) &
								(table['attention'] == attn) &
								(table['contrast'] == contrast), :].copy()

						df['subject'] = df['subject'].astype('category').cat.reorder_categories(subjects)
						df['level'] = df['level'].astype('category').cat.reorder_categories(levels)

						# record data for cross ROI plots
						all_region_vals[m, :, :] = np.array(df['mean']).reshape((len(subjects), 4))

						# plot response magnitudes
						df_means = df.pivot(index='subject', columns='level', values='mean')
						df_sems = df.pivot(index='subject', columns='level', values='sem').values
						df_plot = df_means.plot(kind='bar',
												yerr=df_sems.transpose(),
												ylabel=ylabel,
												rot=0,
												figsize=fig_size_ind,
												color=cols,
												legend=False)
						fig = df_plot.get_figure()
						plt.tick_params(direction='in')
						plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
						plt.ylim(ylims)
						plt.title(f'region: {mask}, {attn}')
						plt.tight_layout()
						fig.savefig(f'{out_dir}/{mask}_{attn}_individual.pdf')
						plt.show()

					# make plots across regions
					means_all = np.mean(all_region_vals[:, :, :], axis=1)
					sems_all = stats.sem(all_region_vals[:, :, :], axis=1)

					plt.figure(figsize=fig_size_grp)
					colours = list(mcolors.TABLEAU_COLORS.keys())
					for l, level in enumerate(levels):
						means = means_all[:, l]
						sems = sems_all[:, l]
						bar_width = 1 / 5
						xshift = (l * bar_width) - (2 * bar_width)
						plt.bar(np.arange(len(masks)) + xshift, means, yerr=sems, width=bar_width, color=[colours[all_colours[c, l]]], label=level)
					plt.xticks(np.arange(len(masks)), labels=masks)  # , rotation=25, ha='right')
					plt.ylabel(ylabel)
					plt.yticks(np.arange(0, 0.5, .1))
					plt.ylim(ylims)
					plt.tight_layout()
					plt.savefig(os.path.join(out_dir, f'all_regions_group_{attn}.pdf'))
					plt.show()

					# object completion / occlusion invariance index
					if contrast in ['object_completion', 'occlusion_invariance']:
						plot_means = []
						for m in range(len(masks)):
							vals = means_all[m,:]

							# formula for object completion and occlusion invariance is the same, index-wise
							# occ_inv_idx = (EsU1 - EdU1) / (EsUb + EdU1)
							# obj_comp_idx = (EsOd - EdOd) / (EsOs - EdOd)
							plot_means.append((vals[1]-vals[3])/(vals[0]-vals[3]))

						obj_comp_df = pd.DataFrame({'region': masks, 'mean': plot_means})
						obj_comp_df.plot(kind='line',
										 rot=0,
										 legend=False,
										 figsize=(fig_size_grp),
										 marker='o',
										 markerfacecolor='white')
						plt.fill_between(np.arange(-.5, 4.5), 1, 2, color='black', alpha=.2, lw=0)
						plt.xticks(np.arange(len(masks)), labels=masks)
						plt.yticks((0, .5, 1))
						#  plt.xlabel('cortical region', size=12)
						if contrast == 'object_completion':
							plt.ylabel('OCI')
							#  plt.title('object completion')
						elif contrast == 'occlusion_invariance':
							plt.ylabel('OII')
							#  plt.title('occlusion invariance')

						plt.ylim((0, 1.25))
						plt.xlim((-.5, 3.5))
						plt.tight_layout()
						plt.savefig(f'{out_dir}/{contrast}_index_{attn}.pdf', dpi=300)
						plt.show()




if __name__ == "__main__":

	os.chdir(op.expanduser("/in_vivo/fMRI/exp2"))
	#time.sleep(6 * 60 * 60)
	start = time.time()
	#RSA()
	RSA_plots()
	finish = time.time()
	print(f'analysis took {seconds_to_text(finish - start)} to complete')