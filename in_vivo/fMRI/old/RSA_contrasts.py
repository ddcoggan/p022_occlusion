import os
import os.path as op
import numpy as np
import pickle as pkl
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import json

import time
#time.sleep(3600)

from utils.seconds_to_text import seconds_to_text
from utils.plot_utils import export_legend, custom_defaults
plt.rcParams.update(custom_defaults)

from in_vivo.fMRI.utils import fMRI, figsizes
fMRI.regions.pop(2)

def RSA():

	subjects = json.load(open('participants.json', 'r+'))


	print(f'Performing RSA')
	

	for norm in fMRI.norms:
		for norm_method in fMRI.norm_methods:
			for dis

			analysis_dir = f'derivatives/RSA/norm-{norm}_{norm_method}'
			RDMs = pkl.load(open(f'{analysis_dir}/RDMs.pkl', 'rb'))
			contrasts_path = f'{analysis_dir}/contrasts.csv'
			contrasts = pd.DataFrame()

			for region in fMRI.regions:

				RDMs_table = pd.DataFrame()
				regression_table = pd.DataFrame()
				for s, subject in enumerate(subjects):

					RDM = RDMs[region][s]

					# convert RDM to table with conditions
					from in_vivo.fMRI.utils import RDM_to_table
					RDM_table = RDM_to_table(RDM)
					RDM_table['subject'] = subject
					RDMs_table = pd.concat([RDMs_table, RDM_table]).reset_index(drop=True)

					# perform regression analyses at individual level
					from in_vivo.fMRI.utils import RDM_regression
					regression_table['subject'] = subject
					regression_table = pd.concat([regression_table, RDM_regression(RDM_table)]).reset_index(drop=True)

				# append individual regression stats to contrasts
				regression_table_ind = regression_table.drop(columns=['subject', 'analysis', 'unit', 'error_unit', 'error']).groupby('level').agg(['mean', 'sem']).reset_index()
				regression_table_ind.columns = ['level', 'value', 'error']
				regression_table_ind['analysis'] = 'regression'
				regression_table_ind['subtype'] = 'ind'
				regression_table_ind['unit'] = 'beta'
				regression_table_ind['error_unit'] = 'sem'
				regression_table_ind['region'] = region
				contrasts = pd.concat([contrasts, regression_table_ind]).reset_index(drop=True)

				# perform regression analysis at the group level
				from in_vivo.fMRI.utils import RDM_regression
				RDMs_table_group_mean = RDM_to_table(RDMs[region][-1])
				regression_table = RDM_regression(RDMs_table_group_mean)
				regression_table['analysis'] = 'regression'
				regression_table['subtype'] = 'grp'
				regression_table['region'] = region
				contrasts = pd.concat([contrasts, regression_table]).reset_index(drop=True)

				# get condition-wise means and errors across subjects
				RDMs_table_cond_mean = RDMs_table.drop(columns=['exemplar_a', 'exemplar_b', 'occluder_a', 'occluder_b']).groupby(['analysis','level', 'subject']).agg('mean').reset_index()
				condition_summary = RDMs_table_cond_mean.drop(columns=['subject']).groupby(['analysis','level']).agg(['mean','sem']).dropna().reset_index()
				condition_summary.columns = ['analysis', 'level', 'value', 'error']
				condition_summary['unit'] = 'Euclidean distance'
				condition_summary['subtype'] = 'condwise'
				condition_summary['error_unit'] = 'sem'
				condition_summary['region'] = region
				contrasts = pd.concat([contrasts, condition_summary]).reset_index(drop=True)

				# calculate occlusion robustness
				from in_vivo.fMRI.utils import calculate_occlusion_robustness
				for sub_analysis, groupbys in zip(['index_ind', 'index_grp', 'index_exem'], [['subject', 'level'], ['level'],
																		   ['analysis', 'level',
																			'exemplar_b']]):
					df = RDMs_table.groupby(groupbys).agg('mean').reset_index()
					robustness_table = calculate_occlusion_robustness(df)
					robustness_table['subtype'] = sub_analysis
					robustness_table['region'] = region
					contrasts = pd.concat([contrasts, robustness_table]).reset_index(drop=True)

				# calculate noise ceiling
				for bound in ['lower', 'upper']:
					vals = []
					for s, subject in enumerate(subjects):

						RDM_ind = RDMs_table[RDMs_table.subject == subject].reset_index(drop=True)
						RDM_ind = RDM_ind.SIMILARITY.values
						RDM_ind_flat_offdiag = RDM_ind.flatten()[fMRI.off_diag_mask_flat]

						if bound == 'lower':
							RDM_grp = RDMs_table[RDMs_table.subject != subject].drop(columns='subject').groupby(
								RDMs_table.columns[:6].tolist()).agg('mean').dropna().reset_index()
						else:
							RDM_grp = RDMs_table_group_mean
						RDM_grp = RDM_grp.SIMILARITY.values
						RDM_grp_flat_offdiag = RDM_grp.flatten()[fMRI.off_diag_mask_flat]

						nc = np.corrcoef(RDM_ind_flat_offdiag, RDM_grp_flat_offdiag)[0, 1]
						vals.append(nc)

					contrasts = pd.concat([contrasts,
										 pd.DataFrame({
											 'region': [region],
											 'analysis': ['RSA'],
											 'subtype': ['noise_ceiling'],
											 'level': [bound],
											 'unit': ['r'],
											 'value': [np.mean(vals)],
											 'error_unit': [None],
											 'error': [None]}),
										 ]).reset_index(drop=True)

				contrasts.to_csv(contrasts_path)


def RSA_plots():

	print(f'Plotting contrasts')

	for norm in fMRI.norms:
		for norm_method in fMRI.norm_methods:
			analysis_dir = f'derivatives/RSA/norm-{norm}_{norm_method}'
			contrasts_path = f'{analysis_dir}/contrasts.csv'
			contrasts = pd.read_csv(open(contrasts_path, 'r+'), index_col=0)
			analyses = [analysis for analysis in contrasts.analysis.unique() if analysis != 'RSA']
			for analysis in analyses:

				out_dir = f'{analysis_dir}/{analysis}'
				shutil.rmtree(out_dir, ignore_errors=True)
				os.makedirs(out_dir, exist_ok=True)

				df_analysis = contrasts[(contrasts.analysis == analysis)].reset_index(drop=True)
				params = fMRI.occlusion_robustness_analyses[analysis]
				cond_labels = params['cond_labels']

				for subtype in df_analysis.subtype.unique():

					df = df_analysis[df_analysis.subtype == subtype].reset_index(drop=True)
					df['region'] = df['region'].astype('category').cat.reorder_categories(fMRI.regions)

					conds = ['raw','norm'] if subtype in ['index_ind', 'index_exem', 'index_grp'] and analysis != 'regression' else params['conds']
					df['level'] = df['level'].astype('category').cat.reorder_categories(conds)

					if analysis == 'regression' or subtype == 'condwise':

						# make clustered bar plots
						ylabel = r"regression coefficient ($\beta$)" if analysis == 'regression' else 'Euclidean distance'
						df_means = df.pivot(index='region', columns='level', values='value')
						df_sems = df.pivot(index='region', columns='level', values='error').values
						df_plot = df_means.plot(kind='bar',
												ylabel=ylabel,
												yerr=df_sems.transpose(),
												rot=0,
												figsize=figsizes['clustered_bar'],
												color=params['colours'],
												legend=False)
						fig = df_plot.get_figure()
						plt.tick_params(direction='in')
						#plt.ylim(ylims)
						plt.tight_layout()
						fig.savefig(f'{out_dir}/{subtype}.png')
						plt.show()

						# legend
						f = lambda m, c: plt.plot([], [], marker=m, markerfacecolor=c, color='white')[0]
						handles = [f('s', colour) for colour in params['colours']]
						legend = plt.legend(handles, cond_labels, loc=3)
						export_legend(legend, filename=f'{out_dir}/legend.png')
						plt.close()

					else:

						# line plots
						for level in df.level.unique():
							df_level = df[df.level == level].reset_index(drop=True)
							ylabel = params['index_label']
							fig, ax = plt.subplots(figsize=figsizes['single_4'])
							if not df_level.error.isnull().all():
								ax.errorbar(range(len(fMRI.regions)),
											df_level['value'].values,
											yerr=df_level['error'].values,
											color='tab:grey',
											linestyle='none',
											capsize=5)
							ax.plot(range(len(fMRI.regions)),
									df_level['value'].values,
									color='tab:grey',
									marker='o',
									markerfacecolor='white')
							ax.fill_between(np.arange(-.5, 4.5), 1, 2, color='black', alpha=.2, lw=0)
							ax.set_xticks(np.arange(len(fMRI.regions)), labels=fMRI.regions)
							ax.set_yticks((0, .5, 1))
							ax.set_ylabel(ylabel)
							ax.set_ylim((0, 1.1))
							ax.set_xlim((-.5, 3.5))
							plt.tight_layout()
							plt.savefig(f'{out_dir}/{ylabel}_{level}_{subtype}.png', dpi=300)
							plt.show()


					"""
					# plot (subject * level for each region)
					for region in fMRI.regions:
						df = df_analysis[(df_analysis.region == region)].copy()
						df['level'] = df['level'].astype('category').cat.reorder_categories(params['conds'])
						df_means = df.pivot(index='subject', columns='level', values='similarity')
						df_plot = df_means.plot(kind='bar',
												ylabel=ylabel,
												rot=0,
												figsize=figsizes['fMRI_exp2_ind'],
												color=params['colours'],
												legend=False)
						fig = df_plot.get_figure()
						plt.tick_params(direction='in')
						plt.xticks(rotation=25, ha="right", rotation_mode="anchor")
						plt.ylim(ylims)
						plt.title(f'region: {region}')
						plt.tight_layout()
						fig.savefig(f'{out_dir}/{region}_individual.png')
						plt.show()
					"""

if __name__ == "__main__":

	os.chdir(op.expanduser("~/david/projects/p022_occlusion/in_vivo/fMRI/exp1_orig"))
	#time.sleep(5 * 60 * 60)
	start = time.time()
	RSA()
	RSA_plots()
	finish = time.time()
	print(f'analysis took {seconds_to_text(finish - start)} to complete')
