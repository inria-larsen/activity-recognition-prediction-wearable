import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import rc

nbr_components = 12

if __name__ == '__main__':
	"""
	This script plot the F1 score of models regarding number of features
	"""

	# path_wrapper = '/home/amalaise/Documents/These/experiments/ANDY_DATASET/AndyData-lab-onePerson/'
	path_score = 'score/'

	name_track = ['general_posture', 'detailed_posture', 'details', 'current_action']
	name_track = ['detailed_posture', 'current_action']
	# code_track = ['GePos', 'DePos', 'Det', 'CuAct']
	code_track = ['Posture', 'Action']
	# code_track = ['DePos']

	flag = 0
	frames = []

	fig = plt.figure(1, figsize=(10,6))
	# ax = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
	ax = [fig.add_subplot(211), fig.add_subplot(212)]
	plt.subplots_adjust(hspace = 0.5)



	# Plot the results for each track in the list
	for track, code in zip(name_track, code_track):
		print('#################################')
		print(track)
		print('#################################')

		score_pca = pd.read_csv(path_score + 'pca/score_pca_' + track + "2.csv")
		method_pca = []
		for i in range(len(score_pca)):
			method_pca.append('PCA')

		method_pca = pd.DataFrame({'method': method_pca})
		score_pca = pd.concat([score_pca, method_pca], axis = 1)

		score_df = score_pca

		file_fisher = path_score + 'filter/score_fisher_' + track + "3.csv"
		if(os.path.isfile(file_fisher)):
			score_tot = pd.read_csv(file_fisher)

			method_fisher = []
			for i in range(len(score_tot)):
				method_fisher.append('Fisher')
			method_fisher = pd.DataFrame({'method': method_fisher})
			score_tot = pd.concat([score_tot, method_fisher], axis = 1)

			score_df = pd.concat([score_tot, score_df])

		score_wrapper = pd.read_csv(path_score + 'wrapper/score_model_wrapper_' + track + ".csv")
		method_wrapper = []
		for i in range(len(score_wrapper)):
			method_wrapper.append('Wrapper')

		method_wrapper = pd.DataFrame({'method': method_wrapper})
		score_wrapper = pd.concat([score_wrapper, method_wrapper], axis = 1)

		# score_df = pd.concat([score_wrapper, score_df])
		score_df = score_wrapper




		nbr_feature_max = np.max(score_df['nbr_components'])

		taxonomy = [track] * len(score_df)
		score_df['taxonomy'] = pd.Series(taxonomy)
		frames.append(score_df)

		df_total = pd.concat(frames)

		percentage_max = 2

		score_max = score_wrapper.groupby(['method', 'nbr_components'], axis=0).mean()
		value_score = score_max.loc[score_max['score'].idxmax()]*(100-percentage_max)
		value_score = value_score.values[0]/100
		id_02 = (score_max['score'].loc[score_max['score']>=value_score].index)[0][1] - 1
		print(score_max.loc[score_max['score'].idxmax()], '\n', score_max.loc[score_max['score']>=value_score])
		print(id_02)


		
		# print(score_max)
		# print(score_max.loc[score_max['score'] >= score_max.loc[score_max['score'].idxmax()]*(100-percentage_max)])


		# score_max = score_pca.groupby(['method', 'nbr_components'], axis=0).mean()
		# print(score_max.loc[score_max['score'].idxmax()])
		# score_max = score_tot.groupby(['method', 'nbr_components'], axis=0).mean()
		# print(score_max.loc[score_max['score'].idxmax()])

		

		a = sns.relplot(x = 'nbr_components', y='score', kind="line", hue = 'method', data=score_df, ax=ax[flag], legend = 'full')
		# a = sns.relplot(x = 'features', y='score', kind="line", hue = 'method', ci="sd", data=score_df, ax=ax[flag], legend = 'full')
		plt.close(a.fig)
		ax[flag].axhline(y = value_score, color='r', linewidth=0.8, linestyle='--')
		ax[flag].axvline(x = id_02, color='r', linewidth=0.8, linestyle='--')
		ax[flag].set_xlabel('Number of features', fontsize = 'x-large')
		ax[flag].set_title(code, fontsize = 'xx-large', fontdict = {'variant': 'small-caps'})
		ax[flag].set_ylabel('F1-Score', fontsize = 'x-large')
		ax[flag].set_xticks(np.arange(1, nbr_feature_max+1, 2))
		ax[flag].set_xticklabels(np.arange(1, nbr_feature_max+2, 2), fontsize = 'x-large')
		ax[flag].set_yticks(np.arange(0.3, 1.05, 0.1))
		ax[flag].set_yticks(np.arange(0.3, 1.05, 0.1))
		zed = [tick.label.set_fontsize(12) for tick in ax[flag].yaxis.get_major_ticks()]
		flag += 1

	fig = plt.figure(1)
	fig.tight_layout()

	# fig.savefig("/home/amalaise/Documents/These/papers/adrien_ra-l/img/output2.pdf", bbox_inches='tight')
	fig.savefig("/home/amalaise/Documents/These/papers/videos/activity_reco/results_selection.pdf", bbox_inches='tight')

	plt.show()
