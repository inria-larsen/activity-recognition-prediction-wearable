import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
	"""
	This script plot the F1 score of models regarding number of features
	"""

	name_track = ['general_posture', 'detailed_posture', 'details', 'current_action']

	flag = 0
	frames = []

	fig = plt.figure(1, figsize=(10,6))
	ax = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
	plt.subplots_adjust(hspace = 0.5)

	# Plot the results for each track in the list
	for track in name_track:
		if(track == 'details'):
			score_df = pd.read_csv('score_dimensions_' + track + "2.csv")
		else:
			score_df = pd.read_csv('score_dimensions_' + track + ".csv")
		nbr_feature_max = np.max(score_df['features'])

		taxonomy = [track] * len(score_df)
		score_df['taxonomy'] = pd.Series(taxonomy)
		frames.append(score_df)

		df_total = pd.concat(frames)

		a = sns.relplot(x = 'features', y='score', kind="line", hue = 'method', ci="sd", data=score_df, ax=ax[flag], legend = 'full')
		plt.close(a.fig)
		ax[flag].set_xlabel('Number of features', fontsize = 'x-large')
		ax[flag].set_title(track, fontsize = 'xx-large')
		ax[flag].set_ylabel('F1-Score', fontsize = 'x-large')
		ax[flag].set_xticks(np.arange(0, nbr_feature_max+1, 2))
		ax[flag].set_xticklabels(np.arange(1, nbr_feature_max+2, 2), fontsize = 'x-large')
		ax[flag].set_yticks(np.arange(0.5, 1.05, 0.1))
		ax[flag].set_yticks(np.arange(0.5, 1.05, 0.1))
		zed = [tick.label.set_fontsize(12) for tick in ax[flag].yaxis.get_major_ticks()]
		flag += 1

	fig = plt.figure(1)
	fig.tight_layout()

	fig.savefig("/home/amalaise/Documents/These/papers/adrien_ra-l/img/output.pdf", bbox_inches='tight')

	plt.show()
