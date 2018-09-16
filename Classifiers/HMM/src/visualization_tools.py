import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import cv2


def draw_distribution(score, list_states, real_labels):
	labels = list_states.tolist()
	for i in range(len(list_states)):
		labels[i] = labels[i].title()
		labels[i] = labels[i].replace("_", " ")

	clrs = np.zeros((len(labels), 3))
	sns.set(font_scale=1.5)
	id_pred = np.argmax(score)
	id_real = list_states.tolist().index(real_labels)
	for x in range(len(labels)):
		if((id_pred == id_real) and (x == id_real)):
			clrs[x] = [0,1,0]
		elif(x == id_real):
			clrs[x] = [0,0,1]
		else:
			clrs[x] = [1,0,0]
	ax = sns.barplot(score, labels, palette=clrs)
	ax.set_xlim(0,1)
	plt.title('Probability distribution')
	ax.title.set_fontsize(20)
	plt.ylabel('States')
	plt.xlabel('Probabilities')
	plt.subplots_adjust(left=0.2)
	return sns.barplot(score, labels, palette=clrs)

def video_distribution(score_samples, list_states, real_labels, fps, path, name_file):
	fig=plt.figure()
	ax = fig.add_subplot(1,1,1)
	n_frame = np.shape(real_labels)[0]

	ax = draw_distribution(score_samples[0], list_states, real_labels[0])

	def animate(i):
		plt.clf()
		ax = draw_distribution(score_samples[i], list_states, real_labels[i])

	anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=n_frame, interval=fps)

	anim.save(path + name_file + '.mp4',writer=animation.FFMpegWriter(fps=8))
	return

def draw_pos(ax, pos_data):
	Xsens_bodies = [ 	'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',  
			'Right Shoulder', 'Right Upper Arm', 'Right Forearm', 'Right Hand',  
			'Left Shoulder', 'Left Upper Arm', 'Left Forearm', 'Left Hand',   	
			'Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe',
			'Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe']

	Xsens_segments = [['Pelvis', 'L5'], ['L5', 'L3'], ['L3', 'T12'], ['T12', 'T8'], ['T8','Neck'], ['Neck', 'Head'], 
		['T8', 'Right Shoulder'], ['Right Shoulder', 'Right Upper Arm'], ['Right Upper Arm', 'Right Forearm'], ['Right Forearm', 'Right Hand'],
		['T8', 'Left Shoulder'],  ['Left Shoulder', 'Left Upper Arm'], ['Left Upper Arm', 'Left Forearm'], ['Left Forearm', 'Left Hand'],
		['Pelvis', 'Right Upper Leg'], ['Right Upper Leg', 'Right Lower Leg'], ['Right Lower Leg', 'Right Foot'], ['Right Foot', 'Right Toe'],
		['Pelvis', 'Left Upper Leg'], ['Left Upper Leg', 'Left Lower Leg'], ['Left Lower Leg', 'Left Foot'], ['Left Foot', 'Left Toe']]

	for seg in Xsens_segments:
		index_ini = Xsens_bodies.index(seg[0])
		x_ini = (pos_data[3*index_ini+0])
		y_ini = (pos_data[3*index_ini+1])
		z_ini = (pos_data[3*index_ini+2])

		index_fin = Xsens_bodies.index(seg[1])
		x_fin = (pos_data[3*index_fin+0])
		y_fin = (pos_data[3*index_fin+1])
		z_fin = (pos_data[3*index_fin+2])

		ax.plot([x_ini, x_fin],	[y_ini, y_fin], [z_ini, z_fin], 'm')

	return ax


def video_sequence(real_labels, predict_labels, video_input, video_output):
		print(len(real_labels), len(predict_labels))
		print(video_input)
		# path = 'C:/Users/amalaise/Documents/These/Xsens/170515_ThyssenkruppData/thyssenkrupp/video_mp4/thyssenkrupp_seq' + str(index) + '.mp4'
		cap = cv2.VideoCapture(video_input)

		# Define the codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(video_output, fourcc, 24.0, (800,600))
		# out = cv2.VideoWriter(path_destination + 'sequence_' + str(index) + '.avi',fourcc, 24.0, (800,600))

		count = 0
		flag = 0
		while(cap.isOpened()):
			ret, frame = cap.read()
			if(ret and flag < len(real_labels)):
				cv2.putText(frame,'Real state: ' + str(real_labels[flag])
					, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
				cv2.putText(frame,'Predict state: ' + str(predict_labels[flag])
					, (10,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
				cv2.imshow('frame', frame)
				out.write(frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				count += 1
				if(count >= 60/24):
					flag += 1
					count = 0
			else:
				break

		cap.release()
		out.release()
		cv2.destroyAllWindows()
		return





