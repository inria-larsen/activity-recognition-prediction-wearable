import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import cv2
from copy import deepcopy


def draw_distribution(score, list_states, real_labels):
	

	clrs = np.zeros((len(list_states), 3))
	# sns.set(font_scale=1.5)
	id_pred = np.argmax(score)
	id_real = list_states.index(real_labels)

	

	for x in range(len(list_states)):
		if((id_pred == id_real) and (x == id_real)):
			clrs[x] = [0,1,0]
		# elif(x == id_real):
		# 	clrs[x] = [0,0,1]
		else:
			clrs[x] = [1,0,0]

	labels = deepcopy(list_states)
	for i in range(len(labels)):
		labels[i] = labels[i].title()
		labels[i] = labels[i].replace("_", " ")
		labels[i] = labels[i].replace("St", "Standing")
		labels[i] = labels[i].replace("Wa", "Walking")
		labels[i] = labels[i].replace("Kn", "Kneeling")
		labels[i] = labels[i].replace("Cr", "Crouching")

		labels[i] = labels[i].replace("Bf", "Bent forward")
		labels[i] = labels[i].replace("Bs", "Strongly bent")
		labels[i] = labels[i].replace("U", "Upright")
		labels[i] = labels[i].replace("Oh", "Hands above head")
		labels[i] = labels[i].replace("Os", "Elbows above shoulders")

		labels[i] = labels[i].replace("Id", "Idle")
		labels[i] = labels[i].replace("Re", "Reach")
		labels[i] = labels[i].replace("Rl", "Release")
		labels[i] = labels[i].replace("Fm", "Fine Manipulation")
		labels[i] = labels[i].replace("Sc", "Screw")
		labels[i] = labels[i].replace("Ca", "Carry")
		labels[i] = labels[i].replace("Pl", "Place")
		labels[i] = labels[i].replace("Pi", "Pick")

	ax = sns.barplot(score, labels, palette=clrs)
	ax.set_xlim(0,1)
	plt.title('Posture')

	ax.title.set_fontsize(50)
	plt.yticks(size = 40)
	plt.xticks(size = 40)
	# plt.ylabel('States')
	plt.xlabel('Probabilities', size=20)
	plt.subplots_adjust(left=0.7)
	
	return sns.barplot(score, labels, palette=clrs)

def video_distribution(score_samples, list_states, real_labels, fps, path, name_file):

	fig=plt.figure(figsize=(15,10))
	# plt.rcParams["figure.figsize"] = (5,50)
	
	ax = fig.add_subplot(1,1,1)
	n_frame = np.shape(real_labels)[0]

	ax = draw_distribution(score_samples[0], list_states, real_labels[0])

	# def animate(i):
	# 	plt.clf()
	# 	ax = draw_distribution(score_samples[i], list_states, real_labels[i])

	# anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=n_frame, interval=fps)

	# anim.save(path + name_file + '.mp4',writer=animation.FFMpegWriter(fps=8))
	plt.show()
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
		cap = cv2.VideoCapture(video_input)

		# Define the codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(video_output, fourcc, 24.0, (1280,720))
		# out = cv2.VideoWriter(path_destination + 'sequence_' + str(index) + '.avi',fourcc, 24.0, (800,600))

		count = 0
		flag = 0
		while(cap.isOpened()):
			ret, frame = cap.read()
			
			if(ret and flag < len(real_labels)):
				cv2.putText(frame,'Predicted Posture: ' + str(real_labels[flag])
					, (50,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
				cv2.putText(frame,'Predicted Action: ' + str(predict_labels[flag])
					, (50,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
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





