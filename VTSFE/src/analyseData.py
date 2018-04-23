# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.my_statistics import My_statistics
with open('data/myLongStats_2LS', 'rb') as fichier:
    mon_dep = pickle.Unpickler(fichier)
    myStats = mon_dep.load()


vall = np.zeros([70,11])
vall[:,0:10] = np.transpose(myStats.mean_dist_real_inf,(1,0))
vall[:,10] = myStats.mean_dist_real_reconstr
labelsName =['10','20','30','40','50','60','70','80','90','100', 'ground \n truth']

plotTest = plt.boxplot(vall, vert=True, patch_artist=True,labels=labelsName)

for i, patch in enumerate(plotTest['boxes']):
    
    if(i==11):
        patch.set_facecolor('green')
        if(i==10):
            patch.set_facecolor('green')
        else:
            patch.set_facecolor('blue')

plt.show()


plotTest = plt.boxplot(np.transpose(myStats.mean_dist_reconstr_inf,[1,0]), notch=True,vert=True, labels=labelsName[0:10])
plt.show()
