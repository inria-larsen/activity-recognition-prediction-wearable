# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pdb

from my_statistics import My_statistics




def setBoxColors(bp):
    #plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['boxes'][0], facecolor='blue')
    #plt.setp(bp['caps'][0], color='blue')
    #plt.setp(bp['caps'][1], color='blue')
    #plt.setp(bp['whiskers'][0], color='blue')
    #plt.setp(bp['whiskers'][1], color='blue')
    #plt.setp(bp['fliers'][0], color='blue')
    #plt.setp(bp['fliers'][1], color='blue')
    plt.setp(bp['medians'][0], color='white')

    plt.setp(bp['boxes'][1], color='red')
    #plt.setp(bp['caps'][2], color='red')
    #plt.setp(bp['caps'][3], color='red')
    #plt.setp(bp['whiskers'][2], color='red')
    #plt.setp(bp['whiskers'][3], color='red')
    #plt.setp(bp['fliers'][2], color='red')
    #plt.setp(bp['fliers'][3], color='red')
    plt.setp(bp['medians'][1], color='white')

    plt.setp(bp['boxes'][2], color='green')
    #plt.setp(bp['caps'][4], color='green')
    #plt.setp(bp['caps'][5], color='green')
    #plt.setp(bp['whiskers'][4], color='green')
    #plt.setp(bp['whiskers'][5], color='green')
    plt.setp(bp['medians'][2], color='white')

    plt.setp(bp['boxes'][3], color='black')
    #plt.setp(bp['caps'][6], color='black')
    #plt.setp(bp['caps'][7], color='black')
    #plt.setp(bp['whiskers'][6], color='black')
    #plt.setp(bp['whiskers'][7], color='black')
    plt.setp(bp['medians'][3], color='white')

    plt.setp(bp['boxes'][4], color='magenta')
    #plt.setp(bp['caps'][8], color='magenta')
    #plt.setp(bp['caps'][9], color='magenta')
    #plt.setp(bp['whiskers'][8], color='magenta')
    #plt.setp(bp['whiskers'][9], color='magenta')
    plt.setp(bp['medians'][4], color='white')







myStats= np.zeros(5,My_statistics)
listLS = ['data/myLittleStatsLS2']#,'data/myLongStats_5LS','data/myLongStatsLS7','data/myLongStatsLS15','data/myLongStatsLS20']
labelsName =['10']#,'20','30','40','50','60','70','80','90','100', 'Control\ngroup']

all_dist_reconstr_real = np.zeros([5,70])


for nbSS, myLS in enumerate(listLS):
    with open(myLS, 'rb') as fichier:
        mon_dep = pickle.Unpickler(fichier)
        myStats[nbSS] = mon_dep.load()
pdb.set_trace()
    

all_dist_reconstr_real[nbSS,:] = myStats[nbSS].mean_dist_real_reconstr #conrol group

    ##plotTest = plt.boxplot(all_dist_real_inf, vert=True, patch_artist=True,labels=labelsName)
    
    ##for i, patch in enumerate(plotTest['boxes']):
        ##if(i==10):
            ##patch.set_facecolor('green')
        ##else:
            ##patch.set_facecolor('blue')
    
    ##plt.show()
    
    
    ##plotTest = plt.boxplot(np.transpose(myStats[nbSS] .mean_dist_reconstr_inf,[1,0]), notch=True,vert=True, labels=labelsName[0:10])
    ##plt.show()
    

fig = plt.figure()
ax = plt.axes()
plt.hold(True)

bp = plt.boxplot(all_dist_reconstr_real[0,:], widths = 0.8,patch_artist = True)
plt.setp(bp['boxes'][0], facecolor='cyan')

## set axes limits and labels
#plt.xlim(0,62)
#plt.ylim(0,0.075)
#plt.title('Error distance between the real trajectory and the infered one' , fontsize=20)
#ax.set_xticklabels(labelsName, fontsize=20)
#ax.yaxis.label.set_size(20)
#ax.set_xticks([3,9,15,21,27,33,39,45,51,57,62])
#ax.yaxis.set_tick_params(labelsize=20)

#ax.set_xlabel('Observed data [%]', fontsize=20)
#ax.set_ylabel('Distance error [m]', fontsize=20)
## draw temporary red and blue lines and use them to create a legend
#hB, = plt.plot([1,1],'b-')
#hR, = plt.plot([1,1],'r-')
#hG, = plt.plot([1,1],'g-')
#hK, = plt.plot([1,1],'k-')
#hM, = plt.plot([1,1],'m-')
##hC, = plt.plot([1,1],'c-')
#plt.legend((hB, hR, hG, hK, hM),('2', '5', '7', '15', '20'), title='Latent Space dimension', fontsize=12)
#hB.set_visible(False)
#hR.set_visible(False)
#hG.set_visible(False)
#hK.set_visible(False)
#hM.set_visible(False)
##hC.set_visible(False)
plt.show()


#fig = plt.figure()
#ax = plt.axes()
#plt.hold(True)

#for i in range(10):
    ## i-th boxplot
    #bp = plt.boxplot(np.transpose(all_dist_reconstr_inf[:,:,i],(1,0)), positions = [ i+1 + i*5, i+2+i*5, i+3+i*5,i+4+i*5, i+(i+1)*5], widths = 0.8,patch_artist = True)
    #setBoxColors(bp)


## set axes limits and labels
#plt.xlim(0,61)
#plt.ylim(0,0.0175)
#plt.title('Error distance between the real reconstructed trajectory and the infered one' , fontsize=20)

#ax.set_xticklabels(labelsName, fontsize=20)

#ax.set_xticks([3,9,15,21,27,33,39,45,51,57])
#ax.yaxis.set_tick_params(labelsize=20)

#ax.set_xlabel('Observed data [%]', fontsize=20)
#ax.set_ylabel('Distance error [m]', fontsize=20)


## draw temporary red and blue lines and use them to create a legend
#hB, = plt.plot([1,1],'b-')
#hR, = plt.plot([1,1],'r-')
#hG, = plt.plot([1,1],'g-')
#hK, = plt.plot([1,1],'k-')
#hM, = plt.plot([1,1],'m-')
##hC, = plt.plot([1,1],'c-')
#plt.legend((hB, hR, hG, hK, hM),('2', '5', '7', '15', '20'), title='Latent Space dimension', fontsize=12)
#hB.set_visible(False)
#hR.set_visible(False)
#hG.set_visible(False)
#hK.set_visible(False)
#hM.set_visible(False)
##hC.set_visible(False)

#plt.savefig('boxcompare.png')
#plt.show()

