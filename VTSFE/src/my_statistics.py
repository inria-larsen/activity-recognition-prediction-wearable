# -*- coding: utf-8 -*-

import pdb
import yarp
import numpy as np



class My_statistics():
    """Classe s'occupant des stats"""
    def __init__(
        self,
        data=[],
        data_inf=[],
        data_reconstr=[],
        data_shape=[5,70,70,69] #percentObservation, nbTrials, nbFrame, nbMembers
    ):

        self.data = data
        self.data_inf = data_inf
        self.data_reconstr = data_reconstr
        self.data_shape = data_shape

        self.dist_real_reconstr = np.zeros(data_shape[1:4]) #70 70 69
        self.dist_real_reconstr_1D = np.zeros(self.data_shape[1:3]) #70 70
        self.dist_real_inf = np.zeros(data_shape) # 5 70 70 69
        self.dist_real_inf_1D = np.zeros(self.data_shape[0:3]) # 5 70 70
        self.dist_reconstr_inf = np.zeros(data_shape) # 5 70 70 69
        self.dist_reconstr_inf_1D = np.zeros(self.data_shape[0:3]) #5 70 70

        self.mean_dist_real_reconstr = np.zeros(data_shape[1]) # 70
        self.mean_dist_real_inf = np.zeros([self.data_shape[0],self.data_shape[2]]) # 5 70
        self.mean_dist_reconstr_inf = np.zeros([self.data_shape[0],self.data_shape[2]]) # 5 70

        self.var_dist_real_reconstr  = np.zeros(data_shape[1])
        self.var_dist_real_inf = np.zeros([self.data_shape[0],self.data_shape[2]])
        self.var_dist_reconstr_inf = np.zeros([self.data_shape[0],self.data_shape[2]])
        
        self.varGlobalErr_real_reconstr = 0
        self.meanGlobalErr_real_reconstr  =0
        

    def get_distance(self):

        #real_reconstr
        self.dist_real_reconstr = abs(self.data - self.data_reconstr)
        self.dist_real_reconstr_1D = np.mean(self.dist_real_reconstr, axis=2) #mean members
        self.mean_dist_real_reconstr = np.mean(self.dist_real_reconstr_1D, axis=1) #mean timesteps and members
        self.var_dist_real_reconstr = np.var(self.dist_real_reconstr_1D, axis=1)
        self.meanGlobalErr_real_reconstr = np.mean(self.mean_dist_real_reconstr)
        self.varGlobalErr_real_reconstr = np.var(self.mean_dist_real_reconstr)
        if(self.data_inf!=[]):
                for i in range(self.data_shape[0]):
                    #real_inf
                    self.dist_real_inf[i] = abs(self.data - self.data_inf[i])
                    self.dist_real_inf_1D[i] = np.mean(self.dist_real_inf[i], axis=2)
                    self.mean_dist_real_inf[i] = np.mean(self.dist_real_inf_1D[i], axis=1)
                    self.var_dist_real_inf[i] = np.var(self.dist_real_inf_1D[i], axis=1)            
                    #reconstr_inf
                    self.dist_reconstr_inf[i] = abs(self.data_reconstr - self.data_inf[i])
                    self.dist_reconstr_inf_1D[i] = np.mean(self.dist_reconstr_inf[i], axis=2)
                    self.mean_dist_reconstr_inf[i] = np.mean(self.dist_reconstr_inf_1D[i], axis=1)
                    self.var_dist_reconstr_inf[i] = np.var(self.dist_reconstr_inf_1D[i], axis=1)        
        
        
        
