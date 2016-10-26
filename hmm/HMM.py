#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

'''
Learn HMM model
'''
from __future__ import print_function
import numpy as np

class hmm(object):
    '''
    you can import a HMM's parameters, and create a HMM object.
    '''
    def __init__(self,num_ostate,num_hstate,transfer_mat,confusion_mat,initial_vector):
        self.hstate_length = num_hstate     #隐含状态数
        self.ostate_length = num_ostate     #观察状态数
        self.A_mat = transfer_mat
        self.B_mat = confusion_mat
        self.pi_vec = initial_vector

    def forward(self,o_seq):
        '''
        o_seq是给出的观察序列，观察状态标签是int型数据，范围是0-num_ostate
        '''
        alpha = np.zeros((self.hstate_length,o_seq.shape[0]),dtype=float)
        alpha[:,0] = self.pi_vec*self.B_mat[:,o_seq[0]]
        for i in range(1,alpha.shape[1]):
            alpha[:,i] = np.dot(alpha[:,i-1],self.A_mat)*self.B_mat[:,o_seq[i]]
        return np.sum(alpha[:,-1])



if __name__ == '__main__':

    transfer_mat = np.array([[0.6,0.3,0.1],[0.2,0.5,0.3],[0.2,0.3,0.5]])
    confusion_mat = np.array([[0.8,0.1,0.1],[0.2,0.7,0.1],[0.1,0.1,0.8]])
    initial_vector = np.array([0.8,0.1,0.1])
    hmarkovm = hmm(3,3,transfer_mat,confusion_mat,initial_vector)
    prob = hmarkovm.forward(np.array([0,0,2]))
    print('The prob of given o_sequence:',prob)
