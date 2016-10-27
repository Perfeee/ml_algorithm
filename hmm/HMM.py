#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

'''
Learn HMM model
'''
from __future__ import print_function
import numpy as np

class hmm_learning(object):
    '''
    use same o_sequences to learn HMM's pamameters (transfer_mat,confusion_mat,pi_vector) with forward-backward algorithm.
    '''
    def __init__(self,):
        pass


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
        for t in range(1,alpha.shape[1]):
            alpha[:,t] = np.dot(alpha[:,t-1],self.A_mat)*self.B_mat[:,o_seq[t]]
        return np.sum(alpha[:,-1])
    
    def backward(self,o_seq):
        '''
        后向算法，计算后向局部概率
        '''
        beta = np.zeros((self.hstate_length,o_seq.shape[0]),dtype=float)
        beta[:,-1] = 1.0
        for t in np.arange(beta.shape[1]-2,-1,-1):
            beta[:,t] = np.dot(beta[:,t+1],self.A_mat)*self.B_mat[:,o_seq[t+1]]
        return np.sum(beta[:,0]*self.pi_vec*self.B_mat[:,o_seq[0]])

    def viterbi(self,o_seq):
        '''
        o_seq是观察序列
        '''
        sigma = np.zeros((self.hstate_length,o_seq.shape[0]),dtype=float)
        sigma[:,0] = self.pi_vec*self.B_mat[:,o_seq[0]]
        path_record = np.zeros(sigma.shape,dtype=int)
        path_record[:,0] = np.arange(0,self.hstate_length,step=1)  #路径记录，行标签就是隐含状态，列标签就是时间步t，某个特定位置上的数值就代表当前t和选取的隐含状态，它的最优前一状态(回溯整个序列构成最优路径)
        for t in range(1,sigma.shape[1]):
            temp = sigma[:,t-1].reshape((3,1))*self.A_mat*(self.B_mat[:,o_seq[t]].reshape((3,1)))   #temp的行标签就是前一状态，列标签就是当前的状态
            sigma[:,t] = np.max(temp,axis=0)    #为当前的每一状态选项选出最佳的前一状态，也就是保存当前每一状态的最佳路径概率
            path_record[:,t] = np.argmax(temp,axis=0) #保存当前每一状态相对应的最佳前一状态
        h_seq = np.empty(o_seq.shape[0],dtype=int)
        h_seq[-1] = path_record[np.argmax(sigma[:,-1]),-1]
        max_prob = np.max(sigma[:,-1])
        for t in np.arange(o_seq.shape[0]-2,-1,-1):
            h_seq[t] = path_record[h_seq[t+1],t]
        return h_seq,max_prob

        

def test_forward_backward():
    A = np.array([[0.5,0.375,0.125],[0.25,0.125,0.625],[0.25,0.375,0.375]])
    B = np.array([[0.6,0.2,0.15,0.05],[0.25,0.25,0.25,0.25],[0.05,0.1,0.35,0.5]])
    pi_vec = np.array([0.63,0.17,0.2])
    hmmodel = hmm(4,3,A,B,pi_vec)
    forward_prob = hmmodel.forward(np.array([0,2,3]))
    backward_prob = hmmodel.backward(np.array([0,2,3]))
    print('forward prob:',forward_prob,'backward prob:',backward_prob)

def test_viterbi():
    A = np.array([[0.333,0.333,0.333],[0.333,0.333,0.333],[0.333,0.333,0.333]])
    B = np.array([[0.5,0.5],[0.75,0.25],[0.25,0.75]])
    pi_vec = np.array([0.333,0.333,0.333])
    hmmodel = hmm(2,3,A,B,pi_vec)
    h_seq,prob = hmmodel.viterbi(np.array([0,0,0,0,1,0,1,1,1,1],dtype=int))
    log_prob = np.log(prob)
    print('The best hidden sequence is: ',h_seq,"Prob:",prob,'log_prob:',log_prob)

if __name__ == '__main__':
     
    test_forward_backward()
    test_viterbi()
    transfer_mat = np.array([[0.6,0.3,0.1],[0.2,0.5,0.3],[0.2,0.3,0.5]])
    confusion_mat = np.array([[0.8,0.1,0.1],[0.2,0.7,0.1],[0.1,0.1,0.8]])
    initial_vector = np.array([0.8,0.1,0.1])
    hmarkovm = hmm(3,3,transfer_mat,confusion_mat,initial_vector)
    f_prob = hmarkovm.forward(np.array([0,0,2]))
    b_prob = hmarkovm.backward(np.array([0,0,2]))
    print('The prob of given o_sequence:',f_prob,b_prob)
