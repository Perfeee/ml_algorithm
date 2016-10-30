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
    
    def _state_gen(self,current_hstate=None,state_type='h'):
        '''
        generate a hidden state or an observation state
        '''
        temp = np.random.rand(1)[0]
        if not current_hstate:
            trans_prob = self.pi_vec
        elif state_type == 'h':
            trans_prob = self.A_mat[current_hstate,:]
        elif state_type == 'o':
            trans_prob = self.B_mat[current_hstate,:]
        assert np.sum(trans_prob)<=1, "trans_prob error"
        for i in range(trans_prob.shape[0]):
                if temp < np.sum(trans_prob[:i+1]):
                    return i
                else:
                    continue
    

    def o_seq_gen(self,length):
        '''
        generate a observation sequence with given HMM model and length.
        '''
        o_seq = np.empty(length,dtype=int)
        h_seq = np.empty(length,dtype=int)
        h_seq[0] = self._state_gen()
        for i in range(length):
            o_seq[i] = self._state_gen(current_hstate=h_seq[i],state_type='o')
            try:
                h_seq[i+1] = self._state_gen(current_hstate=h_seq[i],state_type='h')
            except IndexError:
                return o_seq,h_seq

    def forward(self,o_seq):
        '''
        o_seq是给出的观察序列，观察状态标签是int型数据，范围是0-num_ostate
        '''
        alpha = np.zeros((self.hstate_length,o_seq.shape[0]),dtype=float)
        alpha[:,0] = self.pi_vec*self.B_mat[:,o_seq[0]]
        for t in range(1,alpha.shape[1]):
            alpha[:,t] = np.dot(alpha[:,t-1],self.A_mat)*self.B_mat[:,o_seq[t]]
        return np.sum(alpha[:,-1]),alpha
    
    def backward(self,o_seq):
        '''
        后向算法，计算后向局部概率
        '''
        beta = np.zeros((self.hstate_length,o_seq.shape[0]),dtype=float)
        beta[:,-1] = 1.0
        for t in np.arange(beta.shape[1]-2,-1,-1):
#            beta[:,t] = np.dot(beta[:,t+1],self.A_mat)*self.B_mat[:,o_seq[t+1]]        error expression.
            beta[:,t] = np.dot(self.A_mat,self.B_mat[:,o_seq[t+1]]*beta[:,t+1])
        return np.sum(beta[:,0]*self.pi_vec*self.B_mat[:,o_seq[0]]),beta

    def gamma_gen(self,alpha,beta):
        '''利用前向局部概率alpha和后向局部概率beta计算：给定模型lambda和观测序列O，在某时刻处于某状态的概率
        return gamma  行索引为状态，列索引为t
        '''
        gamma = np.empty_like(alpha)
        temp = alpha*beta
        gamma = temp/np.sum(temp,axis=0)
        return gamma 

    def ksi_gen(self,alpha,beta,o_seq):
        '''利用alpha和beta计算：给定模型lamda和观测序列O，在t时刻为状态i且在t+1时刻为状态j的概率，即在t时刻为i，并转移至状态j的概率
        return ksi  是一个三维矩阵，第一维是时刻t（长度只有T-1），第二维是t时刻状态i，第三维是t+1时刻状态j
        '''
        ksi = np.empty((alpha.shape[1]-1,alpha.shape[0],alpha.shape[0]),dtype=float)
        for t in range(alpha.shape[1]-1):
            temp = alpha[:,t]*self.A_mat*self.B_mat[:,o_seq[t+1]]*beta[:,t+1]
            ksi[t,:,:] = temp/np.sum(temp)
        return ksi

    def approxi(self,o_seq,gamma):
        '''解决decodeing问题，近似算法，在给定lambda和o_seq的情况下，预测每个位置最可能出现的状态，即为该位置的状态。该方法简单，但是可能出现实际不可能出现的状态，比如预测的状态序列有一环的转移概率实际为0。这里的原因是因为，每个时刻隐含状态的选择，仅依赖与当前时刻最可能的状态。而不像viterbi算法，是根据整个序列最后的概率来选择的。
        '''
        return np.argmax(gamma,axis=0)

    def viterbi(self,o_seq):
        '''解决decoding问题，预测最优隐含序列
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

class hmm_learning(hmm):
    '''
    use some o_sequences to learn HMM's pamameters (transfer_mat,confusion_mat,pi_vector) with forward-backward algorithm.
    '''
    def __init__(self,num_ostate,num_hstate,o_seq,init_trans_mat=None,init_conf_mat=None,init_pi_vec=None,):
        '''设置初始参数，如果单观察序列，那就只能随机（针对不同的问题，可能有更好的随机方式/先验分布？）。如果是多观察序列，那么可以先拿一部分训练，得到的参数作为剩下数据的训练的初始参数。
        '''
        def std(mat,axis=1):
            mat = np.abs(mat)
            if axis:
                mat = mat/(np.sum(mat,axis).reshape(mat.shape[0],1))
            else:
                mat= mat/np.sum(mat)
            return mat

        if not init_trans_mat:
            '''使用标准正态分布来初始化参数，注意这是带条件的初始化，概率大于零，并且和为1.
            '''
            init_trans_mat = np.random.randn(num_hstate,num_hstate)
            init_trans_mat = std(init_trans_mat)
            init_conf_mat = np.random.randn(num_hstate,num_ostate)
            init_conf_mat = std(init_conf_mat)
            init_pi_vec = np.random.randn(num_hstate)
            init_pi_vec = std(init_pi_vec,axis=None)
        hmm.__init__(self,num_ostate,num_hstate,init_trans_mat,init_conf_mat,init_pi_vec)
        self.o_seq = o_seq
    
    def fit(self,max_iter=10,):
        '''HMM参数学习，如果是多序列，那么采用Levinson方法，假设它们是独立的。
        '''
        def row_compute(row):
            temp = [np.sum(row[np.where(self.o_seq==o)[0]]) for o in range(self.ostate_length)]
            assert len(temp)==self.ostate_length,'B_mat learning error.'
            return np.array(temp)
        def single_o_seq(o_seq):        
            alpha = self.forward(o_seq)[1]
            beta = self.backward(o_seq)[1]
            gamma = self.gamma_gen(alpha,beta)
            ksi = self.ksi_gen(alpha,beta,o_seq)
            new_A_mat_nume = np.apply_along_axis(np.sum,0,ksi)
            new_A_mat_deno = np.sum(gamma[:,0:-1],axis=1).reshape(self.A_mat.shape[0],1)
            new_B_mat_nume = np.apply_along_axis(row_compute,axis=1,arr=gamma)
            new_B_mat_deno = np.sum(gamma,axis=1).reshape(self.B_mat.shape[0],1)
            new_pi_vec= np.empty_like(self.pi_vec)
            new_pi_vec = gamma[:,0]
            return new_A_mat_nume,new_A_mat_deno,new_B_mat_nume,new_B_mat_deno,new_pi_vec
        if self.o_seq.size == self.o_seq.shape[0]:
            '''单序列
            '''
            for i in range(max_iter):
                new_A_mat_nume,new_A_mat_deno,new_B_mat_nume,new_B_mat_deno,new_pi_vec = single_o_seq(self.o_seq)
                self.A_mat = new_A_mat_nume/new_A_mat_deno
                self.B_mat = new_B_mat_nume/new_B_mat_deno
                self.pi_vec = new_pi_vec
        else:
            for i in range(max_iter):
                A_mat_nume = np.zeros_like(self.A_mat)
                A_mat_deno = np.zeros((self.A_mat.shape[0],1))
                B_mat_nume = np.zeros_like(self.B_mat)
                B_mat_deno = np.zeros((self.B_mat.shape[0],1))
                pi_vec = np.zeros_like(self.pi_vec)
                for j in range(self.o_seq.shape[0]):
                    new_A_mat_nume,new_A_mat_deno,new_B_mat_nume,new_B_mat_deno,new_pi_vec = single_o_seq(self.o_seq[j,:])
                    A_mat_nume += new_A_mat_nume
                    A_mat_deno += new_A_mat_deno
                    B_mat_nume += new_B_mat_nume
                    B_mat_deno += new_B_mat_deno
                    pi_vec += new_pi_vec
                self.A_mat = A_mat_nume/A_mat_deno
                self.B_mat = B_mat_nume/B_mat_deno
                self.pi_vec = pi_vec/self.o_seq.shape[0] 

def test_forward_backward():
    A = np.array([[0.5,0.375,0.125],[0.25,0.125,0.625],[0.25,0.375,0.375]])
    B = np.array([[0.6,0.2,0.15,0.05],[0.25,0.25,0.25,0.25],[0.05,0.1,0.35,0.5]])
    pi_vec = np.array([0.63,0.17,0.2])
    hmmodel = hmm(4,3,A,B,pi_vec)
    forward_prob = hmmodel.forward(np.array([0,2,3]))[0]
    backward_prob = hmmodel.backward(np.array([0,2,3]))[1]
    print('forward prob:',forward_prob,'backward prob:',backward_prob)
    o_seq = hmmodel.o_seq_gen(5)[0]
    print(o_seq)
    f_prob = hmmodel.forward(o_seq)[0]
    b_prob = hmmodel.backward(o_seq)[1]
    print('generated o_seq\'s forward prob and backward prob:',f_prob,b_prob)


def test_viterbi():
    A = np.array([[0.333,0.333,0.333],[0.333,0.333,0.333],[0.333,0.333,0.333]])
    B = np.array([[0.5,0.5],[0.75,0.25],[0.25,0.75]])
    pi_vec = np.array([0.333,0.333,0.333])
    hmmodel = hmm(2,3,A,B,pi_vec)
    o_seq = np.array([0,0,0,0,1,0,1,1,1,1],dtype=int)
    h_seq,prob = hmmodel.viterbi(o_seq)
    log_prob = np.log(prob)
    alpha = hmmodel.forward(o_seq)[1]
    beta = hmmodel.backward(o_seq)[1]
    gamma = hmmodel.gamma_gen(alpha,beta)
    ha_seq = hmmodel.approxi(o_seq,gamma)
    print('The best hidden sequence given by viterbi is: ',h_seq,"Prob:",prob,'log_prob:',log_prob)
    print('The best hidden sequence given by approxi is: ',ha_seq)

def learning_test():
    model = hmm_learning(3,3,np.array([0,1,2]))
    print('fitted before:',model.A_mat,model.B_mat,model.pi_vec)
    model.fit(max_iter=3)
    print('aftered fitted:',model.A_mat,model.B_mat,model.pi_vec)


if __name__ == '__main__':
     
    test_forward_backward()
    test_viterbi()
    
    transfer_mat = np.array([[0.6,0.3,0.1],[0.2,0.5,0.3],[0.2,0.3,0.5]])
    confusion_mat = np.array([[0.8,0.1,0.1],[0.2,0.7,0.1],[0.1,0.1,0.8]])
    initial_vector = np.array([0.8,0.1,0.1])
    hmarkovm = hmm(3,3,transfer_mat,confusion_mat,initial_vector)
    f_prob = hmarkovm.forward(np.array([0,0,2]))[0]
    b_prob = hmarkovm.backward(np.array([0,0,2]))[0]
    print('The prob of given o_sequence:',f_prob,b_prob)
    
    transfer_mat = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
    confusion_mat = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
    initial_vector = np.array([0.2,0.4,0.4])
    hmarkovm = hmm(3,3,transfer_mat,confusion_mat,initial_vector)
    o_seq = np.array([0,1,0])
    f_prob,alpha = hmarkovm.forward(o_seq)
    b_prob,beta = hmarkovm.backward(o_seq)
    print('The prob of given o_sequence:',f_prob,b_prob)

    gamma = hmarkovm.gamma_gen(alpha,beta)
    ksi = hmarkovm.ksi_gen(alpha,beta,o_seq)
    print('gamma:',gamma)
    print('ksi:',ksi,'shape:',ksi.shape)


    learning_test()
