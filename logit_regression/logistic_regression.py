#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com
from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1.0/(1+np.exp(-X))


class logRegressionClassifier(object):

    def __init__(self,**kargs):
        if len(kargs) > 0:
            self.dataMat = kargs['data']
            self.labelMat = kargs['label']
            self.weights = kargs['weight']
            self.learning_rate = kargs['learning_rate']
        else:
            self.dataMat
            self.labelMat
            self.weights
            self.learning_rate = 0.001

    def loadDataSet(self,filename):
        '''导入数据，样本按行，最后一个是lable
        设置默认权重
        '''
        with open(filename,'r') as f:
            data_temp = []
            label_temp = []
            for line in f:
                line_list = line.strip().split(' ')
                line_list = [float(m) for m in line_list]
                label_temp.append(line_list.pop())
                line_list.append(1)
                data_temp.append(line_list)
            self.dataMat = np.array(data_temp,dtype=float)
            self.labelMat = np.array(label_temp,dtype=float)
            self.weights = np.random.ranf(self.dataMat.shape[0]+1)
    def initial_set(self,weight):
        assert weight.shape == self.weights.shape,'weight error.'
        self.weights = weight
    
    def train(self,max_iter,opti_type='randGD',*batch):
        '''opti_type 是一个字符，可以是randGD,batchGD,minibatchGD
        '''
        if opti_type == 'randGD':
            self.weigths = self.randGD(max_iter)
        elif opti_type == 'batchGD':
            self.weights = self.batchGD(max_iter)
        elif opti_type == 'minibatchGD':
            self.weights = self.minibatchGD(max_iter)
    
    def randGD(self,max_iter):
        order = list(range(self.dataMat.shape[0]))
        for i in range(max_iter):
            print('iteration:',i)
            print(self.weights)
            random.shuffle(order)
            for pos in order:
                residual = self.labelMat[pos]-sigmoid(np.sum(self.weights*self.dataMat[pos,:]))
                self.weights += self.learning_rate*residual*self.dataMat[pos,:]

    def predict(self,X):
        X.append(1)
        prob = sigmoid(np.sum(X*self.weights))
        if prob > 0.5:
            return 1.0,prob
        else:
            return 0.0

if __name__ == '__main__':
    
    data = np.array([[1.0,0.0,0.0,1.0],[1.0,0.0,1.0,1.0],[1.0,1.0,1.0,1.0],[0.0,1.0,0.0,1.0],[0.0,0.0,1.0,1.0]])
    labels = np.array([0.0,1.0,1.0,0.0,.0])
    weights = [0.001,0.001,0.0001,0.01]
    lr = logRegressionClassifier(data=data,label=labels,weight=weights,learning_rate=0.001)
    lr.train(max_iter=10)
    print(lr.weights)
    print(lr.predict([0.0,1.0,0.0]))
