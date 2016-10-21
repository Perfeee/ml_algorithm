#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

from __future__ import print_function
from gradient_descent import Rosenbrock as Rb_func
import numpy as np
import matplotlib.pyplot as plt

class Rosenbrock(object):

    def get_value(self,x,y):
        '''获取Rosenbrock函数在某个点的函数值，或者n个点上的函数值，x和y分别是Rosenbrock的自变量
        '''
        return Rb_func(x,y)

    def Hessian(self,x):
        '''
        返回Rosenbrock函数特定某个点的hessian矩阵,x是一个n×1的array，n是自变量的个数，因为这里就是特定指Rosenbrock函数，所以n=2
        '''
        return np.array([[1200*x[0]*x[0]-400*x[1]+2,-400*x[0]],[-400*x[0],200]])

    def derivation(self,x):
        '''
        返回Rosenbrock函数特定某个点的一阶偏导，x是一个n×1的array，n是自变量的个数
        '''
        return np.array([400*np.power(x[0],3)-400*x[0]*x[1]+2*x[0]-2,200*(x[1]-np.power(x[0],2))])


class newton(object):

    def __init__(self,func_object,max_Iter,**kargs):
        self.func = func_object
        self.max_Iter = max_Iter
        self.init_point = np.array([2.0,2.5])
        self.learning_rate = 0.0001
        if kargs:
            try:
                self.init_point = np.array(kargs['init_point'])
                self.learning_rate = kargs['learning_rate']
            except:
                raise KeyError
    def train(self):
        
        record = [self.init_point]
        for i in range(self.max_Iter):
            x = record[-1] - np.linalg.inv(self.func.Hessian(record[-1])).dot(self.func.derivation(record[-1]))
            record.append(x)
        return record



def test():
    rb = Rosenbrock() 
    opti = newton(rb,20)
    record = opti.train()
    record = np.array(record)
    f = rb.get_value(record[:,0],record[:,1])
    print(record)
    print(f)
    
    x = np.arange(-1.5,2.0,0.05)
    y = np.arange(-3.5,3.0,0.05)
    x,y = np.meshgrid(x,y)
    z = rb.get_value(x,y)
    plt.contour(x,y,z,20)
    plt.plot(record[:,0],record[:,1],'g*',record[:,0],record[:,1])
    plt.show()



if __name__ == '__main__':
    test()
