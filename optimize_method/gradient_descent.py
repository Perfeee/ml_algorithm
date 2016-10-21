#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

'''
实现梯度下降算法
'''

from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as amat

def Rosenbrock(x,y):
    '''
    Rosenbrock函数是一个常用来测试最优化演算法的函数，也称山谷或香蕉函数。
    '''
    return np.square(1-x)+100*np.square(y-np.square(x))

class gradient_descent(object):
    '''
    使用梯度下降法寻找函数最优点的类，初始化时，需要参数func，和最大迭代次数
    可选关键字参数是初始点init_point,类型是list
    '''

    def __init__(self,func,max_Iter,**kargs):
        self.func = func
        self.max_Iter = max_Iter
        self.init_point = [4.0,4.0]
        self.learning_rate = 0.001
        if kargs:
            try:
                self.init_point = kargs['init_point']
                self.learning_rate = kargs['learning_rate']
            except:
                raise KeyError

    def grad_descent(self):
        
        x_record = [self.init_point[0]]
        y_record = [self.init_point[1]]
        for i in range(self.max_Iter):
            x = x_record[-1] - self.learning_rate*(400.0*np.power(x_record[-1],3) - 400*x_record[-1]*y_record[-1]+2*x_record[-1]-2)
            y = y_record[-1] - self.learning_rate*(200.0*(y_record[-1]-np.square(x_record[-1])))
            x_record.append(x)
            y_record.append(y)
        return x_record,y_record
    
    def drawPath(self,x,y,f,X,Y):
        fig  = plt.figure()
        ax = Axes3D(fig)
        X,Y = np.meshgrid(X,Y,sparse=True)
        Z = self.func(X,Y)
        plt.title('Gradient Descent')
        ax.set_xlabel('x label',color='r')
        ax.set_ylabel('y label',color='g')
        ax.set_zlabel('z label',color='b')
        ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
        ax.plot(x,y,f,'r.')
        plt.show()


def test():
    print('Rosenbrock\'s minimum value:',Rosenbrock(1,1))
    gd = gradient_descent(Rosenbrock,100)
    x_record,y_record = gd.grad_descent()
    f = [Rosenbrock(x,y) for x,y in zip(x_record,y_record)]
    X = np.arange(-1,2,0.1)
    Y = np.arange(-1,2,0.1)
    gd.drawPath(x_record,y_record,f,X,Y)
    

if __name__ == '__main__':
    test()

