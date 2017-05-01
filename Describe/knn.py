# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors

def L2(pred,true):
    loss = np.square(pred-true)
    return loss.mean()

def L1(pred,true):
    loss = np.abs(pred-true)
    return loss.mean()

def MAPE(pred,true):
    print ("pred",pred)
    print ("true",true)
    loss = np.abs((pred-true)/(true))
    return loss.mean()

#This function chooses the best point estimate for a numpy array, according to a particular loss.
#The loss function should take two numpy arrays as arguments, and return a scalar. One example is MAPE, see above.
def solver(x,loss):
    mean = x.mean()
    best = loss(mean,x)
    result = mean
    for i in x:
        score = loss(i,x)
        if score < best:
            best = score
            result = i
    return result
class NonparametricKNN(object):
    def __init__(self,n_neighbors=5,loss='L2'):
        if loss in ['L1','L2','MAPE']:
            loss = {'L1':L1,'L2':L2,'MAPE':MAPE}[loss]
        self.model = NearestNeighbors(n_neighbors,algorithm='auto',n_jobs=-1)
        self.solver = lambda x:solver(x,loss)
    def fit(self,train,target):#All inputs should be numpy arrays.
        self.model.fit(train)
        self.f=np.vectorize(lambda x:target[x])
        self.f =lambda x:target[x]
        return self
#     在以前的历史数据中找到相同模式的历史数据，用它们的值作为预测值
    def predict(self,test):#Return predictions as a numpy array.
        neighbors = self.model.kneighbors(test,return_distance=False)
        neighbors = self.f(neighbors)
        result = np.apply_along_axis(self.solver,1,neighbors)
        return result