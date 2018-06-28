#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

from sklearn import svm
from sklearn.externals import joblib
from util import *

class classfier:
    def __init__(self,kernel="rbf",gamma1=0.00002,gamma2=0.00002):
        self.svm1=svm.SVC(kernel=kernel,gamma=gamma1)
        self.svm2=svm.SVC(kernel=kernel,gamma=gamma2)
        

    def train(self,x,y):
        self.svm1.fit(x,y)
        idx=np.where((y==0)|(y==1))[0]
        self.svm2.fit(x[idx],y[idx])


    def predict(self,x):
        y=self.svm1.predict(x)
        idx=np.where((y==0)|(y==1))[0]
        if idx.shape[0] >0:
            yy=self.svm2.predict(x[idx])
            y[idx]=yy
        return y

    def load_svm(self,path1,path2):
        self.svm1=joblib.load(path1)
        self.svm2=joblib.load(path2)

    def save_svm(self,path1,path2):
        joblib.dump(self.svm1,path1)
        joblib.dump(self.svm2,path2)

