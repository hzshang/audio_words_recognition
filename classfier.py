#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

from sklearn import svm
from sklearn.externals import joblib

class classfier:
    def __init__(self):
        pass

    def train(self,x,y,kernel="rbf",gamma=0.00002):
        self.svm=svm.SVC(kernel=kernel,gamma=gamma)
        self.svm.fit(x,y)

    def predict(self,x):
        return self.svm.predict(x)

    def load_svm(self,path):
        self.svm=joblib.load(path)

    def save_svm(self,path):
        joblib.dump(self.svm,path)

