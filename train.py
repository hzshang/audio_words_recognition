#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

import numpy as np
from util import *
from classfier import *
# import matplotlib.pyplot as plt

def load_data(file):
    data=np.loadtxt(file)
    x=data[:,:-1]
    y=data[:,-1]
    return x,y.astype(int)

def main():
    train_x,train_y=load_data(TRAIN_FEATURE)
    test_x,test_y=load_data(TEST_FEATURE)
    verify_x,verify_y=load_data(VERIFY_FEATURE)

    cla=classfier(kernel="rbf",gamma1=GAMMA1,gamma2=GAMMA2)
    print "training..."
    cla.train(train_x,train_y)
    print "training done!"

    test_y_predict=cla.predict(test_x)
    train_y_predict=cla.predict(train_x)
    verify_y_predict=cla.predict(verify_x)
    
    print "train acc: %.2f%%"%(100*eval_acc(train_y,train_y_predict))
    print "test acc: %.2f%%"%(100*eval_acc(test_y,test_y_predict))
    print "verity acc: %.2f%%"%(100*eval_acc(verify_y,verify_y_predict))

    cla.save_svm(CLASSFIER1,CLASSFIER2)


if __name__=="__main__":
    main()

