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



def show_exe():
    # train_x,train_y=load_data(TRAIN_FEATURE)
    # test_x,test_y=load_data(TEST_FEATURE)
    # verify_x,verify_y=load_data(VERIFY_FEATURE)
    e=np.linspace(-8,1,20)
    gamma=np.power(10,e)
    test=[]
    train=[]
    verify=[]
    r=0
    # for i in gamma:
    #     print "round:",r
    #     r=r+1
    #     cla=classfier()
    #     cla.train(train_x,train_y,kernel="rbf",gamma=i)
    #     test_y_predict=cla.predict(test_x)
    #     train_y_predict=cla.predict(train_x)
    #     verify_y_predict=cla.predict(verify_x)
    #     test.append(100*eval_acc(test_y,test_y_predict))
    #     train.append(100*eval_acc(train_y,train_y_predict))
    #     verify.append(100*eval_acc(verify_y,verify_y_predict))

    print train
    print test
    print verify
    train=[15.209125475285171, 15.209125475285171, 15.209125475285171, 15.209125475285171, 62.737642585551335, 83.269961977186313, 95.564005069708486, 99.112801013941692, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    test=[15.0, 15.0, 15.0, 15.0, 60.277777777777771, 85.277777777777771, 90.277777777777786, 91.666666666666657, 90.833333333333329, 48.055555555555557, 8.0555555555555554, 5.5555555555555554, 5.4, 5.3, 5.2, 5.0, 5.0, 5.0, 5.0, 5.0]
    verify=[15.0, 15.0, 15.0, 15.0, 50.749999999999993, 70.625, 66.25, 64.25, 17.375, 5.0, 5.0, 5.0, 5.0, 5.25, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    l1,=plt.plot(e,train,color="blue",label="train")
    l2,=plt.plot(e,test,color="red",label="test")
    l3,=plt.plot(e,verify,color="green",label="verify")
    plt.legend(loc='upper right')
    plt.xlabel("log(gamma)")
    plt.ylabel("accuracy (%)")
    plt.show()



def main():
    train_x,train_y=load_data(TRAIN_FEATURE)
    test_x,test_y=load_data(TEST_FEATURE)
    verify_x,verify_y=load_data(VERIFY_FEATURE)

    cla=classfier()
    print "training..."
    cla.train(train_x,train_y,kernel="rbf",gamma=GAMMA)
    print "training done!"

    test_y_predict=cla.predict(test_x)
    train_y_predict=cla.predict(train_x)
    verify_y_predict=cla.predict(verify_x)
    
    print "train acc: %.2f%%"%(100*eval_acc(train_y,train_y_predict))
    print "test acc: %.2f%%"%(100*eval_acc(test_y,test_y_predict))
    print "verity acc: %.2f%%"%(100*eval_acc(verify_y,verify_y_predict))

    cla.save_svm(CLASSFIER)


if __name__=="__main__":
    main()

