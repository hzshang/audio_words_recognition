#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.
import numpy as np
from util import *
from classfier import *
from dump_feature import dir2feature
import sys

def main():
    if len(sys.argv)!= 2:
        print "Usage ./verify.py dir"
    data=dir2feature(sys.argv[1],**config)
    x=data[:,:-1]
    y=data[:,-1]
    cla=classfier()
    cla.load_svm(CLASSFIER)
    y_pred=cla.predict(x)
    print "acc: %.2f%%"%(100*eval_acc(y,y_pred))

if __name__=="__main__":
    main()
