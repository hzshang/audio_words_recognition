#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

"""
init audio from data_set
"""
import wave
import numpy as np
from util import *
import multiprocessing
from multiprocessing.pool import ThreadPool
import sys

def add_noise(array,max=500):
    noise=np.random.normal(0,max,size=array.shape)
#    noise=np.random.randint(-max,max,size=array.shape)
    return array+noise


def add_noise_to_file(file):
    try:
        f=wave.open(file,"r")
        raw_data=f.readframes(f.getnframes())
        array=np.fromstring(raw_data,np.short)
        array=add_noise(array)
        new_file=file.replace(".wav","-a.wav")
        params=f.getparams()
        f.close()
    except Exception as e:
        return
    f=wave.open(new_file,"w")
    f.setparams(params)
    f.writeframes(array.astype(np.short).tostring())
    f.close()

def main():
    files=list_files(sys.argv[1])
    print "add noise to %s..."%sys.argv[1]
    n=multiprocessing.cpu_count()/2
    p=ThreadPool(n)
    p.map(add_noise_to_file,files)


if __name__=="__main__":
    main()


