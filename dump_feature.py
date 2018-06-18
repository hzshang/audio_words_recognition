#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

from util import *
import sys
import os
import signal
from tqdm import tqdm
from voice import *
import numpy as np
import logging
import thread
import multiprocessing
from multiprocessing.pool import ThreadPool


def signal_handler(signal,frame):
    plt.close()
    sys.exit(0)


def list2feature(files,**kw):
    sample_num=len(files)
    l=[]
    for i in tqdm(range(sample_num)):
        try:
            signal,rate,cla=read_wav_cla(files[i])
            feat=wav2feature(signal,rate,**kw)
            if feat is not None:
                feat=np.append(feat,cla)
                l.append(feat)
            else:
                continue
        except Exception as e:
            raise e
            logging.warning("load file %s failed."%files[i])
            if drop:
                continue
            else:
                l.append(np.ones(2*kw["clip_num"]*kw["cep_num"]+1))
    data=np.vstack(l)
    print "\n"
    return data

def dir2feature(dir_name,pool="auto",**kw):
    """
    load dir wav for train or test using multi thread
    """
    l=[]
    if pool == "auto":
        pool=int(multiprocessing.cpu_count()*0.75)

    files=list_files(dir_name)
    length=len(files)
    step=length/pool

    file_n_list=[ files[i:i+step] for i in range(0,length,step)]

    p=ThreadPool(pool)
    data=p.map(lambda x: list2feature(x,**kw),file_n_list)
    p.close()
    return np.concatenate(data,axis=0)

def main():
    print "process train data..."

    train_data=dir2feature(TRAIN_DIR,**config)
    np.savetxt(TRAIN_FEATURE,train_data,fmt="%.10f")

    print "process test data..."
    test_data=dir2feature(TEST_DIR,**config)
    np.savetxt(TEST_FEATURE,test_data,fmt="%.10f")

    print "process verify data..."
    verify_data=dir2feature(VERIFY_DIR,**config)
    np.savetxt(VERIFY_FEATURE,verify_data,fmt="%.10f")


if __name__=="__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()


