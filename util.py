#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

"""
some tools function
"""


import wave
from os import listdir
from os.path import isfile, join
from parse import *
import random
import numpy as np

def feature_normalize(x):
    """
    map voice to [-1,1]
    """
    m=np.abs(np.max(x))
    return x/m

def read_wav(file):
    """
    read wav file to np array
    """
    f=wave.open(file,"r")
    raw_data=f.readframes(f.getnframes())
    array=np.fromstring(raw_data,np.short)
    array.shape=-1,2
    array=array.T.astype(float)[0]
    samplerate=f.getframerate()
    f.close()
    return feature_normalize(array),samplerate

def read_wav_cla(file):
    """
    read wav then ret it's signal,rate,class
    """
    array,rate=read_wav(file)
    cla=search("{:d}-{:d}-{:d}",file)[1]
    return array,rate,cla


def list_files(path):
    files=[join(path,f) for f in listdir(path) if isfile(join(path, f))]
    for i in IGNORE_FILES:
        if i in files:
            files.remove(i)
    # random 
    random.shuffle(files)
    return files

def eval_acc(label,pred):
    sample_num=label.shape[0]
    return np.sum(label==pred)*1.0/sample_num

IGNORE_FILES=[".DS_Store"]
TRAIN_DIR="./dataset/train_data"
TEST_DIR="./dataset/test_data"
VERIFY_DIR="./dataset/verify_data"

TRAIN_FEATURE="./output/train_feature.txt"
TEST_FEATURE="./output/test_feature.txt"
VERIFY_FEATURE="./output/verify_feature.txt"

CLASSFIER1="./output/classfier1.pkl"
CLASSFIER2="./output/classfier2.pkl"
NET_MODEL="./output/model.pt"
LOSS_PATH="./output/loss.txt"
GAMMA1=0.0000081
GAMMA2=0.0000081
CLIP_DIS=0.55

config={
    "preemph":0.62,
    "nfft":2048,
    "frame_len":0.025,
    "frame_step":0.01,
    "clip_num":15,
    "cep_num":20,
}

FEATURE_NUM=2*config["cep_num"]*config["clip_num"]


WORD_TABLE={
    0:"语音",
    1:"余音",
    2:"识别",
    3:"失败",
    4:"中国",
    5:"忠告",
    6:"北京",
    7:"背景",
    8:"上海",
    9:"商行",
    10:"复旦",
    11:"饭店",
    12:"Speech",
    13:"Speaker",
    14:"Signal",
    15:"File",
    16:"Print",
    17:"Open",
    18:"Close",
    19:"Project",
}

LABEL_NUM=len(WORD_TABLE)



