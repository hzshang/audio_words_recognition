#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

"""
map wav to feature
"""

from util import *
from vad import *
from mfcc import *
import numpy as np

def select_vector_by_dis(vector,N):
    """
    clip vector to N by distance
    """
    v1=vector[1:]
    v2=vector[:-1]
    dist=np.linalg.norm(v1-v2,axis=1)
    maxidx=dist.argsort()[-N:][::-1]
    return vector[maxidx]

def select_vector_by_line(vector,N):
    """
    clip vector linearly
    """
    length=vector.shape[0]
    idx=np.linspace(0,length-1,N).astype(int)
    return vector[idx]

def wav2feature(signal,rate,drop=True,**kw):
    """
    kw: 
        preemph
        nfft
        frame_len
        frame_step
        cep_num
        clip_num

    signal: 1d array, wav signal
    rate: sample rate
    """
    clip_num=kw["clip_num"]
    cep_num=kw["cep_num"]
    begin,end=vad(signal)
    if drop and ((end-begin) > len(signal)*CLIP_DIS or (end-begin) < clip_num*2) :
        # drop it
        return None
    feature=mfcc(signal[begin:end],rate,**kw)
    d_feature=dealt(feature,2)

    if feature.shape[0] >clip_num:
        s_feature=select_vector_by_line(feature,clip_num)
        s_dfeature=select_vector_by_line(d_feature,clip_num)
        feat=np.append(s_feature.flatten(),s_dfeature.flatten()).flatten()
        return feat
    elif drop:
        return None
    else:
        return np.ones(2*clip_num*cep_num)

