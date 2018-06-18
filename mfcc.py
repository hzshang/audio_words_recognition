#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

"""
calc mfcc feature
"""

from __future__ import division
import numpy as np
import math
import logging
from scipy.fftpack import dct

def mfcc(signal,samplerate,window_fun=lambda x: np.ones((x,)),nfilter=26,ceplifter=22,**kw):
    """
    kw: preemph
        frame_len
        frame_step
        nfft
        cep_num

    signal a array with shape (N,)
    """
    feature,energy=mfbank(signal,samplerate,kw["preemph"],kw["frame_len"],\
        kw["frame_step"],kw["nfft"],nfilter,window_fun)
    feature=np.log(feature)
    feature = dct(feature, type=2, axis=1, norm='ortho')[:,:kw["cep_num"]]
    feature = lifter_vector(feature,ceplifter)
    feature[:,0]=np.log(energy)
    return feature

def mfbank(signal,samplerate,preemph,frame_len,frame_step,nfft,nfilter,window_fun):
    signal=preemphasis(signal,preemph)
    frame=clip_frame(signal,frame_len*samplerate,frame_step*samplerate,window_fun)
    pframe=power_frame(frame,nfft)
    energy=np.sum(pframe,1)
    energy=np.where(energy==0,np.finfo(float).eps,energy)
    mel_filter=get_mel_filter(nfilter,nfft,samplerate)
    feature=np.dot(pframe,mel_filter.T)
    feature=np.where(feature==0,np.finfo(float).eps,feature)
    return feature,energy


def preemphasis(signal,preemph):
    """
    preemphasis singal
    """
    dev=signal[1:]-signal[:-1]*preemph
    return np.append(signal[0],dev)

def clip_frame(signal,frame_len,frame_step,window_fun):
    length=len(signal)
    frame_len=int(frame_len)
    frame_step=int(frame_step)
    if frame_len > length:
        num_frame=1
    else:
        num_frame=1+int(math.ceil((1.0*length-frame_len)/frame_step))
    padlen=int((num_frame-1)*frame_step+frame_len-length)
    signal=np.concatenate((signal,np.zeros(padlen)))
    window=window_fun(frame_len)
    frames=rolling_window(signal,frame_len,frame_step)
    return frames*window

def rolling_window(a,window,step):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

def power_frame(frame,len_fft):
    if frame.shape[1] > len_fft:
        logging.warning('frame length (%d) is greater than FFT size (%d)'%(frame.shape[1],len_fft))
    fft_signal=np.fft.rfft(frame,len_fft)
    return 1/len_fft*np.square(np.absolute(fft_signal))

def hz2mel(f):
    return 2595*np.log10(1+f/700.0)

def mel2hz(m):
    return (10**(m/2595.)-1)*700

def get_mel_filter(nfilter,nfft,samplerate):
    highfre=samplerate/2
    lowfre=0
    highmel=hz2mel(highfre)
    lowmel=hz2mel(lowfre)
    melpoints=np.linspace(lowmel,highmel,nfilter+2)
    dev = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)
    ret = np.zeros([nfilter,nfft//2+1])
    # ret 26*257
    for j in range(0,nfilter):
        for i in range(int(dev[j]), int(dev[j+1])):
            ret[j,i] = (i - dev[j]) / (dev[j+1]-dev[j])
        for i in range(int(dev[j+1]), int(dev[j+2])):
            ret[j,i] = (dev[j+2]-i) / (dev[j+2]-dev[j+1])
    return ret

def lifter_vector(vector,level):
    if level >0:
        array=np.arange(vector.shape[1])
        l=1+(level/2.0)*np.sin(np.pi*array/level)
        return l*vector
    else:
        return vector

def dealt(vector,N):
    num_feature=vector.shape[0]
    de=N*(N+1)*(2*N+1)/3#2*(i**2+2**2+3**2+....N**2)
    dealt_v=np.zeros(vector.shape)
    pad_v=np.pad(vector,((N,N),(0,0)),mode="edge")
    for i in range(num_feature):
        dealt_v[i]=np.dot(np.arange(-N,N+1),pad_v[(N+i)-N:(N+i)+(N+1)])/de
    return dealt_v


"""
## test module
#a=np.arange(100)
#s=clip_frame(a,10,3,window_fun=lambda x:np.ones(x)*2)
a=np.linspace(0,100,10000)
#mfbank(signal,samplerate,preemph,frame_len,frame_step,nfft,nfilter,window_fun)
f=mfcc(a,10000)
d_f=dealt(f,2)
feature=logfbank(a,10000)
print feature[1:3,:]
"""
