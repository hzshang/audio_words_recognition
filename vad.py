#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Voice_activity_detection
"""
import numpy as np

def vad(voice):
    """
    voice: np.array with shape (n,)
    """
    xx=voice**2
    length=len(voice)
    step=length/40
    m=np.ones(300)
    cc=np.convolve(xx,m,"same")
    noise=(np.mean(cc[:300])+np.mean(cc[-300:]))/2
    threhold=np.max(cc)/100
    t=np.where(cc>threhold)[0]
    begin=t[0]
    end=t[-1]
    begin_t=np.where(cc[:begin]<2.5*noise)[0]
    if begin_t.shape[0]>0 and begin-begin_t[-1]< 5*step:
        begin=begin_t[-1]
    else:
        begin=np.clip(begin-step,0,length)

    end_t=np.where(cc[end:]<3*noise)[0]
    if end_t.shape[0]>0 and end_t[0] < 5*step:
        end=end_t[0]+end
    else:
        end=np.clip(end+step,0,length)
    return begin,end

"""
# test module
import matplotlib.pyplot as plt
from util import *
acc=np.loadtxt("loss.txt")
plt.plot(acc[0],color="red",label="train")
plt.plot(acc[1],color="blue",label="test")
plt.plot(acc[2],color="green",label="verify")
plt.legend(loc='lower right')

plt.show()
# # signal,rate=read_wav("dataset/test_data/14300270021-11-19.wav")
# fig=plt.figure(figsize=(8,4),dpi=80)

# i=0
# for f in [#"/Users/hzshang/Downloads/15307130079/15307130079-06-11.wav",
# # "/Users/hzshang/Downloads/15307130079/15307130079-04-11.wav",
# "/Users/hzshang/Downloads/15307130140/15307130140-09-18.wav",
# "/Users/hzshang/Downloads/15307130140/15307130140-12-09.wav"]:
#     i=i+1
#     signal,_=read_wav(f)
#     begin,end=vad(signal)
#     fig.add_subplot(1,2,i)
#     plt.plot(signal)
#     plt.axvline(x=begin,color="red")
#     plt.axvline(x=end,color="red")
#     plt.text(begin,0.8,'a')
#     plt.text(end,0.8,'b')
#     plt.axvline(x=begin-3000,color="blue")
#     plt.axvline(x=end+5000,color="blue")
#     plt.text(begin-5000,0.8,'a\'',color="blue")
#     plt.text(end+5000,0.8,'b\'',color="blue")

# plt.show()
"""

