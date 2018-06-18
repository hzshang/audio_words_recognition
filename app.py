#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

"""
application

"""

from util import *
from classfier import *
from nn_classfier import *
from voice import *
import pyaudio
import wave
import numpy as np
import sys
import os
import signal

def signal_handler(signal,frame):
    print ""
    sys.exit(0)

def record():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 2
    p = pyaudio.PyAudio()
    frames = []
    stream = p.open(format=FORMAT,
		    channels=CHANNELS,
		    rate=RATE,
		    input=True,
		    frames_per_buffer=CHUNK)
    print("* recording *")
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording *")
    stream.stop_stream()
    stream.close()
    p.terminate()
    s="".join(frames)
    sig=np.fromstring(s,np.short)
    sig=sig.astype(float)
    sig=feature_normalize(sig)
    return sig,RATE

def main():
    signal.signal(signal.SIGINT, signal_handler)
    if len(sys.argv) == 1 or sys.argv[1] != 'net':
        cla=classfier()
        cla.load_svm(CLASSFIER)
    else:
        cla=Net()
        cla.load_file(NET_MODEL)

    first=True
    while True:
        if first:
            first=False
            raw_input("Type Enter key to begin...")
        else:
            raw_input("Type Enter key to continue...")
        sig,rate=record()
        feat=wav2feature(sig,rate,drop=False,**config)
        result=cla.predict(np.array([feat]))[0]
        print "word: %s"%WORD_TABLE[result]

if __name__=="__main__":
    main()





