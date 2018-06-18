#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 hzshang <hzshang15@gmail.com>
#
# Distributed under terms of the MIT license.

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import torch.utils.data as Data
import torch.nn.functional as F
from util import *
from nn_classfier import Net
import numpy as np

def get_data_loader(file):
    data=np.loadtxt(file)
    x=data[:,:-1]
    # change x to 20x30 matrix 
    x.shape=-1,1,config["cep_num"],config["clip_num"]*2
    x=torch.from_numpy(x).float()
    y=data[:,-1].astype(int)
    y=torch.from_numpy(y).long()

    torch_dataset=Data.TensorDataset(x,y)
    loader = Data.DataLoader(
        torch_dataset,              # torch TensorDataset format
        batch_size=64,              # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=10,             # 多线程来读数据
    )
    return loader

def train(net,criterion,optimizer,loader,epoch):
    net.train()
    running_loss=0
    for batch_idx,(data,target) in enumerate(loader):
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()


def test(net,loader):
    net.eval()
    sum=0
    total=0
    with torch.no_grad():
        for data,target in loader:
            output=net(data)
            result=torch.argmax(output,1)
            total+=target.size(0)
            sum+=(result == target).sum().item()

    acc=1.0*sum/total
    return acc

def main():
    # net
    net=Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # load data
    train_loader=get_data_loader(TRAIN_FEATURE)
    test_loader=get_data_loader(TEST_FEATURE)
    verify_loader=get_data_loader(VERIFY_FEATURE)

    # record loss
    train_acc=[]
    test_acc=[]
    verify_acc=[]

    for epoch in range(40):
        train(net,criterion,optimizer,train_loader,epoch)
        # calc acc
        acc_train=test(net,train_loader)
        acc_test=test(net,test_loader)
        acc_verify=test(net,verify_loader)

        train_acc.append(acc_train)
        test_acc.append(acc_test)
        verify_acc.append(acc_verify)
        print '-----------'
        print '[%d] %s acc:%.3f'%(epoch+1,"train",100.0*acc_train)
        print '[%d] %s acc:%.3f'%(epoch+1,"test",100.0*acc_test)
        print '[%d] %s acc:%.3f'%(epoch+1,"verify",100.0*acc_verify)

    print('Finished Training')
    net.save_file(NET_MODEL)
    acc=np.array([train_acc,test_acc,verify_acc])
    np.savetxt(LOSS_PATH,acc,fmt="%.10f")

if __name__=="__main__":
    main()
