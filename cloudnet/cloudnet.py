import numpy as np
import time
import sys,os
from cloudnet.cuda import gnp
from cloudnet.func import *


class cloudet:

    def __init__(self,netin):
        self.netlist = netin
        self.netlayer = len(self.netlist)


    def setup(self,optimizer,activation,shape=(1,28,28),mode="normal",dtype=gnp.float32):
        self.optimizer = optimizer      # 優化器
        self.activation = activation    # 激活函數
        self.mode = mode                # 初始化模式  
        self.dtype = dtype              # 數據型態
        self.shape = shape              # 遮罩

        for i in range(self.netlayer):
            self.netlist[i].init(shape,dtype)



    def forward(self,input):
        for i in range(self.netlayer):
            input = self.netlist[i].forward(input)
        return input
    

    def backward(self,input):
        for i in range(self.netlayer):
            input = self.netlist[i].backward(input)
        return input

    

    def train(self,input,label):
        input = gnp.asarray(input)
        forward_out = self.forward(input)
        label = gnp.asarray(label)
        loss = cross_entropy(forward_out,label)
        error = (forward_out-label)/label.shape[0]
        loss.backward(error)

        return loss



    
