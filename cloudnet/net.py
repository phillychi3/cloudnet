import numpy as np
from activation_function import *


class DNN():

    def __init__(self):
        self.layers = []
        self.params = {}
        self.grads = {}
        self.loss = None
        self.loss_layer = None
        self.activation = ReLu


    def train():
        ...

    def predict():
        ...

    def output():
        ...

    def add(self,layer):
        self.layers.append(layer)

    def init(self):
        for layer in self.layers:
            layer.init(self.params)

    def forward():
        ...

    def backward():
        ...
    
    # 輸入層
    class input_layer():
        ...

    # 隱藏層
    class hidden_layer():
        ...
    
    # 全連接層
    class full_connect_layer():
        ...
    
