# Copyright (c) SheldonFung All Rights Reserved.
# File Name: linearNN.py
# Author: Sheldon Fung
# email: sheldonvon@outlook.com

import numpy as np
import matplotlib.pyplot as plt

"""
             x_1      nerual_11
    X   =>   x_2  =>  nerual_12 =>  neural_21 => Y

"""


class Sigmoid():
    """docstring for Sigmoid"""

    def __call__(self, x):
        """docstring for __call__"""
        return 1 / (1+np.exp(-x))

    def show(self):
        """docstring for show"""
        x = np.linspace(-10, 10, 100)
        y = self(x)
        plt.plot(x, y)
        plt.show()


class MeanSquareError():
    """docstring for MeanSquareError"""
    def __init__(self):
        self.outputs = 0

    def __call__(self, y, label):
        """docstring for __call__"""
        self.outputs = np.sum(np.square(y-label))
        return self.outputs


class Neural():
    """
        The neural is the unit to perform mathematical
        operations. The shape of it therefore needs to
        match with the input data shape.
    """
    def __init__(self, inputs: int, dim: int, bias: bool):
        self.weight = np.random.random(dim, inputs)
        if bias:
            self.bias = np.random.random(inputs)

    def __call__(self, x):
        """docstring for __call__"""
        return np.dot(x, self.weight) + self.bias

    def update(self, dw, db):
        """docstring for update"""
        self.weight -= dw
        self.bias -= db


class LinearLayer():
    """docstring for LinearLayer"""
    def __init__(self, inputs:int, dim: int, bias:bool, num=2):
        self.neurals = [Neural(inputs=inputs, dim=dim, bias=bias) for _ in range(num)]
        self.outputs = np.zeros(num)

    def __call__(self, x):
        """docstring for __call__"""
        self.outputs = np.array([neural(x) for neural in self.neurals])
        return self.outputs

    def update(self):
        """docstring for update"""
        return


class DataGen():
    """docstring for DataGen"""
    def __init__(self, num=10, dim=1, cls=2):
        self.data = np.random.random((cls, num, dim))
        self.label = np.ones((cls, num))
        # Separate data
        for each in range(cls):
            self.data[each] *= (each+1)
            self.label[each] *= (each)
        randSelect = num * cls
        rs = np.random.randint(np.ones(randSelect)*randSelect)
        self.data = np.concatenate(self.data)[rs]
        self.label = np.concatenate(self.label)[rs]

    def split(self, rate=0.8):
        """docstring for split"""
        sp = int(self.label.shape[0] * rate)
        return self.data[:sp], self.label[:sp], self.data[sp:], self.label[sp:]


class NeuralNetwork():
    """docstring for NeuralNetwork"""
    def __init__(self, data=DataGen(num=100)):
        self.tr_x, self.tr_y, self.te_x, self.te_y = data.split()
        self.lossFunc = MeanSquareError()
        self.fcl1 = LinearLayer(inputs=self.tr_x.shape[0], dim=, bias=True, num=2)
        self.fcl2 =  LinearLayer(inputs=self.fcl1.outputs.shape[0], bias=True,num=1)
        self.activation = Sigmoid()

    def forward(self, x):
        """docstring for forward"""
        x = self.fcl1(x)
        x = self.fcl2(x)
        return self.activation(x)

    def backward(self, y):
        """docstring for backward"""
        return

    def train(self, epochs=100):
        """docstring for train"""
        for epoch in range(epochs):
            output = self.forward(self.tr_x)
            loss = self.lossFunc(output, tr_y)
            self.backward(loss)
            print("epoch: {} loss: {}".format(epoch, loss))

    def test(self):
        """docstring for test"""
        return

nn = NeuralNetwork()
nn.train()


