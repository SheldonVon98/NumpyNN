# Copyright (c) SheldonFung All Rights Reserved.
# File Name: linearNN.py
# Author: Sheldon Fung
# email: sheldonvon@outlook.com

import numpy as np
import matplotlib.pyplot as plt

"""
             x_1      neuron_11
    X   =>   x_2  =>  neuron_12 =>  neuron_21 => Y

"""


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


class Neuron():
    """
        The neuron is the unit to perform mathematical
        operations. The shape of it therefore needs to
        match with the input data shape.
    """
    def __init__(self, dim: int, bias: bool):
        self.weight = np.random.random((dim))
        if bias:
            self.bias = np.random.random(1)

    def __call__(self, x):
        """docstring for __call__"""
        print(x.shape)
        print(self.weight.shape)
        print(np.dot(x,self.weight))
        return np.dot(x, self.weight) + self.bias

    def update(self, dw, db):
        """docstring for update"""
        self.weight -= dw
        self.bias -= db


class LinearLayer():
    """
        Docstring for LinearLayer

        in_dim: specify the input dimention of the data.
        out_fim: specify the output dimention of the data.
        bias: enable bias.

        The shape of the input data should be:
            (batch_size, in_dim)
        The output shape would be:
            (batch_size, out_dim)
    """
    def __init__(self, in_dim:int, out_dim: int, bias:bool, num=2):
        self.neurons = [Neuron(dim=in_dim, bias=bias) for _ in range(out_dim)]
        self.outputs = np.zeros((out_dim))

    def __call__(self, x):
        """docstring for __call__"""
        self.outputs = np.array([[neuron(p) for neuron in self.neurons] for p in x])
        return self.outputs

    def update(self):
        """docstring for update"""
        return


class NeuralNetwork():
    """docstring for NeuralNetwork"""
    def __init__(self, data=DataGen(num=10)):
        self.tr_x, self.tr_y, self.te_x, self.te_y = data.split()
        self.lossFunc = MeanSquareError()
        self.fcl1 = LinearLayer(in_dim=self.tr_x.shape[1], out_dim=2, bias=True)
        self.fcl2 = LinearLayer(in_dim=2, out_dim=2, bias=True, num=1)
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


