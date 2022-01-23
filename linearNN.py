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
        self.label = np.ones((cls, num), dtype=np.int32)
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
    result = 0
    def __call__(self, x):
        """docstring for __call__"""
        self.result = 1 / (1+np.exp(-x))
        return self.result

    def differentiate(self):
        """docstring for differentiate"""
        return self.result * (1 - self.result)

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

    def differentiate(self):
        """docstring for differentiate"""
        return 2 * (self.outputs - self.label)

    def __call__(self, y, label):
        """docstring for __call__"""
        self.label = label
        self.outputs = np.sum(np.square(y-self.label))
        return self.outputs


class CrossEntropyLoss():
    """docstring for CrossEntropyLoss"""
    def __init__(self, arg):
        self.arg = arg

    def onehot_label(self, l):
        """
            Convert labels to onehot label:
            [1, 0, 0, 1] => [[0, 1], [1, 0], [1, 0], [0, 1]]
        """
        ele = np.unique(l)
        ohl = np.zeros((l.shape[0], ele.shape[0]))
        for col in range(l.shape[0]):
            ohl[col][l[col]] = 1
        return ohl

    def __call__(self, label):
        """docstring for __call__"""
        self.label = label
        self.outputs = np.sum(np.square(y-self.onehot_label(label)))
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
        """
            o = WX+B
        """
        o = np.dot(x, self.weight) + self.bias
        return o[0] # Get value from numpy array

    def differentiate_wrt_weight(self):
        """docstring for differentiate"""
        return self.weight


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
    def __init__(self, in_dim:int, out_dim: int, bias:bool, activation=None):
        self.neurons = [Neuron(dim=in_dim, bias=bias) for _ in range(out_dim)]
        self.outputs = np.zeros((out_dim))
        self.activation = activation

    def __call__(self, x):
        """docstring for __call__"""
        self.outputs = np.array([[neuron(p) for neuron in self.neurons] for p in x])
        if self.activation is not None:
            return self.activation(self.outputs)
        else:
            return self.outputs

    def differentiate(self, dO):
        print(dO)
        if self.activation is not None:
            print(self.activation.differentiate())
        print(dO)
        exit(0)
        return dO

    def update(self):
        """docstring for update"""
        return


class NeuralNetwork():
    """docstring for NeuralNetwork"""
    def __init__(self, data=DataGen(num=10)):
        self.tr_x, self.tr_y, self.te_x, self.te_y = data.split()
        self.lossFunc = MeanSquareError()
        self.fcl1 = LinearLayer(in_dim=self.tr_x.shape[1], out_dim=2, bias=True)
        self.fcl2 = LinearLayer(in_dim=2, out_dim=1, bias=True)
        self.activate = Sigmoid()

        self.sequential = [
            LinearLayer(in_dim=self.tr_x.shape[1], out_dim=2, bias=True, activation=Sigmoid()),
            LinearLayer(in_dim=2, out_dim=1, bias=True, activation=Sigmoid()),
        ]

    def forward(self, x):
        """docstring for forward"""
        #x = self.fcl1(x)
        #x = self.activate(x)
        #x = self.fcl2(x)
        #o = self.activate(x)
        for layer in self.sequential:
            x = layer(x)
        return x

    def backward(self, y):
        """docstring for backward"""
        d_loss_d_y_pre = self.lossFunc.differentiate()
        dO = d_loss_d_y_pre
        for layer in self.sequential[::-1]: # Traverse the layers reversely
            print(layer)
            dO = layer.differentiate(dO)
            print(dO)
        exit()

    def train(self, epochs=100):
        """docstring for train"""
        for epoch in range(epochs):
            output = self.forward(self.tr_x)
            loss = self.lossFunc(output, self.tr_y)
            self.backward(loss)
            print("epoch: {} loss: {}".format(epoch, loss))

    def test(self):
        """docstring for test"""
        return

nn = NeuralNetwork()
nn.train()


