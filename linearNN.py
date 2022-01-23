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

    def __init__(self, num=10, dim=2, cls=2):
        self.data = np.random.random((cls, num, dim))
        self.label = np.ones((cls, num, 1), dtype=np.int32)
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


class DataGenReg(DataGen):
    """description"""

    def __init__(self, num=50):
        """docstring for __init__"""
        slope = 4
        self.data = np.array([np.arange(num)], dtype=np.float16).reshape(-1, 1)
        self.label = (self.data * slope + np.random.random((num, 1)) * (num/2))
        self.data /= self.data.max()
        self.label /= self.label.max()
        #randSelect = num
        #rs = np.random.randint(np.ones(randSelect)*randSelect)
        #self.data = self.data[rs]
        #self.label = self.label[rs]

    def show(self):
        """docstring for show"""
        plt.scatter(self.data, self.label)
        plt.show()


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
        #print("label", self.label)
        #print("y", self.y)
        return 2 * (self.y - self.label)

    def __call__(self, y, label):
        """docstring for __call__"""
        self.label = label
        self.y = y
        self.outputs = (np.square(self.y-self.label)).mean()
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
    bias = 0

    def __init__(self, dim: int, bias: bool):
        self.weight = np.random.randn((dim))
        if bias:
            self.bias = np.random.random(1)[0]

    def __call__(self, x):
        """
            o = WX+B
        """
        #print("x", x)
        #print("w", self.weight)
        return np.dot(x, self.weight) + self.bias

    def differenciate_wrt_weight(self, x, dO):
        """docstring for differentiate"""
        # print("x",x)
        # print("dO",dO)
        self.dw = dO.T.dot(x)[0]
        #print("dw", self.dw)
        #print('w', self.weight)
        # return self.weight

    def update(self, lr):
        """docstring for update"""
        #print("dw", self.dw)
        self.weight -= lr*self.dw
        #self.bias -= db


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

    def __init__(self, in_dim: int, out_dim: int, bias: bool, activation=None):
        self.neurons = [Neuron(dim=in_dim, bias=bias) for _ in range(out_dim)]
        self.inputs = None
        self.outputs = np.zeros((out_dim))
        self.activation = activation

    def __call__(self, x):
        """docstring for __call__"""
        self.inputs = x
        self.outputs = np.array([[neuron(p) for neuron in self.neurons] for p
                                 in self.inputs])
        if self.activation is not None:
            self.outputs = self.activation(self.outputs)
        return self.outputs

    def differentiate(self, dO, lr):
        #print("dO", dO)
        if self.activation is not None:
            #print("da", self.activation.differentiate())
            #print("dOda", dO *self.activation.differentiate())
            dO = dO * self.activation.differentiate()
            # print("dO",dO)
        for neural in self.neurons:
            neural.differenciate_wrt_weight(self.inputs, dO)
            # print("dO",dO)
        #print(np.array([neural.weight for neural in self.neurons]))
        di = dO.dot(np.array([neural.weight for neural in self.neurons]))
        # print(di)
        self.update(lr)
        return di

    def update(self, lr):
        """docstring for update"""
        for neural in self.neurons:
            neural.update(lr)


class NeuralNetwork():
    """docstring for NeuralNetwork"""

    def __init__(self, lr=0.1, data=DataGenReg(num=100)):
        self.data = data
        self.lr = lr
        self.tr_x, self.tr_y, self.te_x, self.te_y = self.data.split()
        self.lossFunc = MeanSquareError()

        self.sequential = [
            LinearLayer(in_dim=self.tr_x.shape[1], out_dim=10, bias=False,
                        activation=Sigmoid()),
            LinearLayer(in_dim=10, out_dim=1, bias=False,
                        activation=Sigmoid()),
        ]

        # self.sequential = [
        #    LinearLayer(in_dim=1, out_dim=1, bias=False, activation=None),
        #    ]
        # self.sequential = [
        #    LinearLayer(in_dim=self.tr_x.shape[1], out_dim=2, bias=True,
        #        activation=None),
        #    LinearLayer(in_dim=2, out_dim=1, bias=True, activation=None),
        # ]

    def forward(self, x):
        """docstring for forward"""
        #x = self.fcl1(x)
        #x = self.activate(x)
        #x = self.fcl2(x)
        #o = self.activate(x)
        for layer in self.sequential:
            x = layer(x)
            # print(x)
        return x

    def backward(self, y):
        """docstring for backward"""
        d_loss_d_y_pre = self.lossFunc.differentiate()
        dO = d_loss_d_y_pre
        for layer in self.sequential[::-1]:  # Traverse the layers reversely
            # print(layer)
            dO = layer.differentiate(dO.copy(), self.lr)
        # exit()

    def train(self, epochs=2000):
        """docstring for train"""
        for epoch in range(epochs):
            output = self.forward(self.tr_x)
            loss = self.lossFunc(output, self.tr_y)
            self.backward(loss)
            if epoch % 50 == 0:
                self.lr *= 0.8
            print("epoch: {} loss: {} lr: {}".format(epoch, loss, self.lr))

    def test(self):
        """docstring for test"""
        return

    def vis(self):
        """docstring for vis"""
        x = np.array([np.linspace(0, 1, 30)]).reshape(-1, 1)
        y = self.forward(x)
        plt.plot(x, y)
        plt.scatter(self.tr_x, self.tr_y)
        plt.show()


nn = NeuralNetwork()
nn.train()
nn.vis()
