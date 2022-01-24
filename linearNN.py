# Copyright (c) SheldonFung All Rights Reserved.
# File Name: linearNN.py
# Author: Sheldon Fung
# email: sheldonvon@outlook.com

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
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


class DataGenRegSin(DataGen):
    """description"""

    def __init__(self, num=50):
        """docstring for __init__"""
        self.data = np.linspace(0, 3*np.pi, num).reshape(-1, 1)
        self.label = np.array([np.sin(self.data)], dtype=np.float16).reshape(-1, 1)
        self.data /= self.data.max()
        self.label -= self.label.min()
        self.label /= self.label.max()

    def show(self):
        """docstring for show"""
        plt.scatter(self.data, self.label)
        plt.show()


class ReLu():
    """docstring for ReLu"""
    # Activation function output
    afo = 0
    # Derivative of activation function output
    dafo = 0

    def __call__(self, x):
        """docstring for __call__"""
        self.afo = np.maximum(x, 0)
        return self.afo

    def differentiate(self):
        """docstring for differentiate"""
        self.dafo = self.afo.copy()
        self.dafo[self.dafo<=0] = 0
        self.dafo[self.dafo>0] = 1
        return self.dafo

    def show(self):
        """docstring for show"""
        x = np.linspace(-10, 10, 100)
        y = self(x)
        dy = self.differentiate()
        plt.plot(x, y)
        plt.plot(x, dy)
        plt.show()

#ReLu().show()
#exit(0)

class Sigmoid():
    """docstring for Sigmoid"""
    # Activation function output
    afo = 0
    # Derivative of activation function output
    dafo = 0

    def __call__(self, x):
        """docstring for __call__"""
        self.afo = 1 / (1+np.exp(-x))
        return self.afo

    def differentiate(self):
        """docstring for differentiate"""
        self.dafo = self.afo * (1 - self.afo)
        return self.dafo

    def show(self):
        """docstring for show"""
        x = np.linspace(-10, 10, 100)
        y = self(x)
        dy = self.differentiate()
        plt.plot(x, y)
        plt.plot(x, dy)
        plt.show()

#Sigmoid().show()
#exit(0)

class MeanSquareLoss():
    """
 linearlarity       Mean Square Loss
    """

    # Derivative of Loss with respect 
    # to the output of the final layer
    dl_dypred = 0
    # Data label
    ground_truth = 0
    # The output of the final layer
    y_pred = 0

    def differentiate(self):
        """docstring for differentiate"""
        self.dl_dypred = 2 * (self.y_pred - self.ground_truth)
        return self.dl_dypred

    def __call__(self, y_pred, gt):
        """docstring for __call__"""
        self.ground_truth = gt
        self.y_pred = y_pred
        loss = (np.square(self.y_pred-self.ground_truth)).mean()
        return loss

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
    inputs = 0

    def __init__(self, dim: int, bias: bool):
        self.weight = np.random.randn(dim, 1)
        if bias:
            self.bias = np.random.random(1)

    def __call__(self, x):
        """
            Y = WX+B

            x       => (dim)
            weight  => (dim)
        """
        #print("b", self.bias)
        #print("x", x.shape)
        #print("w", self.weight.shape)
        #print("b", self.bias.shape)
        z = self.weight.T.dot(x) + self.bias
        #print(z)
        return z[0]

    def differenciate(self, x, dO, idx):
        """docstring for differentiate"""
        #print("x",x.shape)
        #print("dO",dO.shape)
        #print("dO", dO)
        #print(idx)
        #print("x.T.dot(dO)", x.T.dot(dO).shape)
        self.dw = x.T.dot(dO)[:, idx].reshape(-1, 1)
        self.db = np.sum(dO, axis=0)[idx]
        #print(dO)
        #print(dO.shape)
        #print("b", self.bias.shape)
        #print("dw", self.dw.shape)
        #print("db", self.db)
        #exit(0)
        #print("dw", self.dw)
        #print('w', self.weight.shape)
        # return self.weight
        #print("\n")

    def update(self, lr):
        """docstring for update"""
        #print("w", self.weight.shape)
        #print("dw", self.dw.shape)
        self.weight -= lr*self.dw
        self.bias -= lr*self.db
        #print(self.bias)

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
        #print("\nop", self.outputs.shape)
        if self.activation is not None:
            self.outputs = self.activation(self.outputs)
        return self.outputs

    def differentiate(self, dO, lr):
        #print("\n\ndO__", dO)
        if self.activation is not None:
            #print("da", self.activation.differentiate())
            #print("dOda", dO *self.activation.differentiate())
            #print("dO", dO.shape)
            #print("ad", self.activation.differentiate().shape)
            dO *= self.activation.differentiate()
            #print("dO",dO.shape)
        for idx, neural in enumerate(self.neurons):
            neural.differenciate(self.inputs, dO, idx)
            # print("dO",dO)
        #print(np.array([neural.weight for neural in self.neurons]))
        #print("ns", np.array([neural.weight[:, 0] for neural in self.neurons]).shape)

        di = dO.dot(np.array([neural.weight[:, 0] for neural in self.neurons]))
        # print(di)
        self.update(lr)
        return di

    def update(self, lr):
        """docstring for update"""
        for neural in self.neurons:
            neural.update(lr)


class NeuralNetwork():
    """docstring for NeuralNetwork"""

    def __init__(self, lr=0.0005, data=DataGenRegSin(num=100)):
        self.data = data
        self.lr = lr
        self.tr_x, self.tr_y, self.te_x, self.te_y = self.data.split()
        self.lossFunc = MeanSquareLoss()

        self.sequential = [
            LinearLayer(in_dim=1, out_dim=50, bias=True, activation=ReLu()),
            LinearLayer(in_dim=50, out_dim=1, bias=True, activation=None),
        ]
        self.vis_init()

    def forward(self, x):
        """docstring for forward"""
        for layer in self.sequential:
            x = layer(x)
        return x

    def backward(self, y):
        """docstring for backward"""
        d_loss_d_y_pre = self.lossFunc.differentiate()
        dO = d_loss_d_y_pre
        for layer in self.sequential[::-1]:  # Traverse the layers reversely
            #print(layer)
            dO = layer.differentiate(dO.copy(), self.lr)
        # exit()

    def train(self, epochs=20000):
        """docstring for train"""
        #self.fig.canvas.draw()
        for epoch in range(epochs):
            output = self.forward(self.tr_x)
            loss = self.lossFunc(output, self.tr_y)
            self.backward(loss)
            self.vis()
            if epoch > 2000:
                if epoch % 100 == 0:
                    self.lr *= 0.98
            print("epoch: {} loss: {} lr: {}".format(epoch, loss, self.lr))

    def test(self):
        """docstring for test"""
        return

    def vis_init(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        x = np.array([np.linspace(0, 1, 30)]).reshape(-1, 1)
        y = self.forward(x)
        self.ax.scatter(self.tr_x, self.tr_y, animated=False)
        self.ln = self.ax.plot(x, y, animated=True)[0]
        plt.show(block=False)
        plt.pause(0.1)
        # get copy of entire figure (everything inside fig.bbox) sans animated artist
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        # draw the animated artist, this uses a cached renderer
        self.ax.draw_artist(self.ln)
        # show the result to the screen, this pushes the updated RGBA buffer from the
        # renderer to the GUI framework so you can see it
        self.fig.canvas.blit(self.fig.bbox)

    def vis(self):
        """docstring for vis"""
        x = np.array([np.linspace(0, 1, 30)]).reshape(-1, 1)
        y = self.forward(x)
        self.fig.canvas.restore_region(self.bg)
        # update the artist, neither the canvas state nor the screen have changed
        self.ln.set_ydata(y)
        # re-render the artist, updating the canvas state, but not the screen
        self.ax.draw_artist(self.ln)
        # copy the image to the GUI state, but screen might not be changed yet
        self.fig.canvas.blit(self.fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        self.fig.canvas.flush_events()

nn = NeuralNetwork()
nn.train()
#nn.vis()
plt.show()
