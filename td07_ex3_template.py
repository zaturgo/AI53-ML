# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:20:44 2021

@author: LAURI
"""

import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time


class SOM:
    def __init__(self,minv, maxv, inputs, lr=0.1, nlr=0.01):  # lr=learning rate, #nlr=neighbour learning rate
        plt.xlim(minv, maxv)
        plt.ylim(minv, maxv)
        self.Neurons = np.array([[-5, 0], [5, 0]], dtype=float)
        self.Inputs = inputs
        self.Size = len(self.Inputs)
        self.LR = lr
        self.NLR = nlr

    def animate(self, i=0,bool=False):
        if not bool:
            print('__________________')
            print('Point : ' + str(i))
            self.next(i)
        scat.set_offsets(self.Neurons)
        return scat,

    def next(self,i):
        self.calcCoor(self.Inputs[i])
        pass

    def neuroneClose(self, point):
        finalIndex = -1
        min = np.Inf
        for i in range(len(self.Neurons)):
            distance = self.d(point, self.Neurons[i])
            if distance < min:
                min = distance
                finalIndex = i
        return finalIndex

    def winNeuron(self, index, point):
        print('WINNER:' + str(index))
        self.Neurons[index] += self.LR * (point - self.Neurons[index])

    def calcCoor(self, point):
        index = self.neuroneClose(point)
        self.winNeuron(index, point)
        self.neighbourNeurons(point, index)

    def d(self, x, y):
        return np.linalg.norm(x - y)

    def neighbourNeurons(self, point, indexClose):
        print('NEIGHBOURS:')
        for i in range(len(self.Neurons)):
            if i != indexClose:
                self.Neurons[i] += self.NLR * (point - self.Neurons[i])
                print('Neurone:' + str(i))


# Question 1
# data = np.array([(5, 2), (3, 3), (-1, -2)])

# Question 3
np.random.seed(57)
data1 = np.random.uniform(-1, 1, (5, 2)) + np.array([5, 0])
data2 = np.random.uniform(-1, 1, (5, 2)) + np.array([-8, -4])
data3 = np.random.uniform(-1, 1, (5, 2)) + np.array([3, 5])
data4 = np.random.uniform(-1, 1, (5, 2)) + np.array([-6, 7])
data = np.concatenate((data1, data2, data3, data4), axis=0)

fig = plt.figure()  # initialise la figure
x, y = zip(*data)
plt.scatter(x, y, c='red', s=80)

scat = plt.scatter([], [], c='green', s=160, marker='s', edgecolor='black')
som = SOM(-10, 10, data, lr=0.1, nlr=0.01)
som.animate(bool=True);
ani = animation.FuncAnimation(fig, som.animate,frames=len(data), interval=10, repeat=True)
plt.show()
