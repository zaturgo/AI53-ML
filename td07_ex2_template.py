# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:22:22 2021

@author: LAURI
"""


import numpy as np
import matplotlib
import time
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class KMeans:
    def __init__(self,K,minv,maxv,inputs):
        plt.xlim(minv,maxv)
        plt.ylim(minv,maxv)
        self.K=K
        self.Centers=np.array([[0,-4],[0,8]]) #np.random.uniform(minv,maxv,(K,2))
        self.InitialCenters=self.Centers.copy()
        self.Inputs=inputs
        self.Assigned0=arr = [[],[]]
        self.Size=len(self.Inputs)
        self.C=np.zeros(self.Size,dtype=int)

    def animate(self,i=0,bool=False):#i index du point actuel
        self.Assigned0=[[],[]]
        self.next(i)
        first_scat.set_offsets(self.InitialCenters)
        scat.set_offsets(self.Centers)
        return scat,first_scat

    def next(self,i):
        for j in range(len(self.Inputs)):
            self.assignToCenter(j)
        self.updateCenters(0)
        self.updateCenters(1)
        print('Nouveaux centres:',self.Centers)

    def findIndex(self,array,value):
        for i in range(len(array)):
            if tuple(array[i])==tuple(value):
                return i
        return -1

    def assignToCenter(self,i):
        distanceA = self.d(self.Inputs[i],self.Centers[0])
        distanceB = self.d(self.Inputs[i],self.Centers[1])
        if distanceA<=distanceB:
            self.Assigned0[0].append(self.Inputs[i])
        else:
            self.Assigned0[1].append(self.Inputs[i])


    def updateCenters(self,type):
        sumX = 0
        sumY = 0
        totalX = len(self.Assigned0[type])
        for i in range(len(self.Assigned0[type])):
            sumX += self.Assigned0[type][i][0]
            sumY += self.Assigned0[type][i][1]
        self.Centers[type][0] = sumX / totalX
        self.Centers[type][1] = sumY / totalX


    def d(self,x,y):
        return np.linalg.norm(x-y)


# Dataset
data=np.array([(1,10),(1.5,2),(1,6),(2,1.5),(2,10),(3,2.5),(3,6),(4,2)])
classes=np.array([1,2,1,2,1,2,1,2],dtype=np.int)


# Display the dataset
color_names=['red','blue']
colors=[color_names[c-1] for c in classes]

fig = plt.figure() # initialise la figure
x,y=zip(*data)
plt.scatter(x,y,c=colors)
# plt.show()

first_scat = plt.scatter([], [],c='yellow',s=80,marker='s',edgecolor='black')
scat = plt.scatter([], [],c='green',s=160,marker='s',edgecolor='black')

kmeans=KMeans(2,-11,11,data)
kmeans.animate(bool=True)

ani = animation.FuncAnimation(fig, kmeans.animate, interval=2000, repeat=True)
plt.show()
