#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:58:01 2021

@author: Alfonso S.A
"""

import os  #Module to import data
os.chdir("//home//usuario//Desktop//Simbolos")
os.getcwd()
import matplotlib.pyplot as plt
import numpy            
import pandas as pd            
import random
import math
from scipy.stats import skew
import time

t1=time.time()
print(t1)
#This function calculates epsilon\dotx
def VecP(M,x,p):
   r=M.shape
   n=r[0]
   k=r[1]
   #print(r)
   
  # H1=numpy.zeros((n,k-1))
  # print(H1)
   HM=numpy.ones((n,k))
  # H1=M[:,1:]-M[:,0:k-1]
   #y=numpy.zeros(n)
   #print(M[:,1:],M[:,0:k-1])
   #print(HM,H1)
   for i in range(0,n):
       #print(M[i,:].mean(),H1.mean()) 
       HM[i,:]=p[i]*HM[i,:]
   #print(HM)    
   M1=(HM-M[:,:])/M[:,:]    
   #print(M1)
   y=x[0]*M1[0,:]   
   for i in range(1,n):
       #y=y+M1[i,:]
        y=y+x[i]*M1[i,:]
   return y

#This function calculates the close price matrix
def  ddtt(n):
 
 r=numpy.zeros(n)
 
 for i in range(0,n):
       
    fi='Data'+str(i)+'.csv'
   
    datai=pd.read_csv(fi,header=0)
    yi=datai['close']
    #ki=yi[:5]
    #print(ki)
    r[i]=len(yi)

 k=int(numpy.min(r))
 k=252
 
 #print(k)
#k=5
 h=numpy.zeros((n,k))
 #print(h(1,:))
 for i in range(0,n):
    
    fi='Data'+str(i)+'.csv'
    filenamei=fi
    datai=pd.read_csv(filenamei,header=0)
    yi=datai['close']
    h[i,:]=yi[:k]
    #h(i,:)=yi[:5]
    #print(h[i,:])
    #r[i]=len(yi)
 return h

#P1=ddtt(2)
#print(P1)    
#x=numpy.array([2.0,4.0])
#p=numpy.array([50.0,25.0])
#P2=VecP(P1,x,p)
#print(P2)

def GWO(lb, ub, dim, SearchAgents_no,Max_iter,eta,ups,pA):
 
    MP=ddtt(dim)#Cada fila representa los precios de cierre de un simbolo
    
    
    # Numero de activos
    # Max_iter=1000
    # lb=-100
    # ub=100
    # dim=30
    # SearchAgents_no=5

    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = numpy.zeros(Max_iter)
    class solution:
     def __init__(self):
        self.best = 0
        self.bestIndividual = []
        self.convergence = []
        self.optimizer = ""
        self.objfname = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        self.lb = 0
        self.ub = 0
        self.dim = 0
        self.popnum = 0
        self.maxiers = 0

    s = solution()
    #print(s)
    # Loop counter
    #print('GWO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    #print(s.best,'best')
    #print(s.bestIndividual,'bestIndividual')
    #print(s.convergence,'convergence')
    #print(s.optimizer,'optimizer')
    #print(s.objfname,'objfname')
    #print(s.startTime,'startTime')
    #print(s.endTime,'endTime')
    #print(s.executionTime,'executionTime')
    #print(s.lb,'lb')
    #print(s.ub,'ub')
    #print(s.dim,'dim')
    #print(s.popnum,'popnum')
    #print(s.maxiers,'maxiers')
    # Main loop
    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])
            
            #print(pA)
            #print(MP)
            # Calculate objective function for each search agent
            Positions[i,:]=Positions[i,:]/sum(Positions[i,:])
            eps=VecP(MP,Positions[i, :],pA)   
            #print(eps)      
            fitness = eps.var()
            
            #print('fitness',fitness)
            sk=skew(eps)-eta
            
            mean=eps.mean()-ups
            #print(mean,'media')
            
            #print(buni)
            if sk<0:
               fitness=fitness+10**10.0
            if mean<0:
               fitness=fitness+10**10.0 
           
          
            
            

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness < Beta_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / Max_iter)
        # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a
                # Equation (3.3)
                C1 = 2 * r2
                # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha
                # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                # Equation (3.3)
                C2 = 2 * r2
                # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta
                # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                # Equation (3.3)
                C3 = 2 * r2
                # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta
                # Equation (3.5)-part 3

                Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)

        Convergence_curve[l] = Alpha_score

        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(Alpha_score)]
            )

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GWO"
    #s.objfname = objf.__name__ 
    s.bestIndividual=Alpha_pos
 #   print(s.best,'best')
  #  print(s.bestIndividual,'bestIndividual')
   # print(s.convergence,'convergence')
    #print(s.optimizer,'optimizer')
    #print(s.objfname,'objfname')
    #print(s.startTime,'startTime')
    #print(s.endTime,'endTime')
    #print(s.executionTime,'executionTime')
    #print(s.lb,'lb')
    #print(s.ub,'ub')
    #print(s.dim,'dim')
    #print(s.popnum,'popnum')
    #print(s.maxiers,'maxiers')
 
    return Alpha_pos
dim=100
hh=ddtt(dim)
y=hh[:,251]
#print(y)
#GWO(lb, ub, dim, SearchAgents_no,Max_iter,eta,ups,pA): 
sol=GWO(0.01, 1.0, dim, 100, 1000,0.1,0.1,y)
print(sol)

def  ddtt2(n): 
 r=numpy.zeros(n) 
 for i in range(0,n):       
    fi='Data'+str(i)+'.csv'
    #filenamei=fi
    datai=pd.read_csv(fi,header=0)
    yi=datai['close']
    r[i]=len(yi)

 k=int(numpy.min(r))
  
 h=numpy.zeros((n,k))
 for i in range(0,n):
    
    fi='Data'+str(i)+'.csv'
    #filenamei=fi
    datai=pd.read_csv(fi,header=0)
    yi=datai['close']
    h[i,:]=yi[:k]
    
 return h




BT=ddtt2(dim)
BT0=BT[:,:252]
yBT0=BT[:,253]

epBT0=VecP(BT0,sol,yBT0)


BT1=BT[:,253:283]
yBT=BT[:,283]

epBT=VecP(BT1,sol,yBT)
#Esperanza, riesgo y oblicuidad calculados con la estrategia empleada
print(epBT0.mean())
print(epBT0.var())
print(skew(epBT0))
#Esperanza, riesgo y oblicuidad calculados con la estrategia empleada
print(epBT.mean())
print(epBT.var())
print(skew(epBT))

t2=time.time()

print((t2-t1)/60)

