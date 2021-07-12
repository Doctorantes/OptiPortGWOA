#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:58:01 2021

@author: Alfonso S.A
"""

import os  
os.chdir("//home//usuario//Desktop//Simbolos")
os.getcwd()#Con este modulo abrimos la carpeta de donde sacaremos la informacion
import matplotlib.pyplot as plt
import numpy            
import pandas as pd            
import random
import math
from scipy.stats import skew
import time


t1=time.time()
print(t1)
#Esta funcion (VecP) calcula un vector que obtiene las gananciasdiarias
#De haber invertido una distribucion diaria x
#Y vender de acuerdo a la lista de precios p
#Las observacioes de los precios se alojan en la Matriz M
def VecP(M,x,p):
   r=M.shape
   n=r[0]
   k=r[1]
   HM=numpy.ones((n,k))
   for i in range(0,n):
       
       HM[i,:]=p[i]*HM[i,:]
       
   M1=(HM-M[:,:])/M[:,:]    
   
   y=x[0]*M1[0,:]   
   for i in range(1,n):
       
        y=y+x[i]*M1[i,:]
   return y

#Esta funcion ddtt, guarda los precios regitrados para el entrenamiento 
#Asi como para el test de nuestra estrategia
#La funcion ddtt da una matriz cuya i-esima columna registra los precios
#del i-esimo activo, y la entraja j-esima de dicha fila, es el precio del 
#i-esimo activo al dia j.

#n-Numero de activos a tomar en cuenta
#n=2 implica que se tomara en cuenya el activo 0 y 1.
def  ddtt(n):
 
 r=numpy.zeros(n)
 
 for i in range(0,n):
       
    fi='Data'+str(i)+'.csv'
    
    datai=pd.read_csv(fi,header=0)
    
    yi=datai['close']
          
    r[i]=len(yi)

 k=int(numpy.min(r))
 k=493
 
 h=numpy.zeros((n,k))
 for i in range(0,n):
    
    fi='Data'+str(i)+'.csv'
   
    datai=pd.read_csv(fi,header=0)
    yi=datai['close']
    h[i,:]=yi[:k]
    
 return h
#Implementacion del algoritmo de los lobos grises al modelo 
#EVS, se optimiza la esperanza, con un tope maximo a la varianza(eta) y un
 #tope minimo a la oblicuidad (ups)
#lb.-vector o numero que denota el extremo izquierdo donde se hara la busqueda
# de las x_i a optimizar 
#ub.-vector o numero que denota el extremo derecho donde se hara la busqueda
# de las x_i a optimizar
def GWO(lb, ub, dim, SearchAgents_no,Max_iter,eta,ups,pA):
 
    MP=ddtt(dim)#Lammamos a la matriz de precios
    #MP=MP[:,]
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = -float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = -float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = -float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i])

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
    
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):

            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])
            
            Positions[i,:]=Positions[i,:]/sum(Positions[i,:])
            eps=VecP(MP,Positions[i, :],pA)   
            
            fitness =eps.mean()
            
            vr=eps.var()-eta
            
            sk=skew(eps)-ups
            #Funciones de penalizacion
            # Se evaluara si la varianza no sobrepasa eta
            #Se evaluara si la oblicuidad cumple sobrepasar a ups
            if vr>0:
               fitness=fitness-10**10.0
            if sk<0:
               fitness=fitness-10**10.0 
            
            
            

            if fitness > Alpha_score:
                Delta_score = Beta_score  
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score 
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                
                Alpha_pos = Positions[i, :].copy()

            if fitness < Alpha_score and fitness > Beta_score:
                Delta_score = Beta_score 
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  
                Beta_pos = Positions[i, :].copy()

            if fitness < Alpha_score and fitness < Beta_score and fitness > Delta_score:
                Delta_score = fitness 
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / Max_iter)
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                r1 = random.random()
                r2 = random.random()  

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

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
    s.bestIndividual=Alpha_pos
    return Alpha_pos

#Testeo de la estrategia a partir del dia 253 con termino 
#el dia 452
dim=100
hh=ddtt(dim)
y=hh[:,492]
sol=GWO(0.01, 1.0, dim, 30, 100,0.08,0.001,y)
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
BT0=BT[:,:493]
yBT0=BT[:,492]

epBT0=VecP(BT0,sol,yBT0)


BT1=BT[:,493:510]
yBT=BT[:,508]

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