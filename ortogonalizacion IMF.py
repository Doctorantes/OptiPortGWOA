# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016
@author: Hossam Faris
"""
import random
import numpy
import math
#from solution import solution
import time

def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):

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
    print('GWO is optimizing  "' + objf.__name__ + '"')

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

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])
            print(Positions[i, :])
            print(fitness)
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
    s.objfname = objf.__name__ 
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
 
    return s,s.bestIndividual,objf(s.bestIndividual)
#Funciones de apoyo
def prod(it):
    p = 1
    for n in it:
        p *= n
    return p

def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y
#####################################
def F1(x):
    s = numpy.sum(x ** 2)
    return s

def F2(x):
    o = sum(abs(x)) + prod(abs(x))
    return o
 
def F3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (numpy.sum(x[0:i])) ** 2
    return o

def F4(x):
    o = max(abs(x))
    return o
def F5(x):
    dim = len(x)
    o = numpy.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return o


def F6(x):
    o = numpy.sum(abs((x + 0.5)) ** 2)
    return o


def F7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = numpy.sum(w * (x ** 4)) + numpy.random.uniform(0, 1)
    return o


def F8(x):
    o = sum(-x * (numpy.sin(numpy.sqrt(abs(x)))))
    return o


def F9(x):
    dim = len(x)
    o = numpy.sum(x ** 2 - 10 * numpy.cos(2 * math.pi * x)) + 10 * dim
    return o


def F10(x):
    dim = len(x)
    o = (
        -20 * numpy.exp(-0.2 * numpy.sqrt(numpy.sum(x ** 2) / dim))
        - numpy.exp(numpy.sum(numpy.cos(2 * math.pi * x)) / dim)
        + 20
        + numpy.exp(1)
    )
    return o
def F11(x):
    dim = len(x)
    w = [i for i in range(len(x))]
    w = [i + 1 for i in w]
    o = numpy.sum(x ** 2) / 4000 - prod(numpy.cos(x / numpy.sqrt(w))) + 1
    return o


def F12(x):
    dim = len(x)
    o = (math.pi / dim) * (
        10 * ((numpy.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
        + numpy.sum(
            (((x[: dim - 1] + 1) / 4) ** 2)
            * (1 + 10 * ((numpy.sin(math.pi * (1 + (x[1 :] + 1) / 4)))) ** 2)
        )
        + ((x[dim - 1] + 1) / 4) ** 2
    ) + numpy.sum(Ufun(x, 10, 100, 4))
    return o


def F13(x):
    if x.ndim==1:
        x = x.reshape(1,-1)

    o = 0.1 * (
        (numpy.sin(3 * numpy.pi * x[:,0])) ** 2
        + numpy.sum(
            (x[:,:-1] - 1) ** 2
            * (1 + (numpy.sin(3 * numpy.pi * x[:,1:])) ** 2), axis=1
        )
        + ((x[:,-1] - 1) ** 2) * (1 + (numpy.sin(2 * numpy.pi * x[:,-1])) ** 2)
    ) + numpy.sum(Ufun(x, 5, 100, 4))
    return o


def F14(x):
    aS = [
        [
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
        ],
        [
            -32, -32, -32, -32, -32,
            -16, -16, -16, -16, -16,
            0, 0, 0, 0, 0,
            16, 16, 16, 16, 16,
            32, 32, 32, 32, 32,
        ],
    ]
    aS = numpy.asarray(aS)
    bS = numpy.zeros(25)
    v = numpy.matrix(x)
    for i in range(0, 25):
        H = v - aS[:, i]
        bS[i] = numpy.sum((numpy.power(H, 6)))
    w = [i for i in range(25)]
    for i in range(0, 24):
        w[i] = i + 1
    o = ((1.0 / 500) + numpy.sum(1.0 / (w + bS))) ** (-1)
    return o


def F15(L):
    aK = [
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ]
    bK = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    aK = numpy.asarray(aK)
    bK = numpy.asarray(bK)
    bK = 1 / bK
    fit = numpy.sum(
        (aK - ((L[0] * (bK ** 2 + L[1] * bK)) / (bK ** 2 + L[2] * bK + L[3]))) ** 2
    )
    return fit


def F16(L):
    o = (
        4 * (L[0] ** 2)
        - 2.1 * (L[0] ** 4)
        + (L[0] ** 6) / 3
        + L[0] * L[1]
        - 4 * (L[1] ** 2)
        + 4 * (L[1] ** 4)
    )
    return o


def F17(L):
    o = (
        (L[1] - (L[0] ** 2) * 5.1 / (4 * (numpy.pi ** 2)) + 5 / numpy.pi * L[0] - 6)
        ** 2
        + 10 * (1 - 1 / (8 * numpy.pi)) * numpy.cos(L[0])
        + 10
    )
    return o


def F18(L):
    o = (
        1
        + (L[0] + L[1] + 1) ** 2
        * (
            19
            - 14 * L[0]
            + 3 * (L[0] ** 2)
            - 14 * L[1]
            + 6 * L[0] * L[1]
            + 3 * L[1] ** 2
        )
    ) * (
        30
        + (2 * L[0] - 3 * L[1]) ** 2
        * (
            18
            - 32 * L[0]
            + 12 * (L[0] ** 2)
            + 48 * L[1]
            - 36 * L[0] * L[1]
            + 27 * (L[1] ** 2)
        )
    )
    return o


# map the inputs to the function blocks
def F19(L):
    aH = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
    aH = numpy.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = numpy.asarray(cH)
    pH = [
        [0.3689, 0.117, 0.2673],
        [0.4699, 0.4387, 0.747],
        [0.1091, 0.8732, 0.5547],
        [0.03815, 0.5743, 0.8828],
    ]
    pH = numpy.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * numpy.exp(-(numpy.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F20(L):
    aH = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
    aH = numpy.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = numpy.asarray(cH)
    pH = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
    pH = numpy.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * numpy.exp(-(numpy.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F21(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = numpy.asarray(aSH)
    cSH = numpy.asarray(cSH)
    fit = 0
    for i in range(5):
        v = numpy.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F22(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = numpy.asarray(aSH)
    cSH = numpy.asarray(cSH)
    fit = 0
    for i in range(7):
        v = numpy.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F23(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = numpy.asarray(aSH)
    cSH = numpy.asarray(cSH)
    fit = 0
    for i in range(10):
        v = numpy.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o
#Func,lb,ub,dim
"""" "F1": ["F1", -100, 100, 30],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -100, 100, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],
    }"""    
#GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter)
R=GWO(F3, -100, 100, 30, 50, 1000)
print(R)
