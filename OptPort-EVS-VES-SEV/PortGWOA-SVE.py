"""
Created On Wed May 18 13:09:36 2021

@Author: usuario
"""

import random
import os
import time
from scipy.stats import skew
import pandas as pd
import numpy as np
os.chdir("//home//usuario//Desktop//Simbolos")
os.getcwd()



T1 = time.time()  # We record the time at which our program starts running


def ddtt(n_n):
    """
    The function ddtt, stores the prices recorded for training as well as for
    the test of our strategy.
    The ddtt function has as output a matrix, whose i-th column records the
    prices of the i-th asset.  The j-th entry of this row is the price of the
    i-th asset on the jth day.

    PARAMETERS
    ----------
    n : int
        Number of symbols to be taken into account.

    RETURNS
    -------
    array
        An array ddtt, whose input ddtt[i,j] is the price of the i-th asset on
        the j-th day.
    """
    f_0 = 'Data0.csv'  # Open the first file Data0.csv
    data0 = pd.read_csv(
        f_0, header=0)  # Save the information of this first file
    y_y = data0['date']  # We keep the dates where we have records of
    # this first file AS INITIALIZATION of YY

    for i in range(n_n):

        f_i = 'Data' + str(
            i) + '.csv'  # We give the names of the files that we are going to read

        datai = pd.read_csv(f_i, header=0)  # We load the file data fi
        y_i = datai[['date',
                     'close']]  # We select the dates and the closing prices
        # of the assets on those dates

        y_y = pd.merge(y_y, y_i,
                       on=['date']) # We perform the equivalent operation of an
        # inner join with the data 'tables' y_i, considering only the fields
        #'date' and 'close'. The inner join is calculated on the 'date' field
        # of the tables.
    nmp = y_y.to_numpy()  # We convert our resulting dataframe to array
    nmp0 = nmp[:, 0]  # We store the dates in where the price was recorded
    nmp1 = np.transpose(nmp[:,
                            1:]) # We transpose the prices of the nmp array
        # without the record dates
    return nmp0, nmp1  # return the vector containing the dates where there
        # were records (nmp0), and the array of prices per day (nmp1)


DIM0 = int(
    input(' Introduzca el numero de activos con los que quiere trabajar;'))

P = ddtt(DIM0)
print('Las fechas en las que existen registros son;')
print(P[0]) # We show the dates where we have records



MF = P[0]
MF1 = P[1]
EDATE1 = input(
    'Introduzca la fecha en la que quiere iniciar el periodo de entrenamiento; '
)



EDATE2 = input(
    'Introduzca la fecha en la que quiere terminar el periodo de entrenamiento; '
)


BDATE1 = input(
    'Introduzca la fecha en la que quiere iniciar el periodo de backtesting; ')


BDATE2 = input(
    'Introduzca la fecha en la que quiere terminar el periodo de backtesting; '
)

E1 = np.where(MF == EDATE1)[0]
E2 = np.where(MF == EDATE2)[0] + 1

B1 = np.where(MF == BDATE1)[0]
B2 = np.where(MF == BDATE2)[0] + 1


ME = MF1[:, E1[0]:E2[0]]

BT = MF1[:, B1[0]:B2[0]]


def vec_p(m_m, x_x, p_p):
    """
    La funcion (vec_p) calcula el vector cuyas entradas contienen el porcentaje
    de ganancia diaria del capital  invertido, durante el lapso de tiempo que se
    invirtio, siguiendo una distribucion diaria x_x  del capital, y vender de
    acuerdo a la lista de precios p_p .
    PARAMETERS
    ----------
    m_m:array
        La matriz m_m tiene en sus entrada i,j el precio del activo i-esimo,     al j-esimo dia
    x_x:array(vector)
        El vector x_x tiene en sus entradas el porcentaje x[i] de capital qu
        se invertira en el activo i-esimo
    p_p:array(vector)
        El vector p_p tiene en sus entradas el precio p_p[i] al que se vender
        el activo i-esimo
    RETURNS
    -------
    array
         Un vector cuya entrada i-esima tiene el porcentaje del capital inicial
         que se gano o perdio
    """


    r_r = m_m.shape
    n_n = r_r[0]
    k = r_r[1]
    h_m = np.ones((n_n, k))
    for i in range(n_n):

        h_m[i, :] = p_p[i] * h_m[i, :]

    m_1 = (h_m - m_m[:, :]) / m_m[:, :]

    g_g = x_x[0] * m_1[0, :]
    for i in range(1, n_n):

        g_g = g_g + x_x[i] * m_1[i, :]
    return g_g


#Implementacion del algoritmo de los lobos grises al modelo
#EVS, se optimiza la esperanza, con un tope maximo a la varianza(eta) y un
#tope minimo a la oblicuidad (ups)

#lb.-vector o numero que denota el extremo izquierdo donde se hara la busqueda
# de las x_i a optimizar.

#Si para el parametro lb solo se da un solo valor se entiende que todos los
#invervalos  donde se busca cada una de las entradas de el vector x a optimizar
# tienen   el mismo limite inferior el cual es lb. Si se da un vector,
#este debe tener la misma longitud de x, el cual es el mismo que el numero de
#activos considerados.

#ub.-vector o numero que denota el extremo derecho donde se hara la busqueda
# de las x_i a optimizar.

#Si para el parametro ub solo se da un solo valor se entiende que todos los
#invervalos  donde se busca cada una de las entradas de el vector x a optimizar
# tienen   el mismo limite superior el cual es ub. Si se da un vector,
#este debe tener la misma longitud de x, el cual es el mismo que el numero de
#activos considerados.

#dim.- Es el numero de simbolos que estamos tomando en cuenta

#SearchAgents_no.-Estamos en busca del portafolio optimo x=(x_1,...,x_n).
#Entonces a cada x_i ponemos 'SearchAgents_no' agentes (lobos grises) a buscar a
#x_i

#lim_lu=(l_b,u_b)
def gw_oa(lim_lu, par_gwoa, lim_est, m_p, p_a):

    """
    La funcion gw
    """
    l_b, u_b, eta, ups = lim_lu[0], lim_lu[1], lim_est[0], lim_est[1]
    dim, searchagents_no, max_iter = par_gwoa[0], par_gwoa[1], par_gwoa[2]
    # En la siguiente linea inicializamos los vectores que contendran a las
    # mejores aproximaciones encontradas al portafolio optimo.
    alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    # En la siguiente linea inicializamos la evalucion de la funcion a
    # optimizar en el vector Alpha_pos. En este caso la funcion es la esperanza
    # matematica.
    #alpha_score = -float("inf")

    # En la siguiente linea inicializamos los vectores que contendran a las
    # segundas mejores aproximaciones encontradas al portafolio optimo.
    #beta_pos = np.zeros(dim)

    # En la siguiente linea inicializamos la evalucion de la funcion a
    # optimizar en el vector Beta_pos. En este caso la funcion es la esperanza
    # matematica.
    #beta_score = -float("inf")

    # En la siguiente linea inicializamos los vectores que contendran a las
    # terceras mejores aproximaciones encontradas al portafolio optimo.
    #delta_pos = np.zeros(dim)

    #En la siguiente linea inicializamos la evalucion de la funcion a
    # optimizar en el vector Delta_pos. En este caso la funcion es la esperanza
    #matematica.
    alpha_score, beta_score, delta_score = -float("inf"), -float("inf"), -float("inf")

    # Inicializamos a los limites inferiores de cada uno de los intervlos donde
    # los agentes buscaran a cada una de las componentes x_i del portafolio

    # lb.-Si en el parametro lb hemos dado un numero (y solo en este caso),
    # se aciva en la siguiente linea la creacion un vector de la misma longitud
    # del portafolio que tenga en cada una de sus entradas el numero dado
    if not isinstance(l_b, list):
        l_b = [l_b] * dim

    # ub.-Si en el parametro lb hemos dado un numero (y solo en este caso),
    # se aciva en la siguiente linea la creacion un vector de la misma longitud
    # del portafolio que tenga en cada una de sus entradas el numero dado

    if not isinstance(u_b, list):
        u_b = [u_b] * dim

    # Se inicializan un arreglo de  dimensiones; SearchAgents_no X dim.
    # Para cada una de las dim incognitas se pondran a buscar SearchAgents_no
    # agentes
    pos_pos = np.zeros((searchagents_no, dim))

    # Las entradas del arreglo anterior se inicializan de manera aleatoria
    for i in range(dim):
        pos_pos[:, i] = (np.random.uniform(0, 1, searchagents_no) *
                         (u_b[i] - l_b[i]) + l_b[i])

    # Se define a la clase solucion, dicha clase contendra la mejor aproximacion
    # al portafolio de inversion optimo buscado. Asi como la esperanza obtenida
    # de haber invertido tomando en cuenta dicha distribucion del capital
    class Solution:
        """
        as
        """

        def __init__(self):
            self.best = 0
            self.best_individual = []

    s_s = Solution()

    # Implementamos el algoritmo principal de los lobos grises

    for h_h in range(max_iter):
        for i in range(searchagents_no):

            for j in range(dim):
                pos_pos[i, j] = np.clip(pos_pos[i, j], l_b[j], u_b[j])

            pos_pos[i, :] = pos_pos[i, :] / sum(pos_pos[i, :])
            eps = vec_p(m_p, pos_pos[i, :], p_a)

            fitness, v_r, m_m = skew(eps), eps.var() - eta, eps.mean()-ups



            if v_r > 0 or m_m < 0:
                fitness = fitness - 10**10.0
            #if s_k < 0:
             #   fitness = fitness - 10**10.0

            if fitness > alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                #delta_pos = beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                #beta_pos = alpha_pos.copy()
                alpha_score, alpha_pos = fitness, pos_pos[i, :].copy()

                #alpha_pos = pos_pos[i, :].copy()

            if beta_score < fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                #delta_pos = beta_pos.copy()
                beta_score, beta_pos = fitness, pos_pos[i, :].copy()
               # beta_pos = pos_pos[i, :].copy()

            if fitness < alpha_score and delta_score < fitness < beta_score:
                delta_score, delta_pos = fitness, pos_pos[i, :].copy()
                #delta_pos = pos_pos[i, :].copy()

        a_a = 2 - h_h * ((2) / max_iter)
        for i in range(searchagents_no):
            for j in range(dim):

                r_1, r_2 = random.random(), random.random()
                #r_2 = random.random()

                a_1, c_1 = 2 * a_a * r_1 - a_a, 2 * r_2
                #c_1 = 2 * r_2

                d_alpha = abs(c_1 * alpha_pos[j] - pos_pos[i, j])
                x_1 = alpha_pos[j] - a_1 * d_alpha

                r_1, r_2 = random.random(), random.random()
                #r_2 = random.random()

                a_2, c_2 = 2 * a_a * r_1 - a_a, 2 * r_2
                #c_2 = 2 * r_2

                d_beta = abs(c_2 * beta_pos[j] - pos_pos[i, j])
                x_2 = beta_pos[j] - a_2 * d_beta

                r_1, r_2 = random.random(), random.random()
                #r_2 = random.random()

                a_3, c_3 = 2 * a_a * r_1 - a_a, 2 * r_2
                #c_3 = 2 * r_2

                d_delta = abs(c_3 * delta_pos[j] - pos_pos[i, j])
                x_3 = delta_pos[j] - a_3 * d_delta

                pos_pos[i, j] = (x_1 + x_2 + x_3) / 3  # Equation (3.7)

       # if h_h % 1 == 0:
        #    print([
         #       "At iteration " + str(h_h) + " the best fitness is " +
          #      str(alpha_score)
          #  ])

    #Obtenemos a los outputs deseados.
    #s.bestIndividual.- La oproximacion calculada a mejor portafolio
    #s.best.- La esperanza que se obtiene de calcular dicho portafolio
    s_s.best_individual = alpha_pos
    s_s.best = alpha_score
    print(['La mejor esperanza encontrada es ' + str(s_s.best)])
    return s_s.best_individual


Y = MF1[:, E2 - 1]
P_1 = [0.01, 1.0]
P_2 = [DIM0, 30, 100]
P_3 = [0.08, 0.001]
#print(y)
SOL = gw_oa(P_1, P_2, P_3, ME, Y)

print(SOL)
#print(MF1[:,B2-1])
EPBT = vec_p(BT, SOL, MF1[:, B2 - 1])
print('')
print('El promedio diario de ganacia en el backtesing es;', EPBT.mean())
print('')
print('La desviacion estandar en el backtesting es;', EPBT.var())
print('')
print('El sesgo en el backtesting;', skew(EPBT))
T2 = time.time()
print('')
#Tiempo de ejercucion
print('El tiempo de ejecucion es;', (T2 - T1) / 60, 'minutos')
