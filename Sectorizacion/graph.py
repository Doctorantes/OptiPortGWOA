import networkx as nx
import pandas as pd
import numpy as np

path = '/home/angel/Desktop/Sectorizacion/sp500-1day/'

simbolos = pd.read_csv(path + "sp500_symbols.csv")
simbolos = simbolos['symbol']


def crear_graph():
    '''Ordena los datos y crea una grafica completa
    con los activos como nodos y las correlaciones
    como el peso de las aristas'''

    G = nx.Graph()

    path = '/home/angel/Desktop/Sectorizacion/sp500-1day/'

    simbolos = pd.read_csv(path + "sp500_symbols.csv")

    G.add_nodes_from(list(simbolos['symbol']))
    A = list(G.nodes)

    primeras_fechas = []
    ultimas_fechas = []

    for i in range(len(G.nodes)):
        if A[i] != 'AMCR' and A[i] != 'FANG' and A[i] != 'FLIR':
            df = np.array(pd.read_csv(path + str(A[i]) + ".csv")["date"])
            primeras_fechas.append(min(df))
            ultimas_fechas.append(max(df))

    fecha_inicio = max(primeras_fechas)
    fecha_final = min(ultimas_fechas)

    df1 = pd.read_csv(path + str(A[0]) + '.csv')
    date1 = np.array(df1['date'])

    for i in range(len(G.nodes)):
        if A[i] != 'AMCR' and A[i] != 'FANG' and A[i] != 'FLIR':
            print(i)
            df1 = pd.read_csv(path + str(A[i]) + ".csv")
            date1 = np.array(df1['date'])
            df1 = df1['close'][np.where(date1 == fecha_inicio)[0][0]:np.where(
                date1 == fecha_final)[0][0] + 1]
            for j in range(i, len(G.nodes)):
                if A[j] != 'AMCR' and A[j] != 'FANG' and A[j] != 'FLIR':
                    df2 = pd.read_csv(path + str(A[j]) + ".csv")
                    date2 = np.array(df2['date'])
                    df2 = df2['close'][np.where(
                        date2 == fecha_inicio)[0][0]:np.where(
                            date2 == fecha_final)[0][0] + 1]
                    if len(df1) == len(df2):
                        G.add_edge(A[i],
                                   A[j],
                                   weight=np.corrcoef(df1, df2)[0][1])
    nx.write_graphml(G, "correlaciones.graphml")


def ordenar(x):
    x.sort()
    return (x)


G = nx.read_graphml('correlaciones.graphml')


def vecinos(x):
    '''Encuentra todos los vecinos de x con los que tiene
    una correlacion mayor a 0.95 en valor absoluto'''

    banderas = (np.array(list(G.edges(data=True)))[:, 0] == x) | (np.array(
        list(G.edges(data=True)))[:, 1] == x)
    vecinos = np.concatenate((np.array(list(
        G.edges(data=True)))[:, 0][np.array(list(G.edges(
            data=True)))[:, 1] == x], np.array(list(
                G.edges(data=True)))[:, 1][np.array(list(G.edges(
                    data=True)))[:, 0] == x][1:]),
                             axis=0)
    pesos = np.array(list(G.edges(data=True)))[:, 2][banderas]
    pesos = np.array(list(map(lambda x: abs(x['weight']), pesos)))
    return (vecinos[pesos > 0.95])


def contraccion(x):
    '''Pone en un conjunto todos los activos que se
    deben contraer en uno con x'''

    vec = set(vecinos(x))
    vec1 = vec.copy()
    vec2 = set([x])
    r = True
    while r:
        for k in list(vec):
            if k not in vec2:
                vec = vec | set(vecinos(k))
        if vec == vec1:
            return (vec)
            r = False
        else:
            vec2 = vec1.copy()
            vec1 = vec.copy()
    return (vec)


nodes_nuevos = []
nodes = list(G.nodes)
for i in range(len(nodes)):
    if nodes[i] != 'AMCR' and nodes[i] != 'FANG' and nodes[i] != 'FLIR':
        bandera = True
        for k in nodes_nuevos:
            if nodes[i] in k:
                bandera = False
                break
        if bandera:
            nodes_nuevos.append(contraccion(nodes[i]))
