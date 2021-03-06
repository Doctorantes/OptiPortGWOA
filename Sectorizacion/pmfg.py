import networkx as nx
import pandas as pd
import numpy as np
import operator

path = '/home/angel/Desktop/Datos/sp500-1day/'

simbolos = pd.read_csv(path + "sp500_symbols.csv")
simbolos = simbolos['symbol']

G = nx.read_graphml('correlaciones.graphml')
edges = list(G.edges)
pesos = list(np.array(list(G.edges(data=True)))[:, 2])
pesos = list(map(lambda x: abs(x['weight']), pesos))
edges = dict(zip(edges, pesos))

edges_sort = np.array(
    sorted(edges.items(), key=operator.itemgetter(1), reverse=True))

edges = dict(zip(list(edges_sort[:, 0]), list(edges_sort[:, 1])))


def PMFG():
    G_planar_maximal = nx.Graph()
    G_planar_maximal.add_nodes_from(list(simbolos))
    G_aux = G_planar_maximal.copy()
    for edge in edges.keys():
        G_aux.add_edge(edge[0], edge[1], weight=edges[edge])
        if nx.check_planarity(G_aux)[0]:
            G_planar_maximal.add_edge(edge[0], edge[1], weight=edges[edge])
        G_aux = G_planar_maximal.copy()
    return G_planar_maximal
