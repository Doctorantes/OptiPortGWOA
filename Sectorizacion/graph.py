import networkx as nx
import pandas as pd
import numpy as np

path='/home/angel/Desktop/Sectorizacion/sp500-1day/'

simbolos= pd.read_csv(path+"sp500_symbols.csv")
simbolos=simbolos['symbol']

def crear_graph():
	G=nx.Graph()

	path='/home/angel/Desktop/Sectorizacion/sp500-1day/'

	simbolos= pd.read_csv(path+"sp500_symbols.csv")

	G.add_nodes_from(list(simbolos['symbol']))
	A=list(G.nodes)

	primeras_fechas=[]
	ultimas_fechas=[]

	for i in range(len(G.nodes)):
		if A[i]!='AMCR' and A[i]!='FANG' and A[i]!='FLIR':
			df=np.array(pd.read_csv(path+str(A[i])+".csv")["date"])
			primeras_fechas.append(min(df))
			ultimas_fechas.append(max(df))

	fecha_inicio=max(primeras_fechas)
	fecha_final=min(ultimas_fechas)

	df1=pd.read_csv(path+str(A[0])+'.csv')
	date1=np.array(df1['date'])
	#print(np.where(date1==fecha_inicio)[0][0])
	#print(df1[np.where(df1==fecha_inicio)[0][0]:np.where(df1==fecha_final)[0][0]+1])
	for i in range( len(G.nodes)):
		if A[i]!='AMCR' and A[i]!='FANG' and A[i]!='FLIR':
			print(i)
			df1=pd.read_csv(path+str(A[i])+".csv")
			date1=np.array(df1['date'])
			#print(np.where(date1==fecha_inicio)[0][0],np.where(date1==fecha_final)[0][0]+1)
			df1=df1['close'][np.where(date1==fecha_inicio)[0][0]:np.where(date1==fecha_final)[0][0]+1]
			#print (len(df1))
			for j in range(i,len(G.nodes)):
				#print(j)
				if A[j]!='AMCR' and A[j]!='FANG' and A[j]!='FLIR':
					df2=pd.read_csv(path+str(A[j])+".csv")
					date2=np.array(df2['date'])
					df2=df2['close'][np.where(date2==fecha_inicio)[0][0]:np.where(date2==fecha_final)[0][0]+1]
					if len(df1)==len(df2):
						G.add_edge(A[i], A[j], weight=np.corrcoef(df1,df2)[0][1])
	nx.write_graphml(G, "correlaciones.graphml")


#df1=np.array(pd.read_csv(str(A[0])+".csv")["date"])
#print(np.where(df1==fecha_inicio)[0][0])
#print(df1[np.where(df1==fecha_inicio)[0][0]:np.where(df1==fecha_final)[0][0]+1])

#crear_graph()

#print(fecha_inicio,fecha_final)
def ordenar(x):
	x.sort()
	return(x)

G = nx.read_graphml('correlaciones.graphml')
#edges=list(np.array(list(G.edges(data=True)))[:, :2])
#edges=list(map(ordenar, edges))
#edges=list(map(tuple, edges))
#pesos=list(np.array(list(G.edges(data=True)))[:, 2])
#pesos=list( map(lambda x: x['weight'], pesos) )
#edges=dict(zip(edges, pesos))
#print(edges)

def vecinos(x):
	#print('wwwwwwwwwww')
	#print(np.array(list(G.edges(data=True)))[:, 0])
	#print(np.array(list(G.edges(data=True)))[:, 1])
	banderas=(np.array(list(G.edges(data=True)))[:, 0]==x) |  (np.array(list(G.edges(data=True)))[:, 1]==x)
	#print(banderas)
	#print(len(banderas))
	vecinos=np.concatenate((np.array(list(G.edges(data=True)))[:, 0][np.array(list(G.edges(data=True)))[:, 1]==x],
	np.array(list(G.edges(data=True)))[:, 1][np.array(list(G.edges(data=True)))[:, 0]==x ][1:]),axis=0)
	#print(np.array(list(G.edges(data=True)))[:, 1][np.array(list(G.edges(data=True)))[:, 0]==x ])
	#print(np.array(list(G.edges(data=True)))[:, 0][np.array(list(G.edges(data=True)))[:, 1]==x][1:])
	pesos=np.array(list(G.edges(data=True)))[:, 2][banderas]
	pesos=np.array(list( map(lambda x: abs(x['weight']), pesos) ))
	return(vecinos[pesos>0.95])
#print(vecinos(simbolos[0]))

def cadco(x):
	vec=set(vecinos(x))
	#print(vec)
	vec1=vec.copy()
	vec2=set([x])
	r=True
	while r:
		for k in list(vec):
			if k not in vec2:
				#print(k)
				vec=vec|set(vecinos(k))
		if vec==vec1:
			return (vec)
			r=False
		else:
			#print(len(vec))
			vec2=vec1.copy()
			vec1=vec.copy()
		print(len(vec))
	return (vec)
		


#r=cadco(simbolos[0])
#print(r, len(r))

nodes_nuevos=[]
nodes=list(G.nodes)
for i in range(len(nodes)):
	if nodes[i]!='AMCR' and nodes[i]!='FANG' and nodes[i]!='FLIR':
		bandera = True
		#print ('yyyy',nodes_nuevos)
		for k in nodes_nuevos:
			if nodes[i] in k:
				bandera=False
				break
		if bandera:
			#print(nodes_nuevos)
			nodes_nuevos.append(cadco(nodes[i]))
		print('xxxxxxxxxxxxxxxx',i)
		

print(nodes_nuevos)

def minor():
	contracciones={}
	for x in simbolos:
		Y=set(vecinos(x))
		contracciones[x]=Y
					  
	for x in epsiloncerraduras.keys():
		for y in epsiloncerraduras.keys():
			if x in epsiloncerraduras[y]:
				epsiloncerraduras[y]=epsiloncerraduras[y]|epsiloncerraduras[x]

