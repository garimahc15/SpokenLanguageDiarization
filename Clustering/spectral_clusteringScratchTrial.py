#################### code From scratch-- result: gave memory error
import numpy as np
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
#from matplotlib import pyplot as plt
#import networkx as nx      #module not installed
#import seaborn as sns	    #module not installed
#sns.set()

dfTrain= pd.read_csv("/home/administrator/SLD_19/Garima/LID_files/combined_csv.csv")  #this location is in dileep sir's gpu
print(dfTrain.shape)
dataTrain= np.array(dfTrain)
print(dataTrain.data.shape)

#W= Adjacency matrix  D= Degree Matrix  L=Laplacian matrix

#Adjacency matrix
W = pairwise_distances(dataTrain, metric="euclidean")         #gives memory error
vectorizer = np.vectorize(lambda x: 1 if x < 5 else 0)
W = np.vectorize(vectorizer)(W)
print("adjacency matrix")
print(W)

#networkx library to visualize graph
#def draw_graph(G):
 #   pos = nx.spring_layout(G)
  #  nx.draw_networkx_nodes(G, pos)
   # nx.draw_networkx_labels(G, pos)
    #nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

# degree matrix
D = np.diag(np.sum(np.array(W.todense()), axis=1))
print('degree matrix:')
print(D)

# laplacian matrix
L = D - W
print('laplacian matrix:')
print(L)


e, v = np.linalg.eig(L)# eigenvalues
print('eigenvalues:')
print(e)# eigenvectors
print('eigenvectors:')
print(v)


U = np.array(v[:, i[1]])
km = KMeans(init='k-means++', n_clusters=3)
km.fit(U)
km.labels_
