"""spectral clustering on test cases directly. no prior training. 
involves eigen gap heurestics to find optimum no. of clusters"""

import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from numpy import linalg as la
#from spectralcluster import utils   """module isn't installed"""
from sklearn.metrics import pairwise_distances

subpath = "/home/administrator/SLD_19/garsh/testhintelLID/hintel"
predlabels=[]
clustern=[]
for f in range(750):
	labels=[]
	fn = subpath+str(f+1)+'_bnf.csv'
	print(fn)
	dfTest= pd.read_csv(fn)
	dfTest=dfTest.drop(dfTest.columns[0], axis=1)
	#print(dfTest.head())
	print(dfTest.shape)
	dataTest= np.array(dfTest)
	print(dataTest.shape)
	#am= affinity matrix
	#  Compute affinity matrix.
	"""am = utils.compute_affinity_matrix(X)"""
	#W= adjacency matrix
	W = pairwise_distances(dataTest, metric="euclidean")
	rdist= W*W
	#print(W.shape)
	#print(rdist.std())
	am= np.exp(-(rdist))
	#print(am.shape)
	#print(am[0])
	eigenvalues, eigenvectors= la.eig(am)   # each col of ndarray eigenvectors represents 1 eigvec
	#print("eig vec : ", vec)

	#sort eigenvectors based on eigenvalues
	eigenvectors= eigenvectors[:, np.argsort(eigenvalues)]
	eigenvalues= eigenvalues[np.argsort(eigenvalues)]
        #k= optimum_clusters
	#print(np.diff(eigenvalues))
	index_largest_gap= np.argmax(np.diff(eigenvalues))
	k= index_largest_gap + 1
	print(k)
	clustern.append(k)
	spectral_embeddings = eigenvectors[:, :k]

	kmeans_clusterer = KMeans(n_clusters=k, init="k-means++", max_iter=300, random_state=0)
	labels = kmeans_clusterer.fit_predict(spectral_embeddings)
	predlabels.append(labels)
	print("Done ", f)

predlabels= np.array(predlabels)
print(predlabels.shape)

np.save('/home/administrator/SLD_19/garsh/spectralData/num_clustershintel', clustern)
np.save('/home/administrator/SLD_19/garsh/spectralData/pred_hintel', predlabels)
df= pd.DataFrame(predlabels)
df.to_csv('/home/administrator/SLD_19/garsh/spectralData/pred_hintel.csv')
print(clustern)




