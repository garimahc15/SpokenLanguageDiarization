"""spectral clustering on test cases directly. no prior training. 
involves eigen gap heurestics to find optimum no. of clusters"""

import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from numpy import linalg as la
#from spectralcluster import utils   """module isn't installed"""
from sklearn.metrics import pairwise_distances
from scipy.ndimage import gaussian_filter

p=85
soft_threshold=0.01
print("p = ",p, "threshold= ", soft_threshold)
def cos_sim(M):   #M is data matrix
	l= len(list(M))   #no. of rows
	b= [[0]*l for m in range(l)]
	for r1 in range(l):
		for r2 in range(l):
			if r1!=r2:
				b[r1][r2]= np.sum(M[r1]*M[r2])/((np.sqrt(np.sum(M[r1]*M[r1])))*(np.sqrt(np.sum(M[r2]*M[r2]))))
	b= np.array(b)
	for r1 in range(l):
		b[r1][r1]= np.max(b[r1])
	return b
def refine(M):
	#gsnblr
	gb= gaussian_filter(M, M.std())
	#rwtrshld
	for l in range(len(list(M))):
		p_percentile= np.percentile(gb[l],p)
		for l2 in range(len(list(M))):
			if gb[l][l2]<p_percentile:
				gb[l][l2]=gb[l][l2]*soft_threshold

	#sym
	gb= np.maximum(gb, np.transpose(gb))

	#dsn
	gb= np.matmul(gb, np.transpose(gb))

	#rwmxnrml
	row_max = gb.max(axis=1)
	gb /= np.expand_dims(row_max, axis=1)
	return gb

#subpath = "/home/administrator/SLD_19/garsh/testhintelLID/hintel"
subpath= "/home/administrator/SLD_19/garsh/VAD/LID/hintel/hintel"
predlabels=[]
clustern=[]
for f in range(750):
	labels=[]
	#fn = subpath+str(f+1)+'_bnf.csv'
	fn = subpath+str(f+1)+'.csv'
	#print(fn)
	dfTest= pd.read_csv(fn)
	#dfTest=dfTest.drop(dfTest.columns[0], axis=1)
	#print(dfTest.head())
	#print(dfTest.shape)
	dataTest= np.array(dfTest)
	#print(dataTest.shape)
	#AM= affinity matrix
	#  Compute affinity matrix.

	AM= cos_sim(dataTest)
	#print(AM)
	#print(AM.shape)
	#df=pd.DataFrame(AM)
	#df.to_csv('/home/administrator/SLD_19/garsh/AM.csv')
	#print(AM[0])
	am= refine(AM)
	#df.to_csv('/home/administrator/SLD_19/garsh/AMrefined.csv')
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
	spectral_embeddings = eigenvectors[:, k:]
	opt_clusters= len(list(AM))-k
	clustern.append(opt_clusters)
	kmeans_clusterer = KMeans(n_clusters=opt_clusters, init="k-means++", max_iter=300, random_state=0)
	labels = kmeans_clusterer.fit_predict(spectral_embeddings)
	predlabels.append(labels)

predlabels= np.array(predlabels)
#print(predlabels.shape)

#np.save('/home/administrator/SLD_19/garsh/VAD/num_clustershintel', clustern)
#np.save('/home/administrator/SLD_19/garsh/VAD/predLabels_hintel', predlabels)
#df= pd.DataFrame(predlabels)
#df.to_csv('/home/administrator/SLD_19/garsh/VAD/predLabels_hintel.csv', index=False)
#print(clustern)
unique_elements, counts_elements = np.unique(clustern, return_counts=True)
print(unique_elements, "\n", counts_elements)
