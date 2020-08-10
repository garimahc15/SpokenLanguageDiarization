# -*- coding: utf-8 -*-
"""

"""
from __future__ import division
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import pandas as pd
import glob
import random

from torch.autograd import Variable
from torch import optim, nn

from sklearn.cluster import KMeans
from sklearn import metrics

########################################################################

df= pd.read_csv("/home/administrator/SLD_19/Garima/LID_files/combined_csv.csv")
traindata= np.array(df)
kmeans= KMeans(n_clusters=2).fit(traindata)
centroids= kmeans.cluster_centers_
#labels= kmeans.labels_

print("dimensions of centroid : ", len(centroids[0]))
print("************** 1st centroid ******** \n", centroids[0], "\n ****** 2nd centroid ******** \n", centroids[1])
dftest= pd.read_csv("/home/administrator/SLD_19/Garima/LID_files/test_LID/testhintelLID_combined.csv")
classcol= dftest.iloc[:,128:129]
dftest= dftest.iloc[:,0:128]
dftestdata= np.array(dftest)

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

y_true= classcol
print("true class : \n", y_true)
y_pred = kmeans.predict(dftestdata)

print("purity score : ", purity_score(y_true, y_pred))



