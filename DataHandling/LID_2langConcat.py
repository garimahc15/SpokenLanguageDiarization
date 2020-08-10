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


########################################################################

os.chdir("/home/administrator/SLD_19/garsh/trainLID/hinpunLID")  #set the working directory

extension='csv'    #use glob to match the pattern csv in directory
files_list= [i for i in glob.glob('*.{}'.format(extension))]

#folders = glob.glob('/home/administrator/SLD_19/Garima/LID_files/hintel_mix/*')  # Path for files
#files_list = []
#for folder in folders:
 #   for f in glob.glob(folder+'/*.csv'):
  #      files_list.append(f)
        
#print files__list
T=len(files_list)
print('Total Training files: ',T)
print(files_list)

path = "/home/administrator/SLD_19/garsh/trainLID/combinedLID.csv"  # Path to save the concatted csv file

#rang= [i for i in range(129)] #usecols
#df= pd.read_csv(files_list[1])
#df = df.drop(df.columns[0], axis=1)
#print(df)

#dropping 1st column and exporting to csv file
df1= pd.read_csv(files_list[0])
print("*************df1 with index col : \n", df1.head())
df1 = df1.drop(df1.columns[0], axis=1)
print("****************************df1 without index col : \n", df1.head())
for f in range(1, len(files_list)):
	print("##############################", f, "#############")
	df= pd.read_csv(files_list[f])
	#print("*************df with index col : \n", df.head())
	df = df.drop(df.columns[0], axis=1)
	#print("****************************df1 without index col : \n", df1.head())
	combined_csv= pd.concat([df, df1])
	df1=combined_csv.copy()

print(combined_csv.shape)

#print(pd.read_csv("/home/administrator/SLD_19/Garima/LID_files/hintel_mix/"+files_list[0]))

#combined_csv= pd.concat([pd.read_csv(f) for f in files_list])   #combine all files in list


combined_csv.to_csv(path, index=False, encoding='utf-8-sig')  #export to csv
