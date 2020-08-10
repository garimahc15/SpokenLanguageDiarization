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

import sklearn.metrics

from torch.autograd import Variable
from torch import optim, nn


########################################################################
look_back=40 # For LSTM
NClass=10 # Number of classes
n_classes = 12
seq_length = 40
input_dim = 80
hidden_dim = 128
def lstm_data(f):
    df = pd.read_csv(f,encoding='utf-16',usecols=list(range(0,80)))
    data = df.astype(np.float32)
    X = np.array(data) 
    N,D=X.shape
    print(X.shape)
    Xdata=[] 
    Ydata=[]
    
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1) #so that the standard deviation never becomes zero.
    X = (X - mu) / std # normalize the data
    f1 = os.path.splitext(f)[0]  # gives first name of file without extension
    
    fpath=f1[59:]
    print("fpath:  ",fpath)
    clas=f1[59:62]
    print(clas)
                
    if (clas == 'asm'):
        Y = 0
    elif (clas == 'ben'):
        Y = 1
    elif (clas == 'guj'):
        Y = 2
    elif (clas == 'hin'):
        Y = 3
    elif (clas == 'kan'):
        Y = 4
    elif (clas == 'mal'):
        Y = 5
    elif (clas == 'man'):
        Y = 6

    elif (clas == 'mar'):
        Y = 7 
    elif (clas == 'odi'):
        Y = 8   
  
    elif (clas == 'pun'):
        Y = 9
    elif (clas == 'tel'):
        Y = 10 
    elif (clas == 'urd'):
        Y = 11 

        

    Y=np.array([Y])
    for i in range(len(X)-look_back):            
        a=X[i:(i+look_back),:]        
        Xdata.append(a)        

    Xdata=np.array(Xdata)
    Xdata = torch.from_numpy(Xdata).float()
    Y=torch.from_numpy(Y).long()
    #print('The shape of data after appending look_back:', Xdata.shape)
    return Xdata,Y,fpath

########################################################################


class LSTMNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTM(input_dim, hidden_dim,bidirectional=True)
        self.lstm2 = nn.LSTM(2*hidden_dim, 64,bidirectional=True)
        self.fc1 = nn.Linear(2*64, 32)
        self.fc2 = nn.Linear(32, output_dim, bias=False)

    def forward(self, x,fpath):
        batch_size = x.size()[1]
        h0 = Variable(torch.zeros([1, batch_size, self.hidden_dim]), requires_grad=False) #h_0 of shape (num_layers * num_directions, batch, hidden_size)
        c0 = Variable(torch.zeros([1, batch_size, self.hidden_dim]), requires_grad=False)
        fx, _ = self.lstm1.forward(x) #input of shape (seq_len, batch, input_size): h_0 and c0 of shape (num_layers * num_directions, batch, hidden_size)
        # fx= output of shape (seq_len, batch, num_directions * hidden_size):
        fx, _ = self.lstm2(fx)
        print("fx_size:",fx.size())
        fx=fx[-1]
        
        print("LID_seq_sens size:  ",fx.size())
        
        LSS=fx.detach().numpy()
        LSS_txt = LSS.astype(np.float32)
        
#        np.savetxt('/home/administrator/Muralikrishna_H/LID/IIITH/LID_seq_senones/test_LSS/'+fpath+'.txt', LSS_txt, fmt='%.18e', delimiter=' ')
#        print(fx.size())
        fx= F.tanh(self.fc1(fx))
        fx= self.fc2(fx)
#        print(fx.size())
        return (fx)


def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.item()



def predict(model, x_val,fpath):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x,fpath)
    output=output.data.numpy().mean(axis=0)
    return np.argmax(output)


path =  "/home/administrator/Muralikrishna_H/LID/IIITH/Shrikha/py_models/BLSTM1_2.pth"  #att_VL_1.pth
model=torch.load(path) 

#model = FFAttention()
model.eval()
print(model) 


folders = glob.glob('/home/administrator/Muralikrishna_H/LID/IIITH/test_BNF/*')
files_list = []
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)
        
#print files__list
T=len(files_list)
print('Total Training files: ',T)
random.shuffle(files_list)            
Tru=[]
Pred=[]

i=0
for fn in files_list:
    if i<101:
        print('Reading file: ',fn)
        X, Y ,fpath= lstm_data(fn)                
        X = np.swapaxes(X, 0, 1)  
        P = predict(model,X,fpath)
        
        print("     Predicted class=", P)            
        print("       Actual=  ", Y)
        
        Tru=np.append(Tru,Y)
        Pred=np.append(Pred,P)
        i=i+1
    else:
        break

CM2=sklearn.metrics.confusion_matrix(Tru, Pred)
print(CM2)
acc=sklearn.metrics.accuracy_score(Tru,Pred)
print('Accuracy= ',acc*100)



