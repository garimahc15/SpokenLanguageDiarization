#takes predicted labesls and convert them to timestamps where language is changing
import numpy as np
import pandas as pd

files_len = 1
path = np.load("/home/administrator/SLD_19/garsh/spectralData/pred_hintel.npy", allow_pickle=True)
#print(path[0])
#print(type(path))
#path= np.array([[0,1,1,1,1,2,2,2,3,3,3,3,3,2],[2,1,1,1,1,1,1,2,2,2,2,1,1,2]])
arrc=[]
#for i in range(750):

for i in range(files_len):
	subarrc=[]
	flag=0
	c=1
	j=0
	while(flag<=len(path[i]) and j<len(path[i])):
		if(j==len(path[i])-1 and path[i][j]!=path[i][j-1]):
			subarrc.append(1)
			break
		elif(j<len(path[i])-1 and path[i][j]==path[i][j+1]):
			c=c+1
		else:
			flag=flag+c
			subarrc.append(c)
			c=1
		j=j+1
	arrc.append(subarrc)
#print(np.array(arrc).shape)
print("arrradcsas : ", arrc)
hypothesis = []
for i in range(files_len):
	l= len(arrc[i])
	subts= []
	subts.append((25+39*10)+(40*10)*(arrc[i][0]-1))
	for j in range(1,l-1):
		subts.append((40*10)*arrc[i][j])
	subpath = '/home/administrator/SLD_19/garsh/test/hintel/bnf/hintel'+str(i+1)+'_bnf.csv'
	df = pd.read_csv(subpath, header=None)
	print(subpath)
	print(len(df))
	lbnf=len(df)%40
	#lbnf = 715%40
	if lbnf!=0:
		subts.append((40*10)*(arrc[i][l-1]-1)+((lbnf)*10))
	else:
		subts.append((40*10)*(arrc[i][l-1]))

	#print(len(subts))
	#print((sum(subts))*0.001)
	#print(len(path[0]))

	arrd = list(path[i])
	print(arrd)
	
	arrd=[]
	b= [0]*l
	print(i)
	b= [sum(arrc[i][:j+1]) for j in range(l)]
	for j in b:
		arrd.append(path[i][j-1])
	print("gdshjbdsuudc : ", arrd)
	timest = [0]
	time  = 0
	for i in range(1,len(subts)+1):
		timest.append(sum(subts[:i]))
	#print(len(timest))
		
	hypo = []
	for i in range(1,len(subts)+1):
		hypo.append([arrd[i-1],[timest[i-1]*0.001, timest[i]*0.001]])
	print(hypo)
	print(len(hypo))
	#print(np.array(hypo).shape)
	hypothesis.append(hypo)
#print(np.array(hypothesis).shape)
np.save('hypoLabels_der.npy', hypothesis)
