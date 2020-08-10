from scipy import optimize
import numpy as np

#hypo = [[2, [0.0, 0.415]], [8, [0.415, 0.815]], [5, [0.815, 1.215]], [1, [1.215, 2.015]], [4, [2.015, 2.415]], [12, [2.415, 2.815]], [9, [2.815, 3.215]], [7, [3.215, 3.615]], [0, [3.615, 4.015]], [11, [4.015, 4.415]], [6, [4.415, 4.815]], [0, [4.815, 5.215]], [3, [5.215, 5.615]], [1, [5.615, 6.415]], [10, [6.415, 6.815]], [1, [6.815, 7.165]]]

#ref = [[0,[0,1.88875]],	[1,[1.88875,3.8615]], [0,[3.8615,5.548875]],[1,[3.8615,7.16625]]]

def lengthWav(arr):
	total_length = 0.0
	for element in arr:
		total_length += element[1][1] - element[1][0]
	return total_length
#print(lengthWav(ref))
#print(lengthWav(hypo))

def commonLen(A, B):
	max_start = max(A[0], B[0])
	min_end = min(A[1], B[1])
	return max(0.0, min_end - max_start)

def mapping(ref,hyp):
	h = []
	r = []
	for i in hyp:
		h.append(i[0])
	for j in ref:
		r.append(j[0])
	#print(np.unique(h))
	#print(np.unique(r))
	cost_matrix = np.zeros((len(np.unique(r)),len(np.unique(h))))
	#print(cost_matrix)
	rindx = {lang: i for i, lang in enumerate(np.unique(r))}
	hindx = {lang: i for i, lang in enumerate(np.unique(h))}
	#print(rindx)
	#print(hindx)
	for ref_element in ref:
		for hyp_element in hyp:
			i = rindx[ref_element[0]]
			j = hindx[hyp_element[0]]
			cost_matrix[i, j] += commonLen(ref_element[1],hyp_element[1])
	#print(cost_matrix)
	return cost_matrix
	
def compute_merged_total_length(ref, hyp):
	# Remove language label and merge.
	merged = [(element[1][0], element[1][1]) for element in (ref + hyp)]
	# Sort by start.
	merged = sorted(merged, key=lambda element: element[0])
	num_elements = len(merged)
	for i in reversed(range(num_elements - 1)):
		if merged[i][1] >= merged[i + 1][0]:
			max_end = max(merged[i][1], merged[i + 1][1])
			merged[i] = (merged[i][0], max_end)
			del merged[i + 1]
	total_length = 0.0
	for element in merged:
		total_length += element[1] - element[0]
	return total_length

	
def DER(ref, hypo):
	refLen = lengthWav(ref)
	costMatrix = mapping(ref,hypo)
	row_indx, col_indx = optimize.linear_sum_assignment(-costMatrix)
	#print(row_indx)
	#print(col_indx)
	optimal_match_overlap = costMatrix[row_indx, col_indx].sum()
	union_total_length = compute_merged_total_length(ref, hypo)
	der = (union_total_length - optimal_match_overlap) / refLen
	print(der)
	return der


ref = np.load('/home/administrator/SLD_19/garsh/test/hintel/stitch/hintel_langChange.npy')
hypo = np.load('/home/administrator/SLD_19/garsh/DER/hypoLabels_der.npy', allow_pickle=True)
arrd = [0,1,0,1]
der_val = []
for i in range(2):
	print('_________________',i,'____________________')
	subts = ref[i+1]
	print(subts)
	timest = [0]
	time  = 0
	for i in range(len(subts)):
		timest.append(subts[i])
	#print(timest)
	reference = []
	for i in range(1,len(subts)+1):
		reference.append([arrd[i-1],[timest[i-1], timest[i]]])
	print(reference)
	#print(len(reference))
	print(hypo[i])
	der_val.append(DER(reference, hypo[i]))
	
print(max(der_val))
print(min(der_val))

	


