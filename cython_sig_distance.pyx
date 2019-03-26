import numpy as np
import math

def sig_distance(double[:] sig1, double[:] sig2):

	cdef int n = sig1.shape[0]
	cdef int i, k
	cdef int width = 2
	cdef int depth = (np.log(n*(width-1)+1) / np.log(2)) - 1

	cdef double out = 0
	cdef double out_k
	cdef int ind1 = 1
	cdef int ind2 	

	for k in range(1, depth+1):
		ind2 = int(width**(k+1)-1)
		out_k = 0
		for i in range(ind1, ind2):
			out_k += (sig1[i] - sig2[i])*(sig1[i] - sig2[i])
		out += out_k**(1./k)
		ind1 = ind2
	return out