## Runs LLL on a lattice basis of small dimension.

"""
=================================
| Format of the challenge files |
=================================

line #1:	Challenge lattice dimension m
line #2:	Reference dimension n(m)
line #3:	Modulus q
line #4:	Challenge lattice basis

All entries are in NTL I/O [http://www.shoup.net/ntl] format.

The PI source file contains the sequence pi_1, pi_2, ... as defined in [2].

"""

import numpy as np
import re

# PATH = "Lattices/challenge-200.txt"  # small challenge with dimension 200
PATH = "Lattices/challenge-200.txt"


def gs_coeff(v1, v2):
	return np.dot(v2, v1)/np.dot(v1, v1)

## Takes in a lattice basis and
## returns a Gram-Schmidt orhtogonal basis
## All in numpy
## ref: https://gist.github.com/iizukak/1287876
def Gram_Schmidt(Basis):
	(m, n) = Basis.shape
	GS_Basis = []  # this will store the Gram-Schmidt basis
	for k in range(n):
		w = Basis[:, k]  # at the end of the subsequent loop, this will store the 
		for gs_vec in GS_Basis:
			proj_vec = map(lambda x : x * gs_coeff(gs_vec, Basis[:, k]), gs_vec)
			w = map(lambda x, y : x - y, w, proj_vec)
		GS_Basis.append(w)
	return np.asarray(GS_Basis)

def main():
	f = open(PATH, 'r')  # lattice file containing unreduced basis
	challenge_dimension = int(f.readline())
	reference_dimesion = int(f.readline())
	modulus_q = int(f.readline())

	B = []
	for line in f:
		basis_vector = re.split("\W", line)
		basis_vector = [ x for x in basis_vector if x.isdigit() ]  # clean basis vector
		basis_vector = [int(v) for v in basis_vector]
		if (len(basis_vector) > 0):
			B.append(basis_vector)
	f.close()

	Basis = np.asarray(B, dtype=int)
	assert(np.linalg.det(Basis) != 0)



###################################################################################################
########################################### TESTS #################################################
###################################################################################################

def GS_TEST():
	print("MY OUTPUT:")
	#Test Gram_Schmidt
	test = np.array([[3.0, 1.0], [2.0, 2.0]])
	test2 = np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])
	print np.array(Gram_Schmidt(test))
	print np.array(Gram_Schmidt(test2))

	print("\nCORRECT OUTPUT:")
	def gs_cofficient(v1, v2):
	    return np.dot(v2, v1) / np.dot(v1, v1)

	def multiply(cofficient, v):
		# return v * cofficient
	    return map((lambda x : x * cofficient), v)

	def proj(v1, v2):
	    return multiply(gs_cofficient(v1, v2) , v1)
	    # return (v1 * gs_cofficient(v1, v2))

	def gs(X):
	    Y = []
	    for i in range(len(X)):
	        temp_vec = X[i]
	        for inY in Y :
	            proj_vec = proj(inY, X[i])
	            temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
	        Y.append(temp_vec)
	    return Y

	#Test data
	test = np.transpose(np.array([[3.0, 1.0], [2.0, 2.0]]))
	test2 = np.transpose(np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]]))
	print np.array(gs(test))
	print np.array(gs(test2))
# GS_TEST()  # Uncomment to test gram-schmidt implementation