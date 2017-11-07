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
	print(LLL(Basis))
	assert(np.linalg.det(Basis) != 0)


def LLL(B,delta=1.0/4):
	print("In LLL")
	for i in range(2, len(B)+1):
		print(i)
		B_G = Gram_Schmidt(B)
		for j in reversed(range(1,i)):
		    #B[i] = np.subtract(B[i],m[i,j]*B[i-1])
			B[i-1] = np.subtract(B[i-1],np.multiply(0.5,B[j-1]))
	for i in range(1,len(B)-1):
		m = np.inner(B[:,i+1],B_G[:,i])/np.inner(B_G[:,i],B_G[:,i])
		if np.linalg.norm(B_G[i])**2 < (delta - m**2)*np.linalg.norm(B_G[i-1])**2:
			for k in range(len(B[:,1])):
				c = B[i,k]
				B[i,k] = B[i+1,k]
				B[i+1,k] = c
			return B
			    	
	return B

# main()
###################################################################################################
########################################### TESTS #################################################
###################################################################################################

# ref: https://en.wikipedia.org/wiki/
#		Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm#Example
def LLL_test():
	print("\nLLL Test...")
	test = np.array([[1, -1, 3], [1, 0, 5], [1, 2, 6]])
	print("OUR OUTPUT:")
	print(LLL(test))
	print("\nCORRECT OUTPUT:")
	print(np.array([[0, 1, -1], [1, 0, 0], [0, 1, 2]]))

# Based on reference implementation here:
# ref: https://gist.github.com/iizukak/1287876
def GS_TEST():
	print("\nGram_Schmidt Test...")
	print("OUR OUTPUT:")
	#Test Gram_Schmidt
	test = np.array([[3.0, 1.0], [2.0, 2.0]])
	test2 = np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])
	print np.array(Gram_Schmidt(test))
	print np.array(Gram_Schmidt(test2))

	print("\nCORRECT OUTPUT:")
	def gs_cofficient(v1, v2):
	    return np.dot(v2, v1) / np.dot(v1, v1)

	def multiply(cofficient, v):
	    return map((lambda x : x * cofficient), v)

	def proj(v1, v2):
	    return multiply(gs_cofficient(v1, v2) , v1)

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
	# the transposes are because we do gram schimdt on columns
	# while this test originally did gram schmidt on rows
	test = np.transpose(np.array([[3.0, 1.0], [2.0, 2.0]]))
	test2 = np.transpose(np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]]))
	print np.array(gs(test))
	print np.array(gs(test2))

### TEST CALLS 
GS_TEST()  # Uncomment to test gram-schmidt implementation
LLL_test()