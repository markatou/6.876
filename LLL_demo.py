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

"""
IMPORTANT: All matrices are assumed to be in _ROW_MAJOR_ form
"""

import numpy as np
import re
import sys

RED = "\033[1;31m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"

# PATH = "Lattices/challenge-200.txt"  # small challenge with dimension 200
PATH = "Lattices/challenge-200.txt"

# computes the gram schmidt coefficent of v1 and v2
# the orientation is: <v2, v1>/<v2, v2>
# mu_ij, where v1 = b_i and v2 = b_j 
def gs_coeff(v1, v2):
	try:
		return float(np.dot(v2, v1))/float(np.dot(v2, v2))
	except ValueError:
		return float(np.dot(v1, v2))/float(np.dot(v2, v2))

## Takes in a lattice basis and
## returns a Gram-Schmidt orhtogonal basis
## All in numpy
def Gram_Schmidt(Basis):
	(m, n) = Basis.shape
	GS_Basis = []  # this will store the Gram-Schmidt basis
	GS_Basis.append(Basis[:, 0])
	for k in range(1, n):
		# print("GS[" + str(k-1) + "] = " + str(GS_Basis[k-1]))
		w = Basis[:, k]  # at the end of the subsequent loop, this will store the 
		for gs_vec in GS_Basis:
			proj_vec = map(lambda x : x * gs_coeff(Basis[:, k], gs_vec), gs_vec)
			w = map(lambda x, y : x - y, w, proj_vec)
		GS_Basis.append(w)
	return np.transpose(np.asarray(GS_Basis))

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

# component-wise swap of two vectors
def swap(v1, v2):
	temp = map(lambda v: v, v1)
	v1 = map(lambda v: v, v2)
	v2 = map(lambda v: v, temp)
	return v1, v2

# code taken from wikipedia pseudocode
def LLL(Basis, delta = 3.0/4.0):
	assert(np.linalg.det(Basis) != 0)
	(m,n) = Basis.shape
	GS_Basis = Gram_Schmidt(Basis)
	i = 1
	while i < n:
		for j in range(i-1, -1, -1):
			c_ij = gs_coeff(Basis[:, i], GS_Basis[:, j])
			if (abs(c_ij) > 1.0/2):
				scale_b_j = Basis[:, j] * round(c_ij)
				Basis[:, i] = map(lambda x, y: x - y, Basis[:, i], scale_b_j)
				GS_Basis = Gram_Schmidt(Basis)  # these must be updated, but this method is naive
		left_lovasz = np.dot(GS_Basis[:, i], GS_Basis[:, i])/np.dot(GS_Basis[:, i-1], GS_Basis[:, i-1])
		if  left_lovasz >= (delta - (gs_coeff(Basis[:, i], GS_Basis[:, i-1]))**2):
			i += 1
		else:
			Basis[:, i], Basis[:, i-1] = swap(Basis[:, i], Basis[:, i-1])
			i = max(i-1, 1)
			GS_Basis = Gram_Schmidt(Basis)
	return Basis

# def LLL(B,delta=1.0/4):
# 	again = True
# 	print("In LLL")
# 	while again:
# 		B_G = Gram_Schmidt(B)
# 		B = compute_LLL_basis(B,B_G)
# 		B,again = Lovasz_consdition(B,B_G,delta)
# 	return B

# def compute_LLL_basis(B,B_G):
# 	for i in range(1, len(B)):
# 		for j in reversed(range(0,i-1)):
# 			m = np.inner(B[:,i],B_G[:,j])/np.inner(B_G[:,j],B_G[:,j])
# 			B[:,i] = np.subtract(B[:,i],np.multiply(m,B[:,j]))
# 	return B

# def Lovasz_consdition(B,B_G,delta):
# 	for i in range(0,len(B)-1):
# 		m = np.inner(B[:,i+1],B_G[:,i])/np.inner(B_G[:,i],B_G[:,i])
# 		if np.linalg.norm(B_G[:,i+1])**2 < (delta - m**2)*np.linalg.norm(B_G[:,i])**2:
# 			for k in range(len(B[:,1])):
# 				c = B[i,k]
# 				B[i,k] = B[i+1,k]
# 				B[i+1,k] = c
# 				return B,True
# 	return B,False


###################################################################################################
########################################### TESTS #################################################
###################################################################################################


# ref: https://en.wikipedia.org/wiki/
#		Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm#Example
def LLL_test(verbose=False):
	print("\nLLL Test...")
	test = np.array([[1, -1, 3], [1, 0, 5], [1, 2, 6]])
	ours = LLL(test)
	correct = np.array([[0, 1, -1], [1, 0, 0], [0, 1, 2]])
	if (np.array_equal(correct,ours)):
		sys.stdout.write(GREEN)
		print("PASS")
		sys.stdout.write(RESET)
	else:
		sys.stdout.write(RED)
		print("FAIL")
		sys.stdout.write(RESET)	
	if (verbose):
		print("OUR OUTPUT:")
		print(ours)
		print("\nCORRECT OUTPUT:")
		print(correct)

# Based on reference implementation here:
# ref: https://gist.github.com/iizukak/1287876
def GS_TEST_1(verbose=False):
	print("\nGram_Schmidt Test 1...")
	#Test Gram_Schmidt
	# Reference Implementation
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
	ours_1 = Gram_Schmidt(np.array([[3.0, 1.0], [2.0, 2.0]]))
	ours_2 = Gram_Schmidt(np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]]))
	correct_1 = np.transpose(np.array(gs(test)))
	correct_2 = np.transpose(np.array(gs(test2)))
	if (np.array_equal(correct_1, ours_1) and np.array_equal(correct_2, ours_2)):
		sys.stdout.write(GREEN)
		print("PASS")
		sys.stdout.write(RESET)
	else:
		sys.stdout.write(RED)
		print("FAIL")
		sys.stdout.write(RESET)	
	if (verbose):
		print("OUR OUTPUT:")
		print np.array(ours_1)
		print np.array(ours_2)
		print("\nCORRECT OUTPUT:")
		print correct_1	 
		print correct_2

# Another Gram Schmidt test
# https://www.math.hmc.edu/calculus/tutorials/gramschmidt/
def GS_Test_2(verbose=False):
	print("\nGram-Schmidt Test 2...")
	test = np.asarray([[1, 1, 1],[-1, 0, 1],[1, 1, 2]])
	ours = Gram_Schmidt(test)
	correct = np.asarray([[1, 1.0/3.0, -1.0/2.0],[-1, 2.0/3.0, 0],[1, 1.0/3.0, 1.0/2.0]])
	if (np.isclose(correct, ours).all()):  # needed for floating point equality testing
		sys.stdout.write(GREEN)
		print("PASS")
		sys.stdout.write(RESET)
	else:
		sys.stdout.write(RED)
		print("FAIL")
		sys.stdout.write(RESET)	
	if (verbose):
		print("OUR OUTPUT:")
		print(ours)
		print("\nCORRECT OUTPUT:")
		print(correct)
		print(np.isclose(correct, ours))


def test_all(verbose=False):
	GS_TEST_1(verbose)
	GS_Test_2(verbose)
	LLL_test(verbose)

### TEST CALLS 
## Boolean argument for verbose output
test_all()
# GS_TEST_1(True) 
# GS_Test_2(True)
#LLL_test(True)