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

def Gram_Schmidt(Basis):
	return Basis

def main():
	f = open(PATH, 'r')  # lattice file containing unreduced basis
	challenge_dimension = int(f.readline())
	reference_dimesion = int(f.readline())
	modulus_q = int(f.readline())

	B = []
	print("Start Loop")
	for line in f:
		basis_vector = re.split("\W", line)
		# print(basis_vector)
		basis_vector = [ x for x in basis_vector if x.isdigit() ]  # clean basis vector
		# print(basis_vector)
		basis_vector = [int(v) for v in basis_vector]
		# print(basis_vector)
		if (len(basis_vector) > 0):
			B.append(basis_vector)
	print("End of Loop")

	f.close()
	
	Basis = np.asarray(B, dtype=int)
	print(Basis.shape)
	assert(np.linalg.det(Basis) != 0)

def LLL(B,delta=1.0/4):
	for i in range(2, len(B)+1):
		B_G =B
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

main()
