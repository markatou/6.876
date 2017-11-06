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

PATH = "Lattices/challenge-200.txt"  # small challenge with dimension 200

def main():
	f = open(PATH, 'r')  # lattice file containing unreduced basis
	challenge_dimension = int(f.readline())
	reference_dimesion = int(f.readline())
	modulus_q = int(f.readline())

	f.readline()
	second_last_line, last_line = f.readline(), f.readline()
	B = []
	for line in f:
		#print(second_last_line)
		basis_vector = re.split("\ |[|]", second_last_line)
		basis_vector = basis_vector[1:challenge_dimension-1]
		basis_vector = [int(v) for v in basis_vector]
		B.append(basis_vector)
		second_last_line = last_line
		last_line = line
		# print(re.split(',|[|]|\n', line)[0])
	print(B)
	f.close()

main()