import gsm
import gs
from numpy import *
from matplotlib import pyplot as plt

"""Experiment 2 in Trefethen & Bau
   Demonstrates the instability of the Classical Gram-Schmidt
   algorithm compared with the Modified algorithm
"""
m=80

# create mxm unitary matrices U and V
[U,X] = linalg.qr(random.rand(m,m))
[V,X] = linalg.qr(random.rand(m,m))

# construct an array of eigenvalues growing exponentially smaller
S = diag(2.**(-1-array(range(m))))

# construct matrix A with desired eigenvalues
A = U.dot(S).dot(V)

# compare the Gram-Schmidt algorithms
QC,RC = gs.gs(A)
QM,RM = gsm.gsm(A)

# The classical algorithm fails around the square root of machine epsilon
plt.semilogy(range(m), diag(RC),'bs',diag(RM),'ro')
plt.show()

