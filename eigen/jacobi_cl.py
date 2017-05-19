"""
This is the Jacobi Classical module
the input array is assumed to be real and symmetric
Iterating on jacobi() results in a diagonal matrix (of eigenvalues)
Here is an example:

>>> a = array([[ 1.41661717,  0.68604222,  0.79383087],
...            [ 0.68604222,  0.83396481,  0.83610701],
...            [ 0.79383087,  0.83610701,  0.19235399]])
>>> 
>>> for i in range(10):
...     a = jacobi(a)
...     print(off(a))
... 
2.20164275557
0.1410452601
0.0702475605258
0.000310734313296
1.48591033462e-05
5.49097128878e-10
4.92800492884e-16
1.76185742945e-25
3.35759108329e-33
3.35759108329e-33
>>> a
array([[  2.44031974e+00,   3.50324616e-46,   1.67864254e-51],
       [ -5.03197756e-17,   4.39446423e-01,  -2.32857226e-21],
       [ -2.51002319e-17,  -1.39817601e-17,  -4.36830189e-01]])
"""

from numpy import *

# performs one classical Jacobi step, zeroing out two off-diagonal elements
def jacobi(a):
    n,_ = shape(a)
    j,k = max_offdiag(a) 
    p,q,r = a[j,j], a[j,k], a[k,k]
    th = arctan2(2.0*q,p-r)/2.0
    c, s = cos(th), sin(th)
    J = eye(n,n) 
    J[j,j] = c
    J[k,k] = c
    J[j,k] = -s
    J[k,j] = s
    return J.T.dot(a).dot(J)

# returns the sorted indices of the max off-diagonal element
def max_offdiag(a):
    n,_ = shape(a)
    mx = -1.0e10
    for i in arange(n-1) + 1:
        for j in range(i):
            aa = abs(a[i,j])
            if aa > mx:
                mxi, mxj = i,j
                mx = aa
    std = sort([mxi, mxj]) # the order of these is important!
    return std[0], std[1]

# square of Frobenius norm of offdiagonal elements
def off(a):
    return ((a - diag(diag(a)))**2).sum()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

