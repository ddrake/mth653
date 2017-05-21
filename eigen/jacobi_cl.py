"""
Jacobi Classical algorithm
the input array A is assumed to be real and symmetric
Iterating on jacobi(A) results in a diagonal matrix (of eigenvalues)

>>> A = array([[ 1.41661717,  0.68604222,  0.79383087],
...            [ 0.68604222,  0.83396481,  0.83610701],
...            [ 0.79383087,  0.83610701,  0.19235399]])

>>> l, iters, offnorm = ews(A,True)
>>> iters < 20
True

"""

from numpy import *

def ews(A, details=False):
    """
    Returns the eigenvalues of A
    """
    offnorm = 1.0
    iters = 0
    while offnorm > 1.0e-15 and iters < 1000:
        A = jacobi(A)
        offnorm = off(A)
        iters += 1
    if details:
        return diag(A), iters, offnorm
    else:
        return diag(A)


def jacobi(A):
    """
    performs one classical Jacobi step, zeroing out two off-diagonal elements
    """
    n,_ = shape(A)
    j,k = max_offdiag(A) 
    p,q,r = A[j,j], A[j,k], A[k,k]
    th = arctan2(2.0*q,p-r)/2.0
    c, s = cos(th), sin(th)
    J = eye(n,n) 
    J[j,j] = c
    J[k,k] = c
    J[j,k] = -s
    J[k,j] = s
    return J.T.dot(A).dot(J)

def max_offdiag(A):
    """
    returns the sorted indices of the max off-diagonal element
    """
    n,_ = shape(A)
    mx = -1.0e10
    for i in arange(n-1) + 1:
        for j in range(i):
            aa = abs(A[i,j])
            if aa > mx:
                mxi, mxj = i,j
                mx = aa
    std = sort([mxi, mxj]) # the order of these is important!
    return std[0], std[1]

def off(A):
    """
    square of Frobenius norm of offdiagonal elements
    """
    return linalg.norm(A - diag(diag(A)),'f')


if __name__ == "__main__":
    import doctest
    doctest.testmod()

