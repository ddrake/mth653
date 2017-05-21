"""
The QR iteration for eigenvalues
A 'jewel' of numerical analysis
Requires the input matrix A to be symmetric
Note that the precision is weaker that the other algorithms
While this algorithm gets close to the eigenspace very quickly,
it may not converge if the error bound is too tight for a given matrix.


>>> A = array([[ 0.328,  0.724,  1.227,  1.52 ,  0.323],
...            [ 0.724,  0.347,  1.422,  1.748,  0.712],
...            [ 1.227,  1.422,  1.432,  0.666,  0.862],
...            [ 1.52 ,  1.748,  0.666,  1.366,  1.07 ],
...            [ 0.323,  0.712,  0.862,  1.07 ,  0.063]])

>>> l, V, iters, err = qr_it(A, details=True)
>>> iters < 200
True

"""

from numpy import *


def qr_it(A, details=False):
    """
    Computes the QR factorization of A, then multlplies the factors
    in the reverse order
    """
    n,_ = shape(A)
    T = A.copy()
    V = eye(n)
    iters = 0
    err = 1.0
    while err > 1.0e-13 and iters < 2000:
        Q,R = linalg.qr(T)
        T = R.dot(Q)
        V = V.dot(Q)
        iters += 1
        err = linalg.norm(A.dot(V) - V.dot(diag(diag(T))),'f')
    if details:
        return diag(T), V, iters, err
    else:
        return diag(T), V



if __name__ == "__main__":
    import doctest
    doctest.testmod()
