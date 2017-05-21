"""
Basic power method for computing the dominant eigenvalue
and an associated eigenvector of a symmetric matrix

>>> A = array([[ 1.56260001,  0.22898325,  1.04332261,  0.4558478 ,  0.89455427],
...            [ 0.22898325,  0.54532437,  1.51480666,  0.96986641,  1.5202461 ],
...            [ 1.04332261,  1.51480666,  0.53536642,  0.77569108,  1.23270255],
...            [ 0.4558478 ,  0.96986641,  0.77569108,  1.6110797 ,  1.32133154], 
...            [ 0.89455427,  1.5202461 ,  1.23270255,  1.32133154,  0.27904888]])

>>> lam, v, iters, err = power(A,True)
>>> iters < 30
True

>>> err < 1.0e-15
True
"""

from numpy import *

def power(A, details=False):
    n,_ = shape(A)
    v = random.rand(n)
    iters = 0
    err = 1.0
    while err > 1.0e-15 and iters < 1000:
        w = A.dot(v)
        v = w / linalg.norm(w)
        lam = v.dot(A).dot(v) / v.dot(v)
        err = linalg.norm(A.dot(v) - lam*v) 
        iters += 1
    if details:
        return lam, v, iters, err
    else:
        return lam, v

if __name__ == "__main__":
    import doctest
    doctest.testmod()
