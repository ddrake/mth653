from numpy import *

"""
QR Factorization via Classical Gram Schmidt


>>> a = array([[1 + 2.1j, 2.1 + 3.1j],[2.7 + .1j, 3.1 + .2j],[4.1, 2.3j]])
>>> a
array([[ 1.0+2.1j,  2.1+3.1j],
       [ 2.7+0.1j,  3.1+0.2j],
       [ 4.1+0.j ,  0.0+2.3j]])
>>> q,r = gs(a)
>>> q.dot(r)
array([[  1.00000000e+00+2.1j,   2.10000000e+00+3.1j],
       [  2.70000000e+00+0.1j,   3.10000000e+00+0.2j],
       [  4.10000000e+00+0.j ,  -1.51996096e-16+2.3j]])
"""

eps = 2**-52
def norm(v):
    """return the 2-norm of a complex vector
    """
    return sqrt(abs(inner(v,v)))

def inner(v,w):
    """return the inner product of two vectors
       using the convention of conjugating the first argument
    """
    return v.conj().dot(w)

def gs(a):
    m,n = shape(a)
    r = zeros((n,n),dtype='complex')
    q = zeros((m,n),dtype='complex')
    v = zeros((m,n),dtype='complex')
    for j in range(n):
        v[:,j] = a[:,j]
        update(a[:,j],v,q,r,j)
        if r[j,j] < eps:
            aj = random.rand(m)
            v[:,j]=aj
            update(aj,v,q,r,j)
        q[:,j] = v[:,j] / r[j,j]
    return q, r

def update(aj,v,q,r,j):
    for i in range(j):
        r[i,j] = inner(q[:,i], aj)
        v[:,j] -= r[i,j]*q[:,i]
    r[j,j] = norm(v[:,j])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
