from numpy import *

"""
QR Factorization via Classical Gram Schmidt


>>> a = array([[1 + 2.1j, 2.1 + 3.1j],[2.7 + .1j, 3.1 + .2j],[4.1, 2.3j]])
>>> a
array([[ 1.0+2.1j,  2.1+3.1j],
       [ 2.7+0.1j,  3.1+0.2j],
       [ 4.1+0.j ,  0.0+2.3j]])
>>> q,r = gsm(a)
>>> q.dot(r)
array([[  1.00000000e+00+2.1j,   2.10000000e+00+3.1j],
       [  2.70000000e+00+0.1j,   3.10000000e+00+0.2j],
       [  4.10000000e+00+0.j ,  -1.51996096e-16+2.3j]])

>>> abs(inner(q[:,0],q[:,1]))
3.7341307954279784e-16
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

def gsm(a):
    m,n = shape(a)
    r = zeros((n,n),dtype='complex')
    q = zeros((m,n),dtype='complex')
    v = zeros((m,n),dtype='complex')
    for i in range(n):
        v[:,i] = a[:,i]
    for i in range(n):
        r[i,i] = norm(v[:,i])
        q[:,i] = v[:,i]/r[i,i]
        for j in range(i+1,n):
            r[i,j] = inner(q[:,i],v[:,j])
            v[:,j] -= r[i,j]*q[:,i]
    return q, r

if __name__ == "__main__":
    import doctest
    doctest.testmod()
