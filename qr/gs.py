import numpy as np

"""
QR Factorization via Classical Gram Schmidt (unstable)

Example 1: A is a complex 3x2 matrix with full column rank
>>> A = np.array([[1 + 2.1j, 2.1 + 3.1j],[2.7 + .1j, 3.1 + .2j],[4.1, 2.3j]])
>>> Q,R = gs(A)
>>> np.linalg.norm(Q.dot(R) - A) < eps
True
>>> np.abs(inner(Q[:,0],Q[:,1])) < 2.*eps
True
>>> np.abs(inner(Q[:,1],Q[:,1]) - 1.) < eps
True

Example 2 A is a 3x3 real matrix with linearly dependent columns
>>> A = np.array([1.,2.,3.])[:,np.newaxis]
>>> A = A.dot(A.T)
>>> Q,R = gs(A)
lin dep column found with j=1
lin dep column found with j=2
>>> np.linalg.norm(Q.dot(R) - A) < 32*eps
True
>>> np.abs(inner(Q[:,0],Q[:,1])) < eps
True
>>> np.abs(inner(Q[:,1],Q[:,1]) - 1.) < 2*eps
True

"""

eps = 2**-52

def inner(v,w):
    """return the inner product of two vectors
       using the convention of conjugating the first argument
    """
    return v.conj().dot(w)

def gs(a):
    m,n = np.shape(a)
    tp = a.dtype
    r = np.zeros((n,n),dtype=tp)
    q = np.zeros((m,n),dtype=tp)
    v = np.zeros((m,n),dtype=tp)
    for j in range(n):
        v[:,j] = a[:,j]
        for i in range(j):
            r[i,j] = inner(q[:,i], a[:,j])
            v[:,j] = v[:,j] - r[i,j]*q[:,i]
        r[j,j] = np.linalg.norm(v[:,j])
        if r[j,j] < 32*eps:
            print("lin dep column found with j=%d" % j)
            z = np.random.rand(m)
            v[:,j] = z
            for i in range(j):
                x = inner(q[:,i], z)
                v[:,j] = v[:,j] - x*q[:,i]
            q[:,j] = v[:,j] / linalg.norm(v[:,j])
        else:
            q[:,j] = v[:,j] / r[j,j]
    return q, r


if __name__ == "__main__":
    import doctest
    doctest.testmod()
