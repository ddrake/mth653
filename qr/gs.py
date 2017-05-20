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
>>> np.abs(inner(Q[:,1],Q[:,1]) - 1.) < 2.*eps
True

Example 2: A is a 3x3 real matrix with linearly dependent columns
>>> A = np.array([[1.,2.,3.]])
>>> A = A.T.dot(A)
>>> Q,R = gs(A)
>>> np.linalg.norm(Q.dot(R) - A) < 64*eps
True
>>> np.abs(inner(Q[:,0],Q[:,1])) < 8*eps
True
>>> np.abs(inner(Q[:,1],Q[:,1]) - 1.) < 8*eps
True

Note: test results may vary for Example 2 due to the random vectors added.
"""

eps = 2**-52

def gs(A,dbg=False):
    """
    Classical Gram-Schmidt factorizaton
    takes an mxn complex matrix and returns a QR factorization
    """
    m,n = np.shape(A)
    tp = A.dtype
    R = np.zeros((n,n),dtype=tp)
    Q = np.zeros((m,n),dtype=tp)
    V = np.zeros((m,n),dtype=tp)
    for j in range(n):
        V[:,j] = A[:,j]
        for i in range(j):
            R[i,j] = inner(Q[:,i], A[:,j])
            V[:,j] -= R[i,j]*Q[:,i]
        R[j,j] = np.linalg.norm(V[:,j])
        if R[j,j] < 32*eps:
            if dbg: print("lin dep column found with j=%d" % j)
            Q[:,j] = random_orthonormal(Q,j)
        else:
            Q[:,j] = V[:,j] / R[j,j]
    return Q, R

def random_orthonormal(Q,i):
    """
    get a random unit length vector orthonormal to the 
    first i-1 columns of Q
    """
    m,n=np.shape(Q)
    z = np.random.rand(m)
    v = z[:]
    for j in range(i):
        x = inner(Q[:,j],v)
        v -= x*Q[:,j]
    return v / np.linalg.norm(v)

def inner(v,w):
    """
    return the inner product of two vectors
    using the convention of conjugating the first argument
    """
    return v.conj().T.dot(w)


def bslv(R,c):
    """
    R should be upper triangular mxn with nonzero diagonal entries 
    and c mx1.  Solves Rx = c by back substitution
    """
    tp = R.dtype
    _,n = shape(R)
    x = zeros(n,dtype=tp)
    for i in range(n-1,-1,-1):
        x[i] = (c[i]-R[i,i+1:].dot(x[i+1:])) / R[i,i]
    return x

def slv(A,b):
    """
    Solve linear system Ax = b
    """
    Q,R = gs(A)
    c = Q.T.dot(b)
    x = bslv(R,c)
    return x

if __name__ == "__main__":
    import doctest
    doctest.testmod()
