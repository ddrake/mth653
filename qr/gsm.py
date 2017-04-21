import numpy as np

"""
QR Factorization via Modified Gram Schmidt 

Example 1: A is a complex 3x2 matrix with full column rank
>>> A = np.array([[1 + 2.1j, 2.1 + 3.1j],[2.7 + .1j, 3.1 + .2j],[4.1, 2.3j]])
>>> Q,R = gsm(A)
>>> np.linalg.norm(Q.dot(R) - A) < eps
True
>>> np.abs(inner(Q[:,0],Q[:,1])) < 2.*eps
True
>>> np.abs(inner(Q[:,1],Q[:,1]) - 1.) < eps
True

Example 2: A is a 3x3 real matrix with linearly dependent columns
>>> A = np.array([1.,2.,3.])[:,np.newaxis]
>>> A = A.dot(A.T)
>>> Q,R = gsm(A)
>>> np.linalg.norm(Q.dot(R) - A) < 4*eps
True
>>> np.abs(inner(Q[:,0],Q[:,1])) < 4*eps
True
>>> np.abs(inner(Q[:,1],Q[:,1]) - 1.) < 2*eps
True

Example 3: A is a 3x3 real matrix with one column of zeros
>>> A = np.array([[1,2,3],[0,0,0],[3,2,1]]).T+0.0
>>> Q,R = gsm(A)
>>> np.linalg.norm(Q.dot(R) - A) < 4*eps
True
>>> np.abs(inner(Q[:,0],Q[:,1])) < 2*eps
True
>>> np.abs(inner(Q[:,1],Q[:,1]) - 1.) < 2*eps
True
"""

eps = 2**-52

def gsm(a,dbg=False):
    """modified gram-schmidt (reduced)
       takes a complex matrix of any shape and computes its
       QR factorization using the modified Gram-Schmidt factorization
       Handles the case; input matrix doesn't have full Column rank 
    """ 
    m,n = np.shape(a)
    tp = a.dtype
    r = np.zeros((n,n),dtype=tp)
    q = np.zeros((m,n),dtype=tp)
    v = a.copy()
    for i in range(n):
        r[i,i] = np.linalg.norm(v[:,i])
        if r[i,i] < 2*eps:
            if dbg: print("lin dep column found with i=%d" % i)
            q[:,i] = random_orthonormal(q,i)
        else:
            q[:,i] = v[:,i] / r[i,i]
        for j in range(i+1,n):
            r[i,j] = inner(q[:,i],v[:,j])
            v[:,j] -= r[i,j]*q[:,i]
    return q, r

def random_orthonormal(q,i):
    """get a random unit length vector orthonormal to the 
       first i-1 columns of q
    """
    m,n=np.shape(q)
    z = np.random.rand(m)
    v = z[:]
    for j in range(i):
        x = inner(q[:,j],v)
        v -= x*q[:,j]
    return v / np.linalg.norm(v)

def inner(v,w):
    """return the inner product of two vectors
       using the convention of conjugating the first argument
    """
    return v.conj().dot(w)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
