"""  
Househlolder Triangularization
This algorithm uses a projection idea to rotate vectors
(from each diagonal element down) to align with a coordinate axis,
so the length (norm) is unchanged, but zeros are produced below the 
diagonal element resulting in an upper triangular matrix.
Each rotation is a multiplication by an orthogonal matrix
Q can be produced if needed using get_q(v).  But often it is enough to
compute Q*b or Qx so methods are provided for those operations.
Most production QR algorithms use this approach

>>> a = array([[-2.49952922, -1.03751133, -1.21085266, -1.24115004],
...            [-0.80586535,  1.31580494, -0.25698657, -0.09300651],
...            [-0.31995424, -0.36042199, -1.18012579, -0.12960803],
...            [-0.51933797, -0.29275398,  0.0953217 , -0.32097443],
...            [-0.66819142,  0.18847333, -0.43734184, -0.30990116]])
>>> v,r = hqr(a)
>>> linalg.norm(tril(r,-1),'f') < 1.0e-15
True
>>> q = makeq(v)
>>> linalg.norm(a - q.dot(r),'f') < 1.0e-15
True
"""

from numpy import *

def inner(v,w):
    return sum(v.conj() * w)

def hqr(ao):
    """ 
    Given a matrix, compute the Householder triangularization
    """
    a = ao.copy()
    m,n = shape(a)
    v = zeros((m,n), dtype=complex)
    for k in range(n):
        x = a[k:,k].copy()
        x[0] += sign(x[0])*linalg.norm(x,2)
        vk = x[:,None] / linalg.norm(x,2)
        v[k:,k] = vk[:,0]
        # subtract (m-k+1)x(m-k+1) rank 1 submatrix from a
        a[k:,k:] -= 2*vk.dot(vk.conj().T.dot(a[k:,k:]))
    return v,a

def qhb(v,bo):
    """ 
    Given the matrix v returned from hqr and an m-vector b
    compute the product Q* b
    """
    _,n = shape(v)
    b = bo.copy()
    for k in range(n):
        b[k:] -= 2*v[k:,k]*(v[k:,k].conj().dot(b[k:]))
    return b

def qx(v,xo):
    """
    Given the matrix v returned from hqr and a vector 
    x with m rows, compute the product Q x
    """
    _,n = shape(v)
    x = xo.copy()
    for k in range(n-1,-1,-1):
        x[k:] -= 2*v[k:,k]*(v[k:,k].conj().dot(x[k:]))
    return x

def makeq1(v):
    """
    Given the matrix v returned from hqr, form the matrix Q by 
    computing Q* I and taking the conjugate transpose.
    """
    m,_ = shape(v)
    Q = zeros((m,m),dtype=complex)
    I = eye(m,dtype=complex)
    for i in range(m):
        Q[:,i]=qhb(v,I[:,i])
    return Q.conj().T

def makeq(v):
    """
    Given the matrix v returned from hqr, form the matrix Q by 
    computing Q I 
    """
    m,_ = shape(v)
    Q = zeros((m,m),dtype=complex)
    I = eye(m,dtype=complex)
    for i in range(m):
        Q[:,i]=qx(v,I[:,i])
    return Q


def bslv(r,c):
    """
    r should be upper triangular mxn with nonzero diagonal entries 
    and c mx1.  Solves rx = c by back substitution
    """
    _,n = shape(r)
    x = zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = (c[i]-r[i,i+1:].dot(x[i+1:])) / r[i,i]
    return x

def slv(a,bo):
    v,r = hqr(a)
    c = qhb(v,b)
    x = bslv(r,c)
    return x

if __name__ == "__main__":
    import doctest
    doctest.testmod()
