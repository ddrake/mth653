"""  
Househlolder Triangularization
This algorithm uses a projection idea to rotate vectors
(from each diagonal element down) to align with a coordinate axis,
so the length (norm) is unchanged, but zeros are produced below the 
diagonal element resulting in an upper triangular matrix.
Each rotation is a multiplication by an orthogonal matrix
Q can be produced if needed using get_q(V).  But often it is enough to
compute Q*b or Qx so methods are provided for those operations.
Most production QR algorithms use this approach

>>> A = array([[-2.49952922, -1.03751133, -1.21085266, -1.24115004],
...            [-0.80586535,  1.31580494, -0.25698657, -0.09300651],
...            [-0.31995424, -0.36042199, -1.18012579, -0.12960803],
...            [-0.51933797, -0.29275398,  0.0953217 , -0.32097443],
...            [-0.66819142,  0.18847333, -0.43734184, -0.30990116]])
>>> V,R = hqr(A)
>>> linalg.norm(tril(R,-1),'f') < 1.0e-15
True
>>> Q = makeq(V)
>>> linalg.norm(A - Q.dot(R),'f') < 1.0e-15
True
>>> A = array([[ 0.04+0.46j,  0.02+0.61j,  0.55+0.87j],
...            [ 0.94+0.69j,  1.00+0.94j,  0.17+0.24j],
...            [ 0.07+0.52j,  0.48+0.48j,  0.07+0.49j]])
>>> V,R = hqr(A)
>>> linalg.norm(tril(R,-1),'f') < 1.0e-15
True
>>> Q = makeq(V)
>>> linalg.norm(A - Q.dot(R),'f') < 2.0e-15
True
"""

from numpy import *

def inner(v,w):
    return v.conj().T.dot(w)

def sgn(x):
    """
    To handle the complex case, we need to 
    use this definition of the sign(A) instead of numpy.sign,
    which just takes the sign of the real part.
    """
    return 1 if x == 0 else x/abs(x)

def hqr(Ao):
    """ 
    Given a matrix, compute the Householder triangularization
    """
    A = Ao.copy()
    tp = A.dtype
    m,n = shape(A)
    V = zeros((m,n), dtype=tp)
    for k in range(n):
        x = A[k:,k].copy()
        nx = linalg.norm(x,2)
        x[0] += sgn(x[0])*linalg.norm(x,2)
        vk = x[:,None] / linalg.norm(x,2) #norm of changed x
        V[k:,k] = vk[:,0]
        # subtract an (m-k+1)x(m-k+1) rank 1 submatrix from A
        A[k:,k:] -= 2*vk.dot(inner(vk, A[k:,k:]))
    return V,A

def qhb(V,bo):
    """ 
    Given the matrix V returned from hqr and an m-vector b
    compute the product Q* b
    """
    _,n = shape(V)
    b = bo.copy()
    for k in range(n):
        b[k:] -= 2*V[k:,k]*(V[k:,k].conj().dot(b[k:]))
    return b

def qx(V,xo):
    """
    Given the matrix V returned from hqr and a vector 
    x with m rows, compute the product Q x
    """
    _,n = shape(V)
    x = xo.copy()
    for k in range(n-1,-1,-1):
        #x[k:] -= 2*V[k:,k]*(V[k:,k].conj().dot(x[k:]))
        x[k:] -= 2*V[k:,k]*(inner(V[k:,k], x[k:]))
    return x

def makeq1(V):
    """
    Given the matrix V returned from hqr, form the matrix Q by 
    computing Q* I and taking the conjugate transpose.
    """
    tp = V.dtype
    m,_ = shape(V)
    Q = zeros((m,m),dtype=tp)
    I = eye(m,dtype=tp)
    for i in range(m):
        Q[:,i]=qhb(V,I[:,i])
    return Q.conj().T

def makeq(V):
    """
    Given the matrix V returned from hqr, form the matrix Q by 
    computing Q I 
    """
    tp = V.dtype
    m,_ = shape(V)
    Q = zeros((m,m),dtype=tp)
    I = eye(m,dtype=tp)
    for i in range(m):
        Q[:,i]=qx(V,I[:,i])
    return Q


def bslv(R,c):
    """
    R should be upper triangular mxn with nonzero diagonal entries 
    and c mx1.  Solves rx = c by back substitution
    """
    tp = R.dtype
    _,n = shape(R)
    x = zeros(n,dtype=tp)
    for i in range(n-1,-1,-1):
        x[i] = (c[i]-R[i,i+1:].dot(x[i+1:])) / R[i,i]
    return x

def slv(A,b):
    V,R = hqr(A)
    c = qhb(V,b)
    x = bslv(R,c)
    return x

if __name__ == "__main__":
    import doctest
    doctest.testmod()
