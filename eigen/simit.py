"""
Simultaneous iteration algorithm
Takes a matrix Q and a parameter p denoting the last column
of A considered for computing the eigenspace

>>> A = array([[ 1.079,  1.105,  0.559],
...            [ 1.105,  0.725,  0.988],
...            [ 0.559,  0.988,  0.732]])
>>> Q,L,iters,err = simit(A,3, details=True)
>>> iters < 200
True
>>> err < 1.0e-15
True
"""


from numpy import *

def simit(A, p, details=False):
    n,_ = shape(A)
    t = random.rand(n,p)
    Q,R = linalg.qr(t)
    err = 1.0
    iters = 0
    while err > 1.0e-15 and iters < 1000:
        z = A.dot(Q)
        Q,R = linalg.qr(z)
        err = linalg.norm(A.dot(Q) - z)
        iters += 1
    if details:
        return Q, Q.T.dot(A).dot(Q), iters, err
    else:
        return Q, Q.T.dot(A).dot(Q)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
