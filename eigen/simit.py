"""

Simultaneous iteration algorithm
"""

from numpy import *

def simit(a, p, details=False):
    n,_ = shape(a)
    t = random.rand(n,p)
    q,r = linalg.qr(t)
    err = 1.0
    iters = 0
    #while err > 1.0e-15 and iters < 1000:
    for k in range(50):
        z = a.dot(q)
        q,r = linalg.qr(z)
    return q, q.T.dot(a).dot(q)

