from numpy import *
def qr_it(a):
    n,_ = shape(a)
    t = a.copy()
    qt = eye(n)
    for i in range(10):
        q,r = linalg.qr(t)
        t = r.dot(q)
        # this works to get the evs but we don't want to...
        qt = qt.dot(q)
    return qt,t

