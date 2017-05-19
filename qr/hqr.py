from numpy import *

def inner(v,w):
    return sum(v.conj() * w)

def qr(a):
    (m,n) = shape(a)
    v = zeros((m,n))
    for k in range(n):
        x = a[k:m,k].copy()
        x[0] += sign(x[0])*linalg.norm(x)
        vk = x / linalg.norm(x)
        v[k:m,k] = vk
        a[k:m,k:n] -= 2*vk[:,newaxis].dot(vk.conj().dot(a[k:m,k:n])[newaxis,:])
    return v,a


