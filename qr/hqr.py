from numpy import *

def inner(v,w):
    return sum(v.conj() * w)

def qr(a):
    (m,n) = shape(a)
    v = zeros((m,n))
    for k in range(n):
        print("k=%d" % k)
        x = a[k:m,k]
        x[0] += sign(x[0])*linalg.norm(x)
        vk = x / linalg.norm(x)
        v[k:m,k] = vk
        print(vk)
        a[k:m,k:n] -= 2*vk[:,newaxis].dot(vk.conj().dot(a[k:m,k:n])[newaxis,:])
        print(a)
    return v
