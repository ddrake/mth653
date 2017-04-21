import gsm
import gs

from numpy import *

from matplotlib import pyplot as plt

[U,X] = linalg.qr(random.rand(80,80))
[V,X] = linalg.qr(random.rand(80,80))

S = diag(2.**(-1-array(range(80))))

A = U.dot(S).dot(V)

QC,RC = gs.gs(A)
QM,RM = gsm.gsm(A)

plt.semilogy(range(80), diag(RC),'bs',diag(RM),'ro')
plt.show()

