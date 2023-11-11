import time

import scipy
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import numpy as np
import math

N = 100

start = time.time()
r = 1
for i in range(1000):
    b = np.random.binomial(1,0.1,size=(N,N))
    b_symm = (b + b.T)/2
    np.fill_diagonal(b_symm, 0)
    res = scipy.linalg.expm(b_symm)
    print("expm",res/res.trace()*N)
    A = ss.csr_matrix(b_symm)/2.
    res = np.zeros((N,r))
    randv = np.random.randn(N,r)
    for d in range(100):
        k = d+1
        randv = (A * randv) / k
        res += randv
    res = res.dot(res.transpose())
    res = res/res.trace()*N
    print("appro",res)

end = time.time()
print(end - start)
