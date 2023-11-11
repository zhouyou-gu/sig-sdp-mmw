import time

import scipy
import scipy.sparse as ss
import scipy.sparse.linalg as sl
import numpy as np
N = 100

start = time.time()
for i in range(1000):
    b = np.random.binomial(1,10./N,size=(N,N))
    b_symm = (b + b.T)/2
    np.fill_diagonal(b_symm, 0)
    scipy.linalg.expm(b_symm)
end = time.time()
print(end - start)

start = time.time()
for i in range(1000):
    b = np.random.binomial(1,10./N,size=(N,N))
    b_symm = (b + b.T)/2
    np.fill_diagonal(b_symm, 0)
    scipy.linalg.expm(b_symm)
end = time.time()
print(end - start)
