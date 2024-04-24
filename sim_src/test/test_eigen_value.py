import os

import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.cvxpy_sdp import cvxpy_sdp
from sim_src.env.env import env
from sim_src.util import GLOBAL_PROF_ENABLER, plot_a_array


for K in range(5,10):
    M1 = np.zeros((K,K))
    np.fill_diagonal(M1,-1./K)
    M1[0,0] += 1
    print(np.linalg.eig(M1))

# for Z in range(2,10):
#     for K in range(10,20):
#         M2 = np.zeros((K,K))
#         np.fill_diagonal(M2,-1./K)
#         M2[0,1] = -(Z-1)/2.
#         M2[1,0] = -(Z-1)/2.
#         print(Z,K, -1./K,-(Z-1)/2.,np.linalg.eig(M2))

# p=0.2
# for K in range(5,10):
#     M1 = np.random.choice([0, 1], size=(K,K), p=[1-p, p])
#     M1 = np.triu(M1,1)
#
#     M2 = np.random.randn
#     print(M1)
