import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.mmwm_scipy import mmwm_scipy
from sim_src.env.env import env
e = env(cell_size=5,seed=int(time.time()))
a = mmwm_scipy(e.n_sta,e.min_sinr)
a.set_st(e.rxpr)
CC = a.run_fc_main(10., num_iterations=100)
time.sleep(3)

def rank1skretch(A):
    d=1
    randv = np.random.randn(A.shape[0],d)/math.sqrt(float(d))
    ret = scipy.sparse.linalg.expm_multiply(A.copy(),randv)
    return ret

CC = a.run_fc_exphalffc_plug_indp(10., num_iterations=100,exphalffc=rank1skretch)
time.sleep(3)
CC = a.run_fc_exphalffc_plug(10., num_iterations=100,exphalffc=rank1skretch)

