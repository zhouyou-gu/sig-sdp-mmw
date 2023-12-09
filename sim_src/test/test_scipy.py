import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.mmwm_scipy import mmwm_scipy
from sim_src.env.env import env
from sim_src.util import GLOBAL_PROF_ENABLER

print(np.asarray(5).view())

# GLOBAL_PROF_ENABLER.DISABLE()
e = env(cell_size=15,seed=0)
a = mmwm_scipy(e.n_sta,e.min_sinr)
a.set_st(e.rxpr)
# CC = a.run_fc_main(10., num_iterations=100)
# time.sleep(3)

def rank1skretch(A):
    d=1
    randv = np.random.randn(A.shape[0],d)/math.sqrt(float(d))
    randv = randv/np.linalg.norm(randv,axis=0)
    ret = scipy.sparse.linalg.expm_multiply(A.copy(),randv)
    return ret

# CC = a.run_fc_exphalffc_plug_indp(10., num_iterations=100,exphalffc=rank1skretch)
# time.sleep(3)
CC = a.run_fc_exphalffc_plug(26, num_iterations=400,exphalffc=rank1skretch)


import matplotlib.pyplot as plt
data = np.convolve(a.LOGGED_NP_DATA["pct"][:,3], np.ones(50)/50, mode='valid')

plt.plot(a.LOGGED_NP_DATA["pct"][:data.size,0],data)
# print(a.MOVING_AVERAGE_DICT["pct"])
print(data.shape)
plt.show()
