import os

import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.mmwm_scipy import mmwm_scipy
from sim_src.env.env import env
from sim_src.util import GLOBAL_PROF_ENABLER, plot_a_array

GLOBAL_PROF_ENABLER.DISABLE()
e = env(cell_size=5,seed=0)
a = mmwm_scipy(e.n_sta,e.min_sinr)
a.set_st(e.rxpr)
a._debug(True,1)


def rankdskretch(A, d=1):
    randv = np.random.randn(A.shape[0],d)/math.sqrt(float(d))
    randv = randv/np.linalg.norm(randv,axis=0)
    ret = scipy.sparse.linalg.expm_multiply(A.copy(),randv)
    return ret

CC = a.run_fc_exphalffc_plug(20, num_iterations=int(1/a.ETA**2)*int(math.log(e.n_sta)),exphalffc=rankdskretch)

plot_a_array(a.LOGGED_NP_DATA["pct"][:,3],name="pct",script_file=__file__,save_path=os.path.dirname(os.path.realpath(__file__)))
