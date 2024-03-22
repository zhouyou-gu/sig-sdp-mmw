import os

import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.cvxpy_sdp import cvxpy_sdp
from sim_src.env.env import env
from sim_src.util import GLOBAL_PROF_ENABLER, plot_a_array

GLOBAL_PROF_ENABLER.DISABLE()
e = env(cell_size=5,seed=0)
a = cvxpy_sdp(e.n_sta,e.min_sinr)
a.set_st(e.rxpr_hi)
a._debug(True,1)
